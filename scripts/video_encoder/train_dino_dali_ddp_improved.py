#!/usr/bin/env python3
# Fine-tune DINO-style on small video-frame dataset using DALI + DDP.
# Key changes for ~10k-30k frames: stronger multi-crop, teacher schedules, tiny LR on backbone.

import os, math, random, argparse, json
from datetime import timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import timm
import matplotlib.pyplot as plt

from nvidia.dali import fn, types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# ---------------- Utils ----------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            timeout=timedelta(hours=2))
    return local_rank, rank, world_size

def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def is_main(rank): return rank == 0

# ---------------- Args/Config ----------------
def get_args():
    p = argparse.ArgumentParser(description="DINO small-data fine-tune with DALI + DDP")
    # Data
    p.add_argument("--manifest", type=str, required=True,
                   help="CSV with columns: split, clip_id, frame_path, t_sec")
    p.add_argument("--split", type=str, default="train", help="Split to filter (train/val/test)")
    p.add_argument("--frames_per_clip", type=int, default=4, help="Uniformly sample N frames/clip")
    p.add_argument("--max_clips", type=int, default=20000, help="Optional cap on number of clips")
    # Model
    p.add_argument("--model_name", type=str, default="vit_base_patch14_dinov2",
                   help="Backbone name in timm (e.g., vit_base_patch14_dinov2)")
    # Multi-crop (separate sizes)
    p.add_argument("--global_crops", type=int, default=2)
    p.add_argument("--local_crops",  type=int, default=8)   # stronger local count
    p.add_argument("--global_size",  type=int, default=224)
    p.add_argument("--local_size",   type=int, default=96)
    p.add_argument("--global_area", type=float, nargs=2, default=(0.40, 1.00))
    p.add_argument("--local_area",  type=float, nargs=2, default=(0.05, 0.40))
    p.add_argument("--aspect",      type=float, nargs=2, default=(0.75, 1.33))
    # DINO loss / temps / momentum
    p.add_argument("--student_t", type=float, default=0.1)
    p.add_argument("--teacher_t_end", type=float, default=0.07)      # end temp after warmup
    p.add_argument("--teacher_t_warmup", type=float, default=0.04)   # start temp
    p.add_argument("--teacher_warmup_epochs", type=int, default=5)   # warmup len
    p.add_argument("--teacher_momentum_start", type=float, default=0.99)
    p.add_argument("--teacher_momentum_end",   type=float, default=0.996)
    p.add_argument("--center_momentum", type=float, default=0.9)
    p.add_argument("--head_out_dim",    type=int, default=8192)
    p.add_argument("--head_hidden_dim", type=int, default=1024)
    # Train
    p.add_argument("--batch_size", type=int, default=8, help="Per-rank micro-batch")
    p.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)        # fewer epochs for small data
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--patience", type=int, default=5)       # early stopping
    # I/O
    p.add_argument("--out_dir", type=str, default="./dino_finetune")
    p.add_argument("--save_every_epochs", type=int, default=1)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# --------------- Data listing ---------------
def build_filelist(manifest_path, split_filter, frames_per_clip, max_clips):
    df = pd.read_csv(manifest_path)
    if split_filter is not None:
        df = df[df["split"] == split_filter].copy()
    g = df.groupby("clip_id")["frame_path"].apply(list)
    clip_to_frames = {cid: paths for cid, paths in g.items() if len(paths) > 0}
    clip_ids_sorted = list(clip_to_frames.keys())
    if max_clips is not None:
        clip_ids_sorted = clip_ids_sorted[:max_clips]
    files, clip_labels = [], []
    for idx, cid in enumerate(clip_ids_sorted):
        paths = clip_to_frames[cid]
        if len(paths) > frames_per_clip:
            idxs = np.linspace(0, len(paths)-1, frames_per_clip, dtype=int)
            sel = [paths[i] for i in idxs]
        else:
            sel = paths
        for p in sel:
            files.append(p)
            clip_labels.append(idx)
    return files, clip_labels

# --------------- DALI pipeline ---------------
IM_MEAN_255 = [0.485*255, 0.456*255, 0.406*255]
IM_STD_255  = [0.229*255, 0.224*255, 0.225*255]

@pipeline_def
def multicrop_pipeline(files, labels,
                       global_size, local_size,
                       g_area, l_area, aspect,
                       shard_id, num_shards):
    jpegs, lbls = fn.readers.file(
        files=files, labels=labels, name="Reader",
        random_shuffle=True, shard_id=shard_id, num_shards=num_shards, stick_to_shard=True
    )
    imgs = fn.decoders.image(jpegs, device="mixed")

    mirror = fn.random.coin_flip(probability=0.5)
    b = fn.random.uniform(range=[0.8, 1.2])
    c = fn.random.uniform(range=[0.8, 1.2])
    s = fn.random.uniform(range=[0.8, 1.2])
    h = fn.random.uniform(range=[-0.1, 0.1])
    p_gray  = fn.random.coin_flip(probability=0.1)
    p_grayf = fn.cast(p_gray, dtype=types.FLOAT)
    sig     = fn.random.uniform(range=[0.1, 1.0])

    def aug_branch(img, area_rng, out_size):
        x = fn.random_resized_crop(
            img, size=out_size,
            random_area=area_rng, random_aspect_ratio=aspect,
            device="gpu",
        )
        x = fn.color_twist(x, brightness=b, contrast=c, saturation=s, hue=h)
        x_gray = fn.color_twist(x, saturation=0.0)
        x = x_gray * p_grayf + x * (1.0 - p_grayf)
        x = fn.gaussian_blur(x, sigma=sig)
        x = fn.crop_mirror_normalize(
            x, dtype=types.FLOAT, output_layout="CHW",
            mean=IM_MEAN_255, std=IM_STD_255, mirror=mirror,
        )
        return x

    g_crops = [aug_branch(imgs, g_area, global_size) for _ in range(CONFIG["GLOBAL_CROPS"])]
    l_crops = [aug_branch(imgs, l_area,  local_size) for _ in range(CONFIG["LOCAL_CROPS"])]
    return (*g_crops, *l_crops, lbls)

# --------------- Model / Loss ---------------
class DINOHead(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, hidden_dim:int=2048, bottleneck_dim:int=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        lin = nn.Linear(bottleneck_dim, out_dim, bias=False)
        try:
            from torch.nn.utils.parametrizations import weight_norm as pn_weight_norm
            self.last = pn_weight_norm(lin)
            with torch.no_grad():
                wn = self.last.parametrizations.weight[0]
                if hasattr(wn, "g"): wn.g.fill_(1.0)
        except Exception:
            self.last = nn.utils.weight_norm(lin)
            with torch.no_grad():
                if hasattr(self.last, "weight_g"): self.last.weight_g.fill_(1.0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last(x)

class DINOLoss(nn.Module):
    def __init__(self, out_dim:int, warmup_teacher_temp:float, teacher_temp_end:float,
                 warmup_epochs:int, nepochs:int, student_temp:float, center_momentum:float=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # teacher temperature schedule: warmup → flat
        self.teacher_temp_schedule = np.concatenate([
            np.linspace(warmup_teacher_temp, teacher_temp_end, max(1, warmup_epochs)),
            np.full(max(0, nepochs - warmup_epochs), teacher_temp_end),
        ])

    def forward(self, student_outs:List[torch.Tensor], teacher_outs:List[torch.Tensor], epoch:int):
        Tt = float(self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule)-1)])
        t_probs = [F.softmax((t - self.center) / Tt, dim=-1) for t in teacher_outs]
        s_logs  = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_outs]
        total, n = 0.0, 0
        for t in t_probs:
            for s in s_logs:
                total += torch.mean(torch.sum(-t * s, dim=-1)); n += 1
        loss = total / max(n, 1)
        with torch.no_grad():
            batch_center = torch.cat(teacher_outs, dim=0).mean(dim=0, keepdim=True)
            self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))
        return loss

class DinoBackbone(nn.Module):
    def __init__(self, backbone_name:str, img_size:int):
        super().__init__()
        # pretrained=True => fine-tuning, not from-scratch
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, img_size=img_size)
        self.feat_dim = self.backbone.num_features
    def forward(self, x): return self.backbone(x)

class StudentTeacher(nn.Module):
    def __init__(self, model_name:str, img_size_g:int, img_size_l:int, head_out:int, head_hidden:int):
        super().__init__()
        # Student backbone is defined by global-size; ViTs are size-agnostic for features
        self.student_backbone = DinoBackbone(model_name, img_size_g)
        self.teacher_backbone = DinoBackbone(model_name, img_size_g)
        self.student_head = DINOHead(self.student_backbone.feat_dim, head_out, hidden_dim=head_hidden)
        self.teacher_head = DINOHead(self.teacher_backbone.feat_dim, head_out, hidden_dim=head_hidden)
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in list(self.teacher_backbone.parameters()) + list(self.teacher_head.parameters()):
            p.requires_grad = False
        # Enable grad checkpointing if supported
        try:
            from timm.models.vision_transformer import set_grad_checkpointing
            set_grad_checkpointing(self.student_backbone.backbone, True)
        except Exception:
            pass

def cosine_interp(start: float, end: float, t: float) -> float:
    # t in [0,1]
    return start + (end - start) * (1 - math.cos(math.pi * t)) * 0.5

def cosine_momentum(epoch:int, nepochs:int, start:float, end:float):
    t = 0.0 if nepochs <= 1 else epoch / float(nepochs - 1)
    return cosine_interp(start, end, t)

# --------------- Main ---------------
def main():
    args = get_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    local_rank, rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)
    set_seed(args.seed + rank)

    if is_main(rank):
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        print(json.dumps(vars(args), indent=2))

    # Build file list (all ranks; sharded in DALI)
    files, labels = build_filelist(args.manifest, args.split, args.frames_per_clip, args.max_clips)
    if is_main(rank):
        print(f"Total frames: {len(files)} | Clips: ~{len(set(labels))} | World Size: {world_size}")

    # Global config values needed inside pipeline_def
    global CONFIG
    CONFIG = {"GLOBAL_CROPS": args.global_crops, "LOCAL_CROPS": args.local_crops}

    # DALI per-rank pipeline/iterator
    CROP_NAMES = [*(f"g{i}" for i in range(args.global_crops)),
                  *(f"l{i}" for i in range(args.local_crops)), "label"]

    pipe = multicrop_pipeline(
        files=files, labels=labels,
        batch_size=args.batch_size,
        num_threads=args.num_workers,
        device_id=local_rank,
        seed=args.seed + rank,
        global_size=args.global_size, local_size=args.local_size,
        g_area=tuple(args.global_area),
        l_area=tuple(args.local_area),
        aspect=tuple(args.aspect),
        shard_id=rank, num_shards=world_size,
        prefetch_queue_depth=2
    )
    pipe.build()
    dali_loader = DALIGenericIterator([pipe], output_map=CROP_NAMES, reader_name="Reader", auto_reset=True)

    # Model / opt / loss
    net = StudentTeacher(args.model_name, args.global_size, args.local_size, args.head_out_dim, args.head_hidden_dim).to(device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=False)
    module = net.module

    # Optional: freeze first 4 blocks for tiny data
    try:
        for i, blk in enumerate(module.student_backbone.backbone.blocks):
            if i < 4:
                for p in blk.parameters():
                    p.requires_grad = False
    except Exception:
        pass

    backbone_params = [p for p in module.student_backbone.parameters() if p.requires_grad]
    head_params     = list(module.student_head.parameters())

    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 5e-5},   # small LR for backbone
            {"params": head_params,     "lr": 1e-3},   # larger LR for head
        ],
        weight_decay=args.weight_decay
    )
    # --- LR schedule: warmup + cosine ---
    warmup_epochs = 10
    def cosine_lr(epoch, base_lr):
        # cosine from epoch warmup_epochs -> args.epochs
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = (epoch - warmup_epochs) / max(1, (args.epochs - warmup_epochs))
        return 0.5 * (1 + math.cos(math.pi * t))

    # store base LRs to scale each epoch
    BASE_LRS = [g["lr"] for g in opt.param_groups]

    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    dino_loss = DINOLoss(
        out_dim=args.head_out_dim,
        warmup_teacher_temp=args.teacher_t_warmup,
        teacher_temp_end=args.teacher_t_end,
        warmup_epochs=args.teacher_warmup_epochs,
        nepochs=args.epochs,
        student_temp=args.student_t,
        center_momentum=args.center_momentum
    ).to(device)

    history = {"epoch": [], "loss": [], "teacher_H": [], "student_H": [], "center_norm": []}
    best_loss = float("inf")
    no_improve = 0
    patience = args.patience

    for epoch in range(args.epochs):
        for g, base in zip(opt.param_groups, BASE_LRS):
            g["lr"] = base * cosine_lr(epoch, base)
        
        net.train()
        module.teacher_backbone.eval(); module.teacher_head.eval()

        running_sum = 0.0
        step_count = 0.0
        teacher_H_sum = 0.0
        student_H_sum = 0.0

        m = cosine_momentum(epoch, args.epochs, args.teacher_momentum_start, args.teacher_momentum_end)

        pbar = tqdm(dali_loader, disable=not is_main(rank),
                    desc=f"[DINO+DALI][E{epoch+1}/{args.epochs}]")
        accum = 0

        for batch in pbar:
            b = batch[0]
            per_crop_batches = [b[name].contiguous() for name in CROP_NAMES[:-1]]

            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=args.amp):
                student_outs = []
                for x in per_crop_batches:
                    feat_s = module.student_backbone(x)
                    logits_s = module.student_head(feat_s)
                    student_outs.append(logits_s)

                with torch.no_grad():
                    teacher_outs = []
                    # teacher sees only global crops
                    for i in range(args.global_crops):
                        xg = per_crop_batches[i]
                        feat_t = module.teacher_backbone(xg)
                        logits_t = module.teacher_head(feat_t)
                        teacher_outs.append(logits_t)

                loss = dino_loss(student_outs, teacher_outs, epoch)

                # diagnostics
                with torch.no_grad():
                    Tt = float(dino_loss.teacher_temp_schedule[min(epoch, len(dino_loss.teacher_temp_schedule)-1)])
                    t_probs = [torch.softmax((t - dino_loss.center) / Tt, dim=-1) for t in teacher_outs]
                    s_probs = [torch.softmax(s / args.student_t, dim=-1) for s in student_outs]
                    def mean_entropy(prob_list):
                        return torch.stack([-(p * (p + 1e-12).log()).sum(dim=-1).mean() for p in prob_list]).mean().item()
                    teacher_H_sum += mean_entropy(t_probs)
                    student_H_sum += mean_entropy(s_probs)

            # scale & backward with accumulation
            opt.zero_grad(set_to_none=True) if accum == 0 else None
            scaler.scale(loss / max(1, args.accum_steps)).backward()
            accum += 1

            if accum >= args.accum_steps:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(module.student_backbone.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(module.student_head.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                accum = 0

                # EMA teacher update
                with torch.no_grad():
                    for ps, pt in zip(module.student_backbone.parameters(), module.teacher_backbone.parameters()):
                        if ps.requires_grad:
                            pt.data.mul_(m).add_(ps.data, alpha=1 - m)
                    for ps, pt in zip(module.student_head.parameters(), module.teacher_head.parameters()):
                        pt.data.mul_(m).add_(ps.data, alpha=1 - m)

            running_sum += float(loss.item())
            step_count += 1.0
            if is_main(rank):
                pbar.set_postfix(loss=f"{loss.item():.4f}", ema=f"{m:.4f}",
                                 tH=f"{teacher_H_sum/max(1,step_count):.3f}",
                                 sH=f"{student_H_sum/max(1,step_count):.3f}")

        # flush pending grads if accumulation didn't align perfectly
        if accum > 0:
            scaler.step(opt)
            scaler.update()
            accum = 0
            with torch.no_grad():
                for ps, pt in zip(module.student_backbone.parameters(), module.teacher_backbone.parameters()):
                    if ps.requires_grad:
                        pt.data.mul_(m).add_(ps.data, alpha=1 - m)
                for ps, pt in zip(module.student_head.parameters(), module.teacher_head.parameters()):
                    pt.data.mul_(m).add_(ps.data, alpha=1 - m)

        # reduce metrics across ranks
        t = torch.tensor([running_sum, step_count, teacher_H_sum, student_H_sum], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        running_sum, step_count, teacher_H_sum, student_H_sum = [float(x) for x in t.tolist()]
        avg = running_sum / max(1.0, step_count)
        teacher_H = teacher_H_sum / max(1.0, step_count)
        student_H = student_H_sum / max(1.0, step_count)
        center_norm = float(dino_loss.center.norm().item())

        if is_main(rank):
            print(f"[epoch {epoch+1}] loss={avg:.4f}  |  teacher_H={teacher_H:.3f}  student_H={student_H:.3f}  center_norm={center_norm:.2f}")
            history["epoch"].append(epoch + 1)
            history["loss"].append(avg)
            history["teacher_H"].append(teacher_H)
            history["student_H"].append(student_H)
            history["center_norm"].append(center_norm)

            # periodic checkpoint
            if (epoch + 1) % args.save_every_epochs == 0:
                ckpt_path = Path(args.out_dir) / f"dino_{args.model_name}_e{epoch+1}.pt"
                torch.save({
                    "student_backbone": module.student_backbone.state_dict(),
                    "student_head": module.student_head.state_dict(),
                    "config": vars(args)
                }, ckpt_path)
                print(f"✓ saved {ckpt_path}")

            # best/early stopping
            if avg < best_loss - 1e-6:
                best_loss = avg
                no_improve = 0
                best_path = Path(args.out_dir) / f"best_dino_{args.model_name}.pt"
                torch.save({
                    "student_backbone": module.student_backbone.state_dict(),
                    "student_head": module.student_head.state_dict(),
                    "config": vars(args)
                }, best_path)
                print(f"★ new best ({best_loss:.4f}) → {best_path}")
            else:
                no_improve += 1

        # broadcast early-stop flag
        stop = torch.tensor([1 if (is_main(rank) and no_improve >= patience) else 0], device=device)
        dist.broadcast(stop, src=0)
        if stop.item() == 1:
            if is_main(rank):
                print(f"⏹ Early stopping after {epoch+1} epochs (no improvement for {patience}).")
            break

    if is_main(rank):
        # save metrics + plots
        out_dir = Path(args.out_dir)
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(out_dir / "training_history.csv", index=False)

        plt.figure(); plt.plot(history["epoch"], history["loss"])
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("DINO Train Loss"); plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png"); plt.close()

        plt.figure(); plt.plot(history["epoch"], history["teacher_H"])
        plt.xlabel("Epoch"); plt.ylabel("Entropy (nats)"); plt.title("Teacher Entropy"); plt.tight_layout()
        plt.savefig(out_dir / "teacher_entropy.png"); plt.close()

        plt.figure(); plt.plot(history["epoch"], history["student_H"])
        plt.xlabel("Epoch"); plt.ylabel("Entropy (nats)"); plt.title("Student Entropy"); plt.tight_layout()
        plt.savefig(out_dir / "student_entropy.png"); plt.close()

        plt.figure(); plt.plot(history["epoch"], history["center_norm"])
        plt.xlabel("Epoch"); plt.ylabel("L2 Norm"); plt.title("Center Vector Norm"); plt.tight_layout()
        plt.savefig(out_dir / "center_norm.png"); plt.close()

        print("Saved metrics/charts to:", out_dir)

    cleanup_ddp()

if __name__ == "__main__":
    # minimal globals used by DALI pipeline
    CONFIG = {"GLOBAL_CROPS": 2, "LOCAL_CROPS": 8}
    main()

