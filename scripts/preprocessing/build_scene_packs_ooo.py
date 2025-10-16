#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, time, pickle, hashlib, argparse, logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from logging.handlers import RotatingFileHandler

# ----------------------
# CLI
# ----------------------
def get_args():
    p = argparse.ArgumentParser(description="Build Social Token Packs from frames + caption (no timestamps).")
    p.add_argument("--frames_root", required=True,
                   help="Parent directory containing <video_stem>/frame_*.jpg")
    p.add_argument("--captions_csv", required=True,
                   help="CSV with columns: video_name,caption")
    p.add_argument("--out_dir", required=True, help="Output directory for packs + manifest/logs")
    p.add_argument("--dino_ckpt", required=True, help="Path to finetuned DINO ViT-B/14 checkpoint")
    p.add_argument("--model_name", default="vit_base_patch14_dinov2.lvd142m",
                   help="timm backbone name (num_classes=0)")
    p.add_argument("--max_global_frames", type=int, default=120,
                   help="Max frames sampled for global vector (downsamples if many frames exist)")
    p.add_argument("--top_k_locals", type=int, default=1000,
                   help="Max noun/verb locals to keep")
    p.add_argument("--pad_frames", type=int, default=2,
                   help="+/- frames around each pseudo-timed token window")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for frame embedding")
    p.add_argument("--limit", type=int, default=0, help="Process at most N clips (0 = all)")
    p.add_argument("--overwrite", action="store_true", help="Rebuild packs even if outputs already exist")
    p.add_argument("--resume", action="store_true", help="(Default) Skip clips with existing outputs")
    p.add_argument("--log_file", default="social_packs_ooo.log", help="Main log filename (created under out_dir)")
    p.add_argument("--device", default="auto", choices=["auto","cuda","cpu"], help="Device selection")
    return p.parse_args()

# ----------------------
# Logging
# ----------------------
def setup_logging(out_dir: str, main_log_name: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    main_log_path = Path(out_dir) / main_log_name
    fail_log_path = main_log_path.with_name(main_log_path.stem + "_failures.log")

    logger = logging.getLogger("social_packs")
    logger.setLevel(logging.INFO); logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = RotatingFileHandler(main_log_path, maxBytes=10_000_000, backupCount=3)
    fh.setFormatter(fmt); fh.setLevel(logging.INFO); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.setLevel(logging.INFO); logger.addHandler(ch)

    fail_logger = logging.getLogger("social_packs.failures")
    fail_logger.setLevel(logging.WARNING); fail_logger.handlers.clear()
    fh_fail = RotatingFileHandler(fail_log_path, maxBytes=5_000_000, backupCount=2)
    fh_fail.setFormatter(fmt); fh_fail.setLevel(logging.WARNING); fail_logger.addHandler(fh_fail)
    return logger, fail_logger

# ----------------------
# Data helpers
# ----------------------
def load_frame_paths(frames_dir: str) -> List[str]:
    # frames are frame_000001.jpg ... frame_000032.jpg
    return sorted(str(p) for p in Path(frames_dir).glob("frame_*.jpg"))

def outputs_exist(clip_id: str, out_dir: str) -> bool:
    key = hashlib.md5(clip_id.encode()).hexdigest()
    g = Path(out_dir) / f"{key}_global.npy"
    l = Path(out_dir) / f"{key}_locals.npz"
    m = Path(out_dir) / f"{key}_meta.pkl"
    return g.exists() and l.exists() and m.exists()

def ensure_manifest(manifest_csv: str):
    if not Path(manifest_csv).exists():
        with open(manifest_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "clip_id","key","global_vec","locals_npz","meta_pkl",
                "n_locals","n_words","frames_dir","caption",
                "status","error","duration_s","device"
            ])

def append_manifest(manifest_csv: str, row: List[Any]):
    with open(manifest_csv, "a", newline="") as f:
        csv.writer(f).writerow(row)

# ----------------------
# POS tagging (caption only)
# ----------------------
import spacy
try:
    _NLP = spacy.load("en_core_web_trf", disable=["ner","parser"])
except Exception:
    _NLP = spacy.load("en_core_web_sm", disable=["ner","parser"])

KEEP_POS = {"NOUN", "PROPN", "VERB", "AUX"}

def select_caption_tokens(caption: str) -> Tuple[List[str], List[int]]:
    """
    Returns (tokens, keep_indices) for NOUN/PROPN/VERB/AUX from the caption.
    """
    doc = _NLP(caption)
    toks = [t.text for t in doc]
    keep = [i for i, t in enumerate(doc) if (not t.is_space and not t.is_punct and t.pos_ in KEEP_POS)]
    return toks, keep

# ----------------------
# Model + embedding utils
# ----------------------
def select_device(arg: str) -> torch.device:
    if arg == "cuda": return torch.device("cuda")
    if arg == "cpu": return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dino(checkpoint_path: str, model_name: str, device: torch.device):
    model_backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    model_backbone.eval().to(device)

    sd = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(sd, dict) and "student_backbone" in sd:
        # DINO training wraps backbone in DinoBackbone class, so keys have "backbone." prefix
        # We need to strip this prefix to match the plain timm model
        state = sd["student_backbone"]
        cleaned = {k.replace("backbone.", "", 1): v for k, v in state.items()}
        incompatible = model_backbone.load_state_dict(cleaned, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"WARNING: Checkpoint loading issues:")
            print(f"  Missing keys: {len(incompatible.missing_keys)}")
            print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")
        else:
            print(f"✓ Successfully loaded DINO checkpoint with {len(cleaned)} parameters")
    elif isinstance(sd, dict) and "model" in sd:
        model_backbone.load_state_dict(sd["model"], strict=False)
    elif isinstance(sd, dict) and "state_dict" in sd:
        state = sd["state_dict"]
        cleaned = {k.split("backbone.",1)[-1]: v for k,v in state.items()}
        model_backbone.load_state_dict(cleaned, strict=False)
    else:
        model_backbone.load_state_dict(sd, strict=False)

    for p in model_backbone.parameters():
        p.requires_grad_(False)

    cfg = resolve_data_config({}, model=model_backbone)
    preprocess = create_transform(**cfg)

    @torch.no_grad()
    def cls_embed(imgs: torch.Tensor) -> torch.Tensor:
        feats = model_backbone.forward_features(imgs.to(device, non_blocking=True))
        if isinstance(feats, dict):
            if feats.get("x_norm_clstoken") is not None:
                x = feats["x_norm_clstoken"]
            elif feats.get("cls_token") is not None:
                x = feats["cls_token"]
            elif feats.get("x") is not None:
                x = feats["x"][:, 0]
            else:
                raise RuntimeError(f"Unknown features dict keys: {feats.keys()}")
        else:
            x = feats[:, 0] if feats.dim() == 3 else feats
        return F.normalize(x.float(), dim=-1)

    return preprocess, cls_embed

def pil_from_bgr(bgr) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

@torch.no_grad()
def embed_frames_batched(paths: List[str], preprocess, cls_embed, batch_size: int) -> np.ndarray:
    vecs = []
    batch_imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        pil = pil_from_bgr(img)
        batch_imgs.append(preprocess(pil))
        if len(batch_imgs) == batch_size:
            x = torch.stack(batch_imgs, 0)
            v = cls_embed(x).cpu().numpy()
            vecs.append(v); batch_imgs.clear()
    if batch_imgs:
        x = torch.stack(batch_imgs, 0)
        v = cls_embed(x).cpu().numpy()
        vecs.append(v)
    if not vecs:
        return np.zeros((0, 768), dtype=np.float32)
    return np.concatenate(vecs, axis=0)

def downsample_paths_uniform(paths: List[str], max_n: int) -> List[str]:
    if max_n is None or max_n <= 0 or len(paths) <= max_n:
        return paths
    idx = np.linspace(0, len(paths)-1, num=max_n)
    idx = np.round(idx).astype(int)
    idx = sorted(set(idx.tolist()))
    return [paths[i] for i in idx]

# ----------------------
# Social vectors (global + locals w/o timestamps)
# ----------------------
@torch.no_grad()
def global_social_vector(frames_dir: str, preprocess, cls_embed,
                         max_global_frames: int, batch_size: int) -> np.ndarray:
    paths = load_frame_paths(frames_dir)
    if not paths:
        raise FileNotFoundError(f"No frames in {frames_dir}")
    paths = downsample_paths_uniform(paths, max_global_frames)
    embs = embed_frames_batched(paths, preprocess, cls_embed, batch_size)
    return embs.mean(0) if embs.shape[0] else np.zeros((768,), dtype=np.float32)

@torch.no_grad()
def local_vectors_from_caption(frames_dir: str, caption: str, preprocess, cls_embed,
                               top_k_locals: int, pad_frames: int, batch_size: int
                               ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Approximate locals by mapping each kept POS token uniformly along the frame index axis,
    then averaging embeddings in a small window around the mapped frame.
    """
    all_paths = load_frame_paths(frames_dir)
    if not all_paths:
        return [], []

    tokens, keep_idxs = select_caption_tokens(caption)
    if not keep_idxs:
        return [], tokens

    keep_idxs = keep_idxs[:top_k_locals]
    N = len(all_paths)
    M = len(keep_idxs)

    # Pre-embed all frames once for efficiency
    frame_embs = embed_frames_batched(all_paths, preprocess, cls_embed, batch_size)  # [N, D]
    if frame_embs.shape[0] != N:
        # Some frames failed to load—fall back to what we have
        N = frame_embs.shape[0]

    locals_list = []
    for rank, k in enumerate(keep_idxs):
        # Map the k-th kept token (rank order) to a frame index
        # position = (rank + 0.5) / M in [0,1] -> frame idx in [0, N-1]
        f_center = int(round((rank + 0.5) / M * (N - 1)))
        s = max(0, f_center - pad_frames)
        e = min(N - 1, f_center + pad_frames)
        if e < s: 
            continue
        vec = frame_embs[s:e+1].mean(0)
        locals_list.append({"ti": int(k), "token": tokens[k], "vec": vec})

    return locals_list, tokens

# ----------------------
# Save pack
# ----------------------
def save_pack(clip_id: str, caption: str, global_vec: np.ndarray,
              locals_list: List[Dict[str, Any]], tokens: List[str], out_dir: str
              ) -> Tuple[str, Dict[str, str]]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(clip_id.encode()).hexdigest()

    paths = {
        "global": os.path.join(out_dir, f"{key}_global.npy"),
        "locals": os.path.join(out_dir, f"{key}_locals.npz"),
        "meta":   os.path.join(out_dir, f"{key}_meta.pkl"),
    }

    np.save(paths["global"], global_vec)
    np.savez_compressed(
        paths["locals"],
        **{f"{it['ti']}_{it['token']}": it["vec"] for it in locals_list}
    )
    meta = {
        "clip_id": clip_id,
        "caption": caption,
        "tokens": tokens,
        "locals_idx": [it["ti"] for it in locals_list],
        "locals_tokens": [it["token"] for it in locals_list],
    }
    with open(paths["meta"], "wb") as f:
        pickle.dump(meta, f)
    return key, paths

# ----------------------
# Discovery from CSV + frames_root
# ----------------------
def load_captions(captions_csv: str) -> List[Tuple[str, str]]:
    """
    Returns list of (video_stem, caption)
    """
    rows = []
    with open(captions_csv, "r") as f:
        reader = csv.DictReader(f)
        assert "video_name" in reader.fieldnames and "caption" in reader.fieldnames, \
            "captions_csv must have columns: video_name,caption"
        for r in reader:
            rows.append((r["video_name"], r["caption"]))
    return rows

# ----------------------
# Main
# ----------------------
def main():
    args = get_args()
    logger, fail_logger = setup_logging(args.out_dir, args.log_file)
    manifest_csv = str(Path(args.out_dir) / "social_packs_manifest.csv")
    ensure_manifest(manifest_csv)

    device = select_device(args.device)
    logger.info("==== Build Social Token Packs (caption-only locals) ====")
    logger.info(f"frames_root={args.frames_root}")
    logger.info(f"captions_csv={args.captions_csv}")
    logger.info(f"out_dir={args.out_dir}")
    logger.info(f"dino_ckpt={args.dino_ckpt}")
    logger.info(f"model_name={args.model_name}")
    logger.info(f"device={device}")

    preprocess, cls_embed = load_dino(args.dino_ckpt, args.model_name, device)

    rows = load_captions(args.captions_csv)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    logger.info(f"Found {len(rows)} captioned videos.")

    processed = 0
    for video_stem, caption in rows:
        video_stem = video_stem[:-4]
        t0 = time.time()
        clip_id = video_stem
        frames_dir = str(Path(args.frames_root) / video_stem)

        try:
            if args.resume and not args.overwrite and outputs_exist(clip_id, args.out_dir):
                logger.info(f"[SKIP existing] {clip_id}")
                key = hashlib.md5(clip_id.encode()).hexdigest()
                append_manifest(manifest_csv, [
                    clip_id, key,
                    str(Path(args.out_dir)/f"{key}_global.npy"),
                    str(Path(args.out_dir)/f"{key}_locals.npz"),
                    str(Path(args.out_dir)/f"{key}_meta.pkl"),
                    "", "", frames_dir, caption, "skipped", "", 0.0, str(device)
                ])
                continue

            # Global vector
            gvec = global_social_vector(
                frames_dir=frames_dir,
                preprocess=preprocess,
                cls_embed=cls_embed,
                max_global_frames=args.max_global_frames,
                batch_size=args.batch_size
            )

            # Local vectors from caption (no timestamps)
            locals_list, tokens = local_vectors_from_caption(
                frames_dir=frames_dir,
                caption=caption,
                preprocess=preprocess,
                cls_embed=cls_embed,
                top_k_locals=args.top_k_locals,
                pad_frames=args.pad_frames,
                batch_size=args.batch_size
            )

            key, paths = save_pack(
                clip_id=clip_id,
                caption=caption,
                global_vec=gvec,
                locals_list=locals_list,
                tokens=tokens,
                out_dir=args.out_dir
            )
            dt = time.time() - t0
            append_manifest(manifest_csv, [
                clip_id, key, paths["global"], paths["locals"], paths["meta"],
                len(locals_list), len(tokens), frames_dir, caption, "ok","", f"{dt:.3f}", str(device)
            ])
            logger.info(f"[OK] {clip_id} key={key} | locals={len(locals_list)} | {dt:.2f}s")
            processed += 1

        except Exception as e:
            dt = time.time() - t0
            err = f"{type(e).__name__}: {e}"
            logger.warning(f"[FAIL] {clip_id} ({dt:.2f}s) → {err}")
            fail_logger.warning(f"{clip_id}\t{err}")
            append_manifest(manifest_csv, [
                clip_id, "", "", "", "",
                "", "", frames_dir, caption, "error", err, f"{dt:.3f}", str(device)
            ])

    logger.info(f"Done. Processed={processed} | Total candidates={len(rows)}")
    logger.info(f"Manifest: {manifest_csv}")

if __name__ == "__main__":
    main()

