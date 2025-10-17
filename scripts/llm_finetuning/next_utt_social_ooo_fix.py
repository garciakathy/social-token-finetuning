# -*- coding: utf-8 -*-
"""
Next-Word Caption pipeline (DDP, frames or vectors)

NEW:
  --caption-nextword     # single-caption, next-word prediction per caption

Vectors (recommended here):
- For each caption row, creates (len(words)-1) samples:
    prompt = <SOC_G> [CAP]: w1 [<SOC_L>?] ... w_{i-1} [<SOC_L>?] [CAP]:
    target = [CAP]: w_i
  Inline <SOC_L> appears only after prefix words whose meta idx appear in locals_npz.
- global_vec is (768,) .npy per caption
- locals_npz has 768-D vectors keyed by word idx (int or string with a number inside)

Frames mode (unchanged from prior; left intact for parity).

DDP-safe, checkpointing, metrics CSV preserved.
"""

import os, re, math, json, argparse, time, csv, pickle, sys, random
from glob import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

SOC_G = "<SOC_G>"
SOC_L = "<SOC_L>"

# ---------------------- Metrics I/O ----------------------
METRIC_COLS = [
    "split","epoch","step","loss","ppl","tokens","tok_per_s","dt_s",
    "lr_proj","lr_dino","gpu_mem_mb","g_has","l_has","ablation_mode",
    "val_ppl_vis","val_ppl_no_vis","test_ppl_vis","test_ppl_no_vis"
]
def init_metrics_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(METRIC_COLS)
def write_metric_row(path: str, **kw):
    row = []
    for c in METRIC_COLS:
        v = kw.get(c, "")
        if isinstance(v, float):  v = float(f"{v:.8g}")
        row.append(v)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

# ---------------------- Utils ----------------------
def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def path_join(parent: str, p: Optional[str]) -> str:
    if not isinstance(p, str): return ""
    s = p.strip()
    if s == "" or s.lower() in ("none", "nan", "null"): return ""
    return s if os.path.isabs(s) else os.path.normpath(os.path.join(parent, s))

def parse_scene_part(clip_id: str) -> Tuple[str, Optional[str]]:
    m = re.match(r"^(.*)_P(\d+)$", str(clip_id))
    if m: return m.group(1), f"P{m.group(2)}"
    return str(clip_id), None

def init_distributed():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"]); world = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, local_rank, world, torch.device(f"cuda:{local_rank}")
    return False, 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_id_set(path: str) -> set[str]:
    """
    Accepts:
      - .txt: one clip_id per line
      - .csv: must contain a 'clip_id' column (fallback: first column header that contains 'clip')
    Returns a set of strings.
    """
    if not path: return set()
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"ID list not found: {path}")
    if p.suffix.lower() == ".txt":
        ids = [ln.strip() for ln in open(path, "r") if ln.strip()]
        return set(ids)
    # CSV
    import csv as _csv
    with open(path, newline="") as f:
        r = _csv.DictReader(f)
        cols = r.fieldnames or []
        pick = "clip_id" if "clip_id" in cols else next((c for c in cols if "clip" in c.lower()), None)
        if not pick:
            raise ValueError(f"No 'clip_id' (or *clip*) column found in {path}. Columns={cols}")
        return set(row[pick].strip() for row in r if row.get(pick))


# ---------------------- DINO Encoder ----------------------
class DINOEncoder(nn.Module):
    def __init__(self, model_name="vit_base_patch14_dinov2", tune_mode="cls_adapter", last_n=0,
                 checkpoint_path: str = "", checkpoint_key: str = "", checkpoint_strict: bool = False,
                 input_size: int = 518, autocast_dtype=torch.bfloat16):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, features_only=False)
        self.backbone.eval()
        self.tune_mode = tune_mode
        self.last_n = last_n
        self.autocast_dtype = autocast_dtype

        if input_size and input_size > 0:
            cfg = resolve_data_config({'input_size': (3, input_size, input_size)}, model=self.backbone)
        else:
            cfg = resolve_data_config({}, model=self.backbone)
        self.tf = create_transform(**cfg)
        self.feat_dim = getattr(self.backbone, "num_features", 768)

        # freeze all
        for p in self.backbone.parameters(): p.requires_grad = False
        # adapters / selective unfreeze
        if tune_mode == "cls_param_only":
            if hasattr(self.backbone, "cls_token"):
                self.backbone.cls_token.requires_grad = True
            else:
                raise ValueError("Backbone has no cls_token; use 'cls_adapter' or 'last_n'.")
        elif tune_mode == "cls_adapter":
            self.cls_adapter = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.GELU(),
                nn.Linear(self.feat_dim, self.feat_dim),
            )
        elif tune_mode == "last_n":
            if last_n <= 0: raise ValueError("last_n must be > 0")
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is None: raise ValueError("Backbone has no .blocks")
            for blk in blocks[-last_n:]:
                for p in blk.parameters(): p.requires_grad = True
        elif tune_mode == "full":
            for p in self.backbone.parameters(): p.requires_grad = True
        elif tune_mode == "frozen":
            pass
        else:
            raise ValueError(f"Unknown tune_mode={tune_mode}")

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path, checkpoint_key, checkpoint_strict)

    def _load_checkpoint(self, path: str, key: str = "", strict: bool = False):
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict):
            if key and key in sd and isinstance(sd[key], dict): sd = sd[key]
            elif "state_dict" in sd and isinstance(sd["state_dict"], dict): sd = sd["state_dict"]
            elif "model" in sd and isinstance(sd["model"], dict): sd = sd["model"]
            elif "student_backbone" in sd and isinstance(sd["student_backbone"], dict): sd = sd["student_backbone"]
            elif "teacher_backbone" in sd and isinstance(sd["teacher_backbone"], dict): sd = sd["teacher_backbone"]
        if isinstance(sd, dict) and len(sd) == 1 and isinstance(next(iter(sd.values())), dict):
            sd = next(iter(sd.values()))
        def strip_prefix(k: str):
            for pref in ("module.", "backbone.", "model.", "encoder.", "student_backbone.", "teacher_backbone."):
                if k.startswith(pref): return k[len(pref):]
            return k
        if isinstance(sd, dict): sd = {strip_prefix(k): v for k, v in sd.items()}
        missing, unexpected = self.backbone.load_state_dict(sd, strict=strict)
        print(f"[DINO] loaded: {path} | missing={len(missing)} unexpected={len(unexpected)}")
        if missing: print("  missing (first 10):", missing[:10])
        if unexpected: print("  unexpected (first 10):", unexpected[:10])

    @torch.no_grad()
    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.tf(img.convert("RGB"))

    def _extract_cls(self, feats):
        if isinstance(feats, dict):
            if "x_norm_clstoken" in feats: return feats["x_norm_clstoken"]
            if "tokens" in feats and feats["tokens"].dim() == 3: return feats["tokens"][:, 0]
            if "x" in feats and feats["x"].dim() == 3: return feats["x"][:, 0]
        if torch.is_tensor(feats) and feats.dim() == 3: return feats[:, 0]
        raise RuntimeError("Cannot find CLS token in DINO forward_features output.")

    def forward_images(self, images: List[Image.Image]) -> torch.Tensor:
        dev = next(self.parameters()).device
        batch = torch.stack([self._preprocess(im) for im in images]).to(dev)
        with torch.autocast(device_type=dev.type, dtype=self.autocast_dtype, enabled=True):
            feats = self.backbone.forward_features(batch)
            cls = self._extract_cls(feats)
            if hasattr(self, "cls_adapter"): cls = self.cls_adapter(cls)
        return cls

    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        return self.forward_images(images)

# ---------------------- Projection & LM ----------------------
class ScaleShift(nn.Module):
    def __init__(self, dim, init_scale=1.0):
        super().__init__()
        self.g = nn.Parameter(torch.full((dim,), float(init_scale)))
        self.b = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        eps = 1e-6
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        x = x / rms
        return x * self.g + self.b

class VisualProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 use_scaleshift: bool = True,
                 init_scale_in: float = 1.0,
                 init_scale_out: float | None = None,
                 dropout: float = 0.0):
        super().__init__()
        mods = []
        if use_scaleshift: mods.append(ScaleShift(in_dim, init_scale=init_scale_in))
        else: mods.append(nn.LayerNorm(in_dim))
        mods += [nn.Linear(in_dim, out_dim), nn.GELU()]
        if dropout > 0: mods.append(nn.Dropout(dropout))
        mods.append(nn.Linear(out_dim, out_dim))
        if use_scaleshift:
            if init_scale_out is None: init_scale_out = 1.0
            mods.append(ScaleShift(out_dim, init_scale=init_scale_out))
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x)

class GemmaWithInjection(nn.Module):
    def __init__(self, lm_name: str, add_tokens: List[str], torch_dtype=torch.bfloat16, zero_init=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        existing = set(self.tokenizer.get_vocab().keys())
        to_add = [t for t in add_tokens if t not in existing]
        if to_add:
            self.tokenizer.add_special_tokens({"additional_special_tokens": to_add})
            if hasattr(self.tokenizer, "unique_no_split_tokens"):
                for t in to_add:
                    if t not in self.tokenizer.unique_no_split_tokens:
                        self.tokenizer.unique_no_split_tokens.append(t)
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name, torch_dtype=torch.bfloat16)
        self.lm.resize_token_embeddings(len(self.tokenizer))
        if zero_init and to_add:
            with torch.no_grad():
                emb = self.lm.get_input_embeddings().weight
                for tok in to_add:
                    tid = self.tokenizer.convert_tokens_to_ids(tok)
                    emb[tid].zero_()
        for p in self.lm.parameters(): p.requires_grad = False
        self.lm.eval()
        self.id_soc_g = self.tokenizer.convert_tokens_to_ids(SOC_G)
        self.id_soc_l = self.tokenizer.convert_tokens_to_ids(SOC_L)
        assert self.id_soc_g != self.tokenizer.unk_token_id, f"{SOC_G} not in tokenizer"
        assert self.id_soc_l != self.tokenizer.unk_token_id, f"{SOC_L} not in tokenizer"

    def forward(self, input_ids, attention_mask, labels, proj_global, proj_locals, inject_visuals=True, ablation_mode="both"):
        """
        ablation_mode: 'both', 'global_only', 'local_only', 'none'
        """
        emb = self.lm.get_input_embeddings()(input_ids)  # [B,T,H]
        if inject_visuals:
            dtype = emb.dtype
            # Global token injection
            if ablation_mode in ("both", "global_only") and proj_global is not None:
                pos_g = (input_ids == self.id_soc_g).nonzero(as_tuple=False)
                if pos_g.numel() > 0:
                    b = pos_g[:, 0]; t = pos_g[:, 1]
                    emb[b, t, :] = proj_global[b, :].to(dtype=dtype)
            elif ablation_mode == "local_only":
                # In local_only mode, zero out <SOC_G> to avoid NaN from uninitialized embeddings
                pos_g = (input_ids == self.id_soc_g).nonzero(as_tuple=False)
                if pos_g.numel() > 0:
                    b = pos_g[:, 0]; t = pos_g[:, 1]
                    emb[b, t, :] = torch.zeros_like(emb[b, t, :])
            # Local token injection
            if ablation_mode in ("both", "local_only") and proj_locals is not None:
                B = input_ids.size(0)
                for b in range(B):
                    lpos = (input_ids[b] == self.id_soc_l).nonzero(as_tuple=False).squeeze(-1)
                    if lpos.numel() > 0 and proj_locals[b] is not None:
                        L = min(lpos.numel(), proj_locals[b].size(0))
                        emb[b, lpos[:L], :] = proj_locals[b][:L, :].to(dtype=dtype)
            elif ablation_mode == "global_only":
                # In global_only mode, zero out <SOC_L> to avoid NaN from uninitialized embeddings
                B = input_ids.size(0)
                for b in range(B):
                    lpos = (input_ids[b] == self.id_soc_l).nonzero(as_tuple=False).squeeze(-1)
                    if lpos.numel() > 0:
                        emb[b, lpos, :] = torch.zeros_like(emb[b, lpos, :])
        out = self.lm(inputs_embeds=emb, attention_mask=attention_mask, labels=labels)
        return out

# ---------------------- Inline helpers / loaders ----------------------
def _parse_npz_word_indices(npz_path: str) -> List[int]:
    if not (npz_path and os.path.exists(npz_path)): return []
    arr = np.load(npz_path, allow_pickle=True)
    try: keys = list(arr.keys())
    except Exception: keys = list(arr.files)
    out = []
    for k in keys:
        if isinstance(k, (int, np.integer)): out.append(int(k))
        else:
            m = re.search(r"(\d+)", str(k))
            if m: out.append(int(m.group(1)))
    return sorted(set(out))

def _load_meta_words(meta_pkl_path: str) -> List[Dict[str, Any]]:
    if not (meta_pkl_path and os.path.exists(meta_pkl_path)): return []
    try:
        with open(meta_pkl_path, "rb") as f: meta = pickle.load(f)
    except Exception:
        return []
    words = []
    if isinstance(meta, dict) and isinstance(meta.get("words"), list):
        for i, w in enumerate(meta["words"]):
            ws = float(w.get("start", w.get("begin", 0.0)) or 0.0)
            we = float(w.get("end",   w.get("finish", ws)) or ws)
            wt = str(w.get("text", w.get("token", w.get("word", ""))) or "")
            words.append({"idx": int(w.get("idx", i)), "start": ws, "end": we, "text": wt})
    return words

def load_transcript_segments(x):
    data = None
    if isinstance(x, (list, dict)): data = x
    elif isinstance(x, str):
        if os.path.exists(x):
            with open(x, "r") as f: data = json.load(f)
        else:
            try: data = json.loads(x)
            except Exception:
                return [{"start": 0.0, "end": 0.0, "text": x.strip(), "words": []}]
    else:
        return [{"start": 0.0, "end": 0.0, "text": "", "words": []}]

    def norm_words(words_like):
        out = []
        for w in (words_like or []):
            if isinstance(w, dict):
                ws = float(w.get("start", w.get("begin", 0.0)) or 0.0)
                we = float(w.get("end",   w.get("finish", ws)) or ws)
                wt = str(w.get("text", w.get("token", w.get("word", ""))) or "")
            else:
                ws, we, wt = 0.0, 0.0, str(w)
            out.append({"start": ws, "end": we, "text": wt})
        return out

    def seg_from_words(words):
        words = norm_words(words)
        if words:
            s0, e1 = words[0]["start"], words[-1]["end"]
            txt = " ".join([w["text"] for w in words]).strip()
        else:
            s0 = e1 = 0.0; txt = ""
        return {"start": s0, "end": e1, "text": txt, "words": words}

    if isinstance(data, list):
        out = []
        for d in data:
            if isinstance(d, dict):
                words = d.get("words", d.get("tokens", []))
                if words: out.append(seg_from_words(words))
                else:
                    s0 = float(d.get("start", d.get("begin", 0.0)) or 0.0)
                    e1 = float(d.get("end",   d.get("finish", s0)) or s0)
                    txt = str(d.get("text", d.get("transcript", "")) or "").strip()
                    out.append({"start": s0, "end": e1, "text": txt, "words": []})
            else:
                out.append({"start": 0.0, "end": 0.0, "text": str(d), "words": []})
        return out

    if isinstance(data, dict):
        for key in ("metadata:transcript", "segments"):
            if isinstance(data.get(key), list):
                return load_transcript_segments(data[key])
        if isinstance(data.get("words"), list):
            return [seg_from_words(data["words"])]
        if ("text" in data) or ("transcript" in data):
            txt = str(data.get("text", data.get("transcript", "")) or "").strip()
            return [{"start": float(data.get("start", 0.0) or 0.0),
                     "end":   float(data.get("end",   0.0) or 0.0),
                     "text":  txt,
                     "words": []}]
    return [{"start": 0.0, "end": 0.0, "text": str(data), "words": []}]

# Extra helpers for NPZ key→position mapping
def _digits_to_int(s: str):
    m = re.findall(r"\d+", str(s))
    return int(m[-1]) if m else None

def _npz_build_pos2key(arr) -> Dict[int, str]:
    # NpzFile exposes .files
    keys = []
    try:
        keys = list(arr.files)
    except Exception:
        try:
            keys = list(arr.keys())
        except Exception:
            keys = []
    pos2key = {}
    for k in keys:
        d = _digits_to_int(k)
        if d is not None:
            pos2key[int(d)] = k
    return pos2key

# ---------------------- Frames helpers (kept) ----------------------
def list_frames(frames_dir: str) -> List[str]:
    if not frames_dir: return []
    pats = ("*.jpg","*.jpeg","*.png","*.webp")
    allf = []
    for p in pats: allf += glob(os.path.join(frames_dir, p))
    def sk(p):
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else p
    return sorted(allf, key=sk)

def select_local_frames(frames: List[str], word_times: List[float],
                        clip_start: float, clip_end: float,
                        K: int = 32, topup_uniform: bool = True) -> List[str]:
    if not frames: return []
    T = max(clip_end - clip_start, 1e-6)
    idxs = []
    for wt in word_times:
        pos = (wt - clip_start) / T
        i = int(round(pos * (len(frames) - 1)))
        idxs.append(max(0, min(len(frames) - 1, i)))
    idxs = sorted(set(idxs))
    if len(idxs) > K:
        step = len(idxs) / float(K)
        idxs = [idxs[int(round(i*step))] for i in range(K)]
    elif topup_uniform and len(idxs) < min(K, len(frames)):
        uni = np.linspace(0, len(frames)-1, num=min(K, len(frames)), dtype=int).tolist()
        seen = set(idxs)
        for u in uni:
            if u not in seen:
                idxs.append(u); seen.add(u)
                if len(idxs) >= min(K, len(frames)): break
        idxs = sorted(idxs)
    return [frames[i] for i in idxs]

def select_global_frames(frames: List[str], n: int = 4) -> List[str]:
    if not frames: return []
    n = max(1, min(n, len(frames)))
    idxs = np.linspace(0, len(frames)-1, num=n, dtype=int).tolist()
    return [frames[i] for i in idxs]

def pick_first_path(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v: return v
    return ""

# ---------------------- Splits ----------------------
def build_splits(manifest_csv: str, parent_dir: str, seed: int, train_frac: float, val_frac: float):
    df = pd.read_csv(manifest_csv)
    # normalize path-like columns
    for col in ["global_vec","locals_npz","meta_pkl","frames_dir","frames_path","frames_root",
                "transcript_json","transcripts_json","transcript_path","transcript","asr_json"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: path_join(parent_dir, x))
    tmp = df["clip_id"].apply(parse_scene_part)
    df["scene_root"] = tmp.apply(lambda x: x[0])
    df["participant"] = tmp.apply(lambda x: x[1])

    rng = np.random.RandomState(seed)
    scenes = df["scene_root"].drop_duplicates().values
    rng.shuffle(scenes)
    n = len(scenes)
    n_train = int(round(train_frac * n))
    n_val   = int(round(val_frac * n))
    train_s = set(scenes[:n_train]); val_s = set(scenes[n_train:n_train+n_val]); test_s = set(scenes[n_train+n_val:])
    return {
        "train": df[df.scene_root.isin(train_s)].copy(),
        "val":   df[df.scene_root.isin(val_s)].copy(),
        "test":  df[df.scene_root.isin(test_s)].copy(),
    }

# ---------------------- NEW: caption → next-word ----------------------
def _inline_soc_l_for_prefix(prefix_words, npz_indices, max_locals: int) -> tuple[str, list[int]]:
    npz_set = set(int(i) for i in npz_indices)
    used, pieces = [], []
    for w in prefix_words:
        pieces.append(str(w.get("text","")))
        wid = int(w.get("idx", -1))
        if wid in npz_set and ((max_locals <= 0) or (len(used) < max_locals)):
            pieces.append(SOC_L); used.append(wid)
    return (" ".join(pieces).replace("  "," ").strip(), used)

def build_caption_nextword_rows(rows: list[dict[str,Any]],
                                max_locals: int,
                                col_transcript: str = "",
                                align_eps: float = 0.5) -> list[dict[str,Any]]:
    out = []
    CAP = "CAP"
    for r in rows:
        # Accept either a file path OR raw caption text (string in the CSV cell)
        tval = r.get(col_transcript) if col_transcript else pick_first_path(
            r, ["caption","transcript_json","transcripts_json","transcript_path","transcript","asr_json"]
        )
        if tval is None:
            continue

        segs = load_transcript_segments(tval)  # <- this handles file path, JSON string, or raw text
        if not segs:
            continue
        seg = segs[0]
        words = seg.get("words") or []
        if not words:
            toks = [w for w in (seg.get("text","").strip().split()) if w]
            words = [{"idx": i, "start": 0.0, "end": 0.0, "text": t} for i, t in enumerate(toks)]

        meta_pkl = r.get("meta_pkl",""); locals_npz = r.get("locals_npz","")

        # If meta exists, ALWAYS borrow idx from it so <SOC_L> aligns with your locals_npz
        if meta_pkl and os.path.exists(meta_pkl):
            meta_words = _load_meta_words(meta_pkl)
            meta_by_order = {i:m for i,m in enumerate(meta_words)}
            if len(meta_words) >= len(words):
                for i in range(len(words)):
                    words[i]["idx"] = meta_by_order.get(i, {"idx": i}).get("idx", i)

        npz_ids = _parse_npz_word_indices(locals_npz)
        use_order_mode = bool(npz_ids)

        # Slide over words → (len(words)-1) next-word samples
        for i in range(1, len(words)):
            prefix = words[:i]
            if npz_ids:
                prompt_inline, used_ids = _inline_soc_l_for_prefix(prefix, npz_ids, max_locals)
            else:
                prompt_inline, used_ids = (" ".join([w.get("text","") for w in prefix]).strip(), [])

            prompt_text = f"{SOC_G} [{CAP}]: {prompt_inline} [{CAP}]: "
            target_text = f"[{CAP}]: {words[i].get('text','').strip()}"

            out.append({
                "scene_root": r.get("clip_id"),
                "clip_id": r.get("clip_id"),
                "participant": CAP,
                "prompt_text": prompt_text,
                "target_text": target_text,
                "global_frames": [],
                "local_frames": [],
                "transcript_json": tval,
                "global_vec": r.get("global_vec"),
                "locals_npz": locals_npz if used_ids else "",
                "meta_pkl": meta_pkl if used_ids else "",
                "loc_word_ids": used_ids,                   # positions in prefix
                "locals_order_mode": use_order_mode,        # NEW flag for loader
                "fallback_block_locals": False,
            })
    return out


# ---------------------- Legacy expand (dialogue) kept ----------------------
_FALLBACK_WARN_MAX = 100
_FALLBACK_WARN_CNT = 0
def warn_once(msg: str):
    global _FALLBACK_WARN_CNT
    if _FALLBACK_WARN_CNT < _FALLBACK_WARN_MAX:
        print(f"[warn:inline_soc_l] {msg}", file=sys.stderr)
        _FALLBACK_WARN_CNT += 1
        if _FALLBACK_WARN_CNT == _FALLBACK_WARN_MAX:
            print("[warn:inline_soc_l] further warnings suppressed", file=sys.stderr)

def merge_dialogue_by_time(segs_a, segs_b, spk_a, spk_b):
    tagged = [{"speaker": spk_a, **s} for s in segs_a] + [{"speaker": spk_b, **s} for s in segs_b]
    tagged.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
    return tagged

def _select_words_in_window(meta_words, t0: float, t1: float, eps: float = 0.5):
    sel = []
    lo, hi = t0 - eps, t1 + eps
    for w in meta_words:
        if not (w["end"] < lo or w["start"] > hi):
            sel.append(w)
    return sel

def _inline_soc_l_for_segment(seg, meta_words, npz_indices, max_locals, eps: float):
    t0 = float(seg.get("start", 0.0)); t1 = float(seg.get("end", t0))
    in_win = _select_words_in_window(meta_words, t0, t1, eps=eps)
    if not in_win:
        return (seg.get("text","").strip(), [])
    npz_set = set(int(i) for i in npz_indices)
    used = []; pieces = []
    for w in in_win:
        pieces.append(str(w["text"]))
        if int(w["idx"]) in npz_set:
            if (max_locals <= 0) or (len(used) < max_locals):
                pieces.append(SOC_L); used.append(int(w["idx"]))
    return (" ".join(pieces).replace("  "," ").strip(), used)

def expand_to_nextutt_rows(rows: List[Dict[str, Any]],
                           context_turns: int,
                           visual_mode: str,
                           max_locals: int,
                           col_transcript: str = "",
                           col_frames: str = "",
                           fallback_split: str = "none",
                           split_k: int = 12,
                           split_sec: float = 1.5,
                           global_nframes: int = 4,
                           locals_topup: bool = True,
                           require_visuals: bool = False,
                           align_eps: float = 1.0) -> List[Dict[str, Any]]:
    # (unchanged from your original; left here for completeness in case you still use dialogue mode)
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        root, _ = parse_scene_part(r.get("clip_id"))
        r["scene_root"] = root
        by_scene.setdefault(root, []).append(r)

    out = []
    for scene_root, recs in by_scene.items():
        scene_frames_dir = None
        if col_frames:
            for r in recs:
                fdir = r.get(col_frames)
                if fdir and os.path.isdir(fdir):
                    scene_frames_dir = fdir; break
        else:
            for r in recs:
                fdir = pick_first_path(r, ["frames_dir","frames_path","frames_root"])
                if fdir and os.path.isdir(fdir):
                    scene_frames_dir = fdir; break

        by_spk = { r.get("participant"): r for r in recs if r.get("participant") is not None }
        if not by_spk: continue

        side = {}
        for spk, r in by_spk.items():
            tpath = r.get(col_transcript) if col_transcript else pick_first_path(r, ["transcript_json","transcripts_json","transcript_path","transcript","asr_json"])
            if not tpath or not os.path.exists(tpath): continue
            segs = load_transcript_segments(tpath)
            side[spk] = {"rec": r, "segs": segs}
        if not side: continue

        if len(side) == 2:
            spk_a, spk_b = sorted(side.keys())
            merged = merge_dialogue_by_time(side[spk_a]["segs"], side[spk_b]["segs"], spk_a, spk_b)
        else:
            spk = list(side.keys())[0]
            merged = [{"speaker": spk, **s} for s in side[spk]["segs"]]

        if len(merged) <= context_turns and fallback_split != "none":
            new_by_spk = {}
            for spk_key, payload in side.items():
                segs = payload["segs"]; ex = []
                for seg in segs:
                    if fallback_split == "fixed_words":
                        words = seg.get("words", []) or []
                        if not words:
                            txt = (seg.get("text","")).strip()
                            toks = [t for t in txt.split() if t]
                            words = [{"start": seg.get("start",0.0), "end": seg.get("end",0.0), "text": w} for w in toks]
                        for i in range(0, len(words), max(1, split_k)):
                            wchunk = words[i:i+split_k]
                            if not wchunk: continue
                            s0 = float(wchunk[0].get("start", seg.get("start",0.0)) or 0.0)
                            e1 = float(wchunk[-1].get("end", s0) or s0)
                            ex.append({"start": s0, "end": e1, "text": " ".join([w.get("text","") for w in wchunk]).strip(), "words": wchunk})
                    elif fallback_split == "fixed_seconds":
                        words = seg.get("words", []) or []
                        chunks, cur, cur_start = [], [], None
                        for w in words:
                            ws = float(w.get("start", seg.get("start",0.0)) or 0.0)
                            we = float(w.get("end", ws) or ws)
                            if cur_start is None: cur_start = ws
                            cur.append(w)
                            if we - cur_start >= max(0.5, split_sec):
                                chunks.append({"start": cur_start, "end": we, "text":" ".join([x.get("text","") for x in cur]).strip(), "words": cur})
                                cur, cur_start = [], None
                        if cur:
                            ws = float(cur[0].get("start", seg.get("start",0.0)) or 0.0)
                            we = float(cur[-1].get("end", ws) or ws)
                            chunks.append({"start": ws, "end": we, "text":" ".join([x.get("text","") for x in cur]).strip(), "words": cur})
                        ex += chunks
                    else:
                        ex.append(seg)
                new_by_spk[spk_key] = ex
            if len(side) == 2:
                merged = merge_dialogue_by_time(new_by_spk[spk_a], new_by_spk[spk_b], spk_a, spk_b)
            else:
                spk = list(side.keys())[0]
                merged = [{"speaker": spk, **s} for s in new_by_spk[spk]]

        for i in range(context_turns, len(merged)):
            target = merged[i]
            tgt_spk = target["speaker"]
            tgt_rec = side[tgt_spk]["rec"] if tgt_spk in side else recs[0]
            ctx = merged[i-context_turns:i]
            prompt_parts = [SOC_G]
            loc_word_ids: List[int] = []
            locals_npz_path_for_inline = ""
            meta_pkl_for_inline = ""
            for j, seg in enumerate(ctx):
                spk_c = seg["speaker"]
                seg_text = (seg.get("text") or "").strip()
                if (j == len(ctx) - 1) and (visual_mode == "vectors"):
                    rec_c = side.get(spk_c, {}).get("rec", {})
                    meta_pkl = rec_c.get("meta_pkl", ""); locals_npz = rec_c.get("locals_npz", "")
                    meta_words = _load_meta_words(meta_pkl); npz_ids = _parse_npz_word_indices(locals_npz)
                    if meta_words and npz_ids:
                        seg_text_inline, used_ids = _inline_soc_l_for_segment(seg, meta_words, npz_ids, max_locals=max_locals, eps=align_eps)
                        if used_ids:
                            seg_text = seg_text_inline
                            loc_word_ids = used_ids
                            locals_npz_path_for_inline = locals_npz
                            meta_pkl_for_inline = meta_pkl
                prompt_parts.append(f"[{spk_c}]: {seg_text}")
            prompt_prefix = " ".join(prompt_parts) + f" [{tgt_spk}]: "
            target_text = f"[{tgt_spk}]: {(target.get('text') or '').strip()}"

            if visual_mode == "frames":
                fdir = tgt_rec.get(col_frames) if col_frames else pick_first_path(tgt_rec, ["frames_dir","frames_path","frames_root"])
                frames = list_frames(fdir) if (fdir and os.path.isdir(fdir)) else []
                words = target.get("words", [])
                clip_start = float(merged[0].get("start", 0.0)); clip_end = float(merged[-1].get("end", clip_start))
                mids = [0.5*(float(w.get("start", clip_start)) + float(w.get("end", clip_start))) for w in words]
                local_frames = select_local_frames(frames, mids, clip_start, clip_end, K=max_locals, topup_uniform=True)
                global_frames = select_global_frames(frames, n=4)
                prompt_text = prompt_prefix + (" " + " ".join([SOC_L]*len(local_frames)) if local_frames else "")
                out.append({
                    "scene_root": scene_root, "clip_id": tgt_rec.get("clip_id"), "participant": tgt_spk,
                    "prompt_text": prompt_text, "target_text": target_text,
                    "global_frames": global_frames, "local_frames": local_frames,
                    "transcript_json": pick_first_path(tgt_rec, [col_transcript] if col_transcript else ["transcript_json","transcripts_json","transcript_path","transcript","asr_json"]),
                    "global_vec": tgt_rec.get("global_vec"), "locals_npz": tgt_rec.get("locals_npz"), "meta_pkl": tgt_rec.get("meta_pkl"),
                    "loc_word_ids": [], "fallback_block_locals": False,
                })
            else:
                out.append({
                    "scene_root": scene_root, "clip_id": tgt_rec.get("clip_id"), "participant": tgt_spk,
                    "prompt_text": prompt_prefix, "target_text": target_text,
                    "global_frames": [], "local_frames": [],
                    "transcript_json": pick_first_path(tgt_rec, [col_transcript] if col_transcript else ["transcript_json","transcripts_json","transcript_path","transcript","asr_json"]),
                    "global_vec": tgt_rec.get("global_vec"),
                    "locals_npz": locals_npz_path_for_inline, "meta_pkl": meta_pkl_for_inline,
                    "loc_word_ids": loc_word_ids, "fallback_block_locals": False,
                })
    return out

# ---------------------- Dataset / Collator ----------------------
class NextUttJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows = []
        self.jsonl_path = jsonl_path
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r") as f:
                for line in f:
                    if line.strip():
                        self.rows.append(json.loads(line))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

class Collator:
    def __init__(self, tokenizer, visual_mode: str = "frames", max_len: int = 1024, max_locals: int = 50):
        self.tok = tokenizer
        self.visual_mode = visual_mode
        self.max_len = max_len
        self.max_locals = max_locals
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids_list, attn_list, labels_list = [], [], []
        global_frames_batch, local_frames_batch = [], []
        global_vecs, locals_npzs = [], []
        loc_word_ids_batch, fallback_batch = [], []
        locals_order_mode_batch = []

        for s in batch:
            prompt = s["prompt_text"]; target = s["target_text"]
            enc_p = self.tok(prompt, add_special_tokens=False)
            enc_t = self.tok(target + self.tok.eos_token, add_special_tokens=False)
            ids = (enc_p["input_ids"] + enc_t["input_ids"])[: self.max_len]
            attn = [1]*len(ids)
            labels = ([-100]*len(enc_p["input_ids"]) + enc_t["input_ids"])[: self.max_len]

            input_ids_list.append(ids); attn_list.append(attn); labels_list.append(labels)
            global_frames_batch.append(s.get("global_frames") or [])
            local_frames_batch.append(s.get("local_frames") or [])
            global_vecs.append(s.get("global_vec"))
            locals_npzs.append(s.get("locals_npz"))
            loc_word_ids_batch.append(s.get("loc_word_ids") or [])
            fallback_batch.append(bool(s.get("fallback_block_locals", False)))
            locals_order_mode_batch.append(bool(s.get("locals_order_mode", False)))

        if not input_ids_list:
            return dict(
                input_ids=torch.zeros((0,1), dtype=torch.long),
                attention_mask=torch.zeros((0,1), dtype=torch.long),
                labels=torch.ones((0,1), dtype=torch.long) * -100,
                global_frames=[], local_frames=[], global_vecs=[], locals_npzs=[],
                loc_word_ids=[], fallback_block_locals=[], locals_order_mode=[]
            )

        maxT = max(len(x) for x in input_ids_list)
        pad_id = self.tok.pad_token_id
        def pad(arr, pad_val): return arr + [pad_val]*(maxT - len(arr))
        input_ids = torch.tensor([pad(x, pad_id) for x in input_ids_list], dtype=torch.long)
        attention_mask = torch.tensor([pad(x, 0) for x in attn_list], dtype=torch.long)
        labels = torch.tensor([pad(x, -100) for x in labels_list], dtype=torch.long)

        return dict(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            global_frames=global_frames_batch, local_frames=local_frames_batch,
            global_vecs=global_vecs, locals_npzs=locals_npzs,
            loc_word_ids=loc_word_ids_batch, fallback_block_locals=fallback_batch,
            locals_order_mode=locals_order_mode_batch
        )

# ---------------------- Image Cache ----------------------
class ImageCache:
    def __init__(self, max_items=4096, verbose=False):
        self.cache = {}; self.order = []; self.max_items = max_items; self.verbose = verbose
    def get(self, path: str) -> Optional[Image.Image]:
        if not path: return None
        if path in self.cache: return self.cache[path]
        try:
            if not os.path.exists(path):
                if self.verbose: print(f"[frames] missing path: {path}")
                return None
            img = Image.open(path).convert("RGB"); img.load()
        except Exception as e:
            if self.verbose: print(f"[frames] failed to open {path}: {e}")
            return None
        self.cache[path] = img; self.order.append(path)
        if len(self.order) > self.max_items:
            old = self.order.pop(0); self.cache.pop(old, None)
        return img

# ---------------------- Train / Eval ----------------------
def unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m

def save_checkpoint(tag: str, epoch: int, val_ppl: float, output_dir: str,
                    projector: nn.Module, dino: nn.Module, opt, sched,
                    run_config: Dict[str, Any], tokenizer, rank: int,
                    save_adapter_only: bool):
    if rank != 0: return
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "epoch": epoch, "val_ppl": float(val_ppl), "args": run_config,
        "projector": unwrap(projector).state_dict(),
        "dino": unwrap(dino).state_dict(),
        "opt": opt.state_dict() if opt is not None else None,
        "sched": sched.state_dict() if sched is not None else None,
        "tokens": {"soc_g": SOC_G, "soc_l": SOC_L},
        "lm_name": run_config.get("lm_name"),
        "dino_name": run_config.get("dino_name"),
        "dino_tune_mode": run_config.get("dino_tune_mode"),
        "dino_last_n": run_config.get("dino_last_n"),
    }
    p = os.path.join(ckpt_dir, f"{tag}.pt"); torch.save(payload, p)
    print(f"[ckpt] saved {p} (epoch={epoch}, val_ppl={val_ppl:.4f})")
    torch.save({"epoch": epoch, "val_ppl": float(val_ppl), "projector": unwrap(projector).state_dict()},
               os.path.join(ckpt_dir, "projector_only.pt"))
    if save_adapter_only:
        dino_sd = unwrap(dino).state_dict()
        adapter_sd = {k: v for k, v in dino_sd.items() if k.startswith("cls_adapter.")}
        if adapter_sd:
            torch.save({"epoch": epoch, "val_ppl": float(val_ppl), "cls_adapter": adapter_sd},
                       os.path.join(ckpt_dir, "dino_adapter_only.pt"))
            print("[ckpt] saved dino_adapter_only.pt")

def load_resume(resume_path: str, projector: nn.Module, dino: nn.Module, opt=None, sched=None, map_location="cpu"):
    ckpt = torch.load(resume_path, map_location=map_location)
    if "projector" in ckpt: unwrap(projector).load_state_dict(ckpt["projector"], strict=True)
    if "dino" in ckpt: unwrap(dino).load_state_dict(ckpt["dino"], strict=False)
    if opt is not None and ckpt.get("opt") is not None: opt.load_state_dict(ckpt["opt"])
    if sched is not None and ckpt.get("sched") is not None: sched.load_state_dict(ckpt["sched"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("val_ppl", float("inf")))
    print(f"[resume] loaded {resume_path} @ epoch={start_epoch-1}, best_val_ppl={best_val:.4f}")
    return start_epoch, best_val

def dino_encode_images_in_chunks(dino, imgs: List[Image.Image], chunk: int) -> torch.Tensor:
    outs = []
    for i in range(0, len(imgs), max(1, chunk)):
        outs.append(dino(imgs[i:i+chunk]))
    return torch.cat(outs, dim=0) if outs else None

def audit_frames(jsonl_path, out_csv, N=5000):
    try:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        ok_g=miss_g=ok_l=miss_l=0
        with open(jsonl_path) as f, open(out_csv, "w", newline="") as w:
            wr = csv.writer(w); wr.writerow(["idx","clip_id","participant","global_count","local_count"])
            for i, line in enumerate(f):
                if i>=N: break
                s = json.loads(line)
                g = s.get("global_frames") or []
                l = s.get("local_frames") or []
                ok_g += int(len(g)>0); miss_g += int(len(g)==0)
                ok_l += int(len(l)>0); miss_l += int(len(l)==0)
                wr.writerow([i, s.get("clip_id",""), s.get("participant",""), len(g), len(l)])
        print(f"[audit] {jsonl_path}: global>0={ok_g} zero={miss_g} | locals>0={ok_l} zero={miss_l}")
    except Exception as e:
        print(f"[audit] failed: {e}")

def train_and_eval(
    parent_dir: str,
    output_dir: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    context_turns: int,
    visual_mode: str,
    max_locals: int,
    global_nframes: int,
    locals_topup: bool,
    require_visuals: bool,
    lm_name: str,
    dino_name: str,
    dino_tune_mode: str,
    dino_last_n: int,
    epochs: int,
    batch_size: int,
    lr_proj: float,
    lr_dino: float,
    warmup_steps: int,
    dino_checkpoint: str,
    dino_checkpoint_key: str,
    dino_checkpoint_strict: bool,
    col_transcript: str,
    col_frames: str,
    max_len: int,
    num_workers: int,
    fallback_split: str,
    split_k: int,
    split_sec: float,
    log_interval: int,
    limit_train_steps: int,
    limit_val_steps: int,
    dino_local_batch: int,
    dino_input_size: int,
    metrics_csv: str,
    resume_path: str,
    save_every_epochs: int,
    save_adapter_only: bool,
    grad_clip: float,
    align_eps: float,
    caption_nextword: bool,
    train_id_list: str, val_id_list: str, test_id_list: str, val_frac_of_train: float,
    train_ablation_mode: str,
    eval_ablations: List[str]
):
    is_dist, rank, local_rank, world, device = init_distributed()
    if rank == 0: print(f"[DDP] is_dist={is_dist} world={world} device={device}")
    set_all_seeds(seed + rank)

    # Try multiple manifest filenames (prefer newer format)
    manifest_candidates = [
        "social_packs_manifest.csv",
        "scene_packs_manifest_recovered.csv",
        "scene_packs_manifest.csv"
    ]
    manifest_csv = None
    for candidate in manifest_candidates:
        candidate_path = os.path.join(parent_dir, candidate)
        if os.path.exists(candidate_path):
            manifest_csv = candidate_path
            break
    if manifest_csv is None:
        raise FileNotFoundError(f"No manifest found in {parent_dir}. Tried: {manifest_candidates}")
    if rank == 0 and not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Missing manifest CSV at {manifest_csv}")

    os.makedirs(output_dir, exist_ok=True)
    split_dir = os.path.join(output_dir, "splits")
    json_dir  = os.path.join(output_dir, "jsonl")
    log_dir   = os.path.join(output_dir, "logs")
    if rank == 0:
        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    run_config = {
        "parent_dir": parent_dir, "output_dir": output_dir, "seed": seed,
        "train_frac": train_frac, "val_frac": val_frac, "context_turns": context_turns,
        "visual_mode": visual_mode, "max_locals": max_locals, "global_nframes": global_nframes,
        "locals_topup": locals_topup, "require_visuals": require_visuals,
        "lm_name": lm_name, "dino_name": dino_name, "dino_tune_mode": dino_tune_mode, "dino_last_n": dino_last_n,
        "epochs": epochs, "batch_size": batch_size,
        "lr_proj": lr_proj, "lr_dino": lr_dino, "warmup_steps": warmup_steps,
        "dino_checkpoint": dino_checkpoint, "dino_checkpoint_key": dino_checkpoint_key,
        "dino_checkpoint_strict": dino_checkpoint_strict,
        "col_transcript": col_transcript, "col_frames": col_frames,
        "max_len": max_len, "num_workers": num_workers,
        "fallback_split": fallback_split, "split_k": split_k, "split_sec": split_sec,
        "log_interval": log_interval, "limit_train_steps": limit_train_steps, "limit_val_steps": limit_val_steps,
        "dino_local_batch": dino_local_batch, "dino_input_size": dino_input_size,
        "metrics_csv": metrics_csv, "resume_path": resume_path,
        "save_every_epochs": save_every_epochs, "save_adapter_only": save_adapter_only,
        "grad_clip": grad_clip, "caption_nextword": caption_nextword,
        "train_ablation_mode": train_ablation_mode, "eval_ablations": eval_ablations,
    }
    if rank == 0:
        with open(os.path.join(output_dir, "run_config.json"), "w") as f: json.dump(run_config, f, indent=2)

    # Build splits & JSONL on rank0
    if rank == 0:
        df_all = pd.read_csv(manifest_csv)

        normalize_cols = ["global_vec","locals_npz","meta_pkl","frames_dir","frames_path","frames_root"]
        if not caption_nextword:
            normalize_cols += ["transcript_json","transcripts_json","transcript_path","transcript","asr_json"]

        for col in normalize_cols:
            if col in df_all.columns:
                df_all[col] = df_all[col].apply(lambda x: path_join(parent_dir, x))

        use_lists = bool(train_id_list or test_id_list or val_id_list)
        splits = {}

        if use_lists:
            # helper: load ids from .txt (one per line) or .csv (with clip_id column)
            def _load_id_set_local(path: str) -> set[str]:
                if not path: return set()
                p = Path(path)
                if not p.exists(): raise FileNotFoundError(f"ID list not found: {path}")
                if p.suffix.lower() == ".txt":
                    return set(ln.strip() for ln in open(path, "r") if ln.strip())
                import csv as _csv
                with open(path, newline="") as f:
                    r = _csv.DictReader(f); cols = r.fieldnames or []
                    pick = "clip_id" if "clip_id" in cols else next((c for c in cols if "clip" in c.lower()), None)
                    if not pick:
                        raise ValueError(f"No 'clip_id' (or *clip*) column in {path}. Columns={cols}")
                    return set(row[pick].strip() for row in r if row.get(pick))

            train_ids = _load_id_set_local(train_id_list) if train_id_list else set()
            test_ids  = _load_id_set_local(test_id_list)  if test_id_list  else set()
            val_ids   = _load_id_set_local(val_id_list)   if val_id_list   else set()

            if not train_ids:
                if test_ids:
                    all_ids = set(df_all["clip_id"].astype(str).tolist())
                    train_ids = all_ids - test_ids
                else:
                    raise ValueError("Provide at least --train-id-list or --test-id-list.")

            if not val_ids:
                rng = np.random.RandomState(seed)
                train_ids_list = list(train_ids); rng.shuffle(train_ids_list)
                k_val = int(round(val_frac_of_train * len(train_ids_list)))
                val_ids = set(train_ids_list[:k_val]); train_ids = set(train_ids_list[k_val:])

            df_all["clip_id"] = df_all["clip_id"].astype(str)
            splits["train"] = df_all[df_all["clip_id"].isin(train_ids)].copy()
            splits["val"]   = df_all[df_all["clip_id"].isin(val_ids)].copy()
            splits["test"]  = df_all[df_all["clip_id"].isin(test_ids)].copy()
        else:
            # fall back to scene-root split
            splits = build_splits(manifest_csv, parent_dir, seed, train_frac, val_frac)

        for name, df in splits.items():
            out_csv = os.path.join(split_dir, f"clips_{name}.csv"); df.to_csv(out_csv, index=False)
            rows = df.to_dict(orient="records")
            if caption_nextword:
                rows_next = build_caption_nextword_rows(
                    rows, max_locals=max_locals, col_transcript=col_transcript, align_eps=align_eps
                )
                ann = "nextword"
            else:
                rows_next = expand_to_nextutt_rows(
                    rows, context_turns=context_turns, visual_mode=visual_mode, max_locals=max_locals,
                    col_transcript=col_transcript, col_frames=col_frames,
                    fallback_split=fallback_split, split_k=split_k, split_sec=split_sec,
                    global_nframes=global_nframes, locals_topup=locals_topup,
                    require_visuals=require_visuals, align_eps=align_eps,
                )
                ann = "nextutt"
            out_jsonl = os.path.join(json_dir, f"{name}_{ann}.jsonl"); write_jsonl(out_jsonl, rows_next)
            print(f"[rank0] {name}: {len(df)} clips → {len(rows_next)} samples | wrote {out_csv} & {out_jsonl}")
            audit_frames(out_jsonl, os.path.join(log_dir, f"frames_audit_{name}.csv"), N=5000)
    if is_dist: dist.barrier()


    # Models
    gemma = GemmaWithInjection(lm_name, add_tokens=[SOC_G, SOC_L], torch_dtype=torch.bfloat16).to(device)
    with torch.no_grad():
        emb = gemma.lm.get_input_embeddings().weight.detach().float()
        lm_rms = emb.pow(2).mean(dim=1, keepdim=True).sqrt().median().item()
    H = gemma.lm.config.hidden_size
    projector = VisualProjector(in_dim=768, out_dim=H, use_scaleshift=True,
                                init_scale_in=1.0, init_scale_out=lm_rms, dropout=0.1).to(device).to(torch.float32)
    dino = DINOEncoder(model_name=dino_name, tune_mode=dino_tune_mode, last_n=dino_last_n,
                       checkpoint_path=dino_checkpoint, checkpoint_key=dino_checkpoint_key,
                       checkpoint_strict=dino_checkpoint_strict,
                       input_size=dino_input_size, autocast_dtype=torch.bfloat16).to(device)

    if is_dist:
        projector = DDP(projector, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if any(p.requires_grad for p in dino.parameters()):
            dino = DDP(dino, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Data
    suffix = "nextword" if caption_nextword else "nextutt"
    train_ds = NextUttJsonlDataset(os.path.join(json_dir, f"train_{suffix}.jsonl"))
    val_ds   = NextUttJsonlDataset(os.path.join(json_dir, f"val_{suffix}.jsonl"))
    test_ds  = NextUttJsonlDataset(os.path.join(json_dir, f"test_{suffix}.jsonl"))
    if rank == 0:
        print(f"[sizes] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
        if len(train_ds) > 0:
            s = train_ds.rows[0]
            print("[sample] prompt_text:", (s.get("prompt_text","")[:200]).replace("\n"," "))
            print("[sample] target_text:", (s.get("target_text","")[:200]).replace("\n"," "))

    collate = Collator(gemma.tokenizer, visual_mode=visual_mode, max_len=max_len, max_locals=max_locals)
    train_sampler = DistributedSampler(train_ds, world, rank, shuffle=True, seed=seed) if is_dist else None
    val_sampler   = DistributedSampler(val_ds,   world, rank, shuffle=False, seed=seed) if is_dist else None
    test_sampler  = DistributedSampler(test_ds,  world, rank, shuffle=False, seed=seed) if is_dist else None

    dl_kwargs = dict(batch_size=batch_size, pin_memory=True, collate_fn=collate)
    if num_workers > 0: dl_kwargs.update(num_workers=num_workers, persistent_workers=True, prefetch_factor=2)
    else:               dl_kwargs.update(num_workers=0)
    train_loader = DataLoader(train_ds, shuffle=(train_sampler is None), sampler=train_sampler, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, sampler=val_sampler, **{**dl_kwargs, "num_workers": max(1, num_workers//2) if num_workers>0 else 0})
    test_loader  = DataLoader(test_ds,  shuffle=False, sampler=test_sampler, **{**dl_kwargs, "num_workers": max(1, num_workers//2) if num_workers>0 else 0})

    # Optimizer & sched
    params = [{"params": projector.parameters(), "lr": lr_proj, "weight_decay": 0.05}]
    trainable_dino = [p for p in dino.parameters() if p.requires_grad]
    if trainable_dino:
        params.append({"params": trainable_dino, "lr": lr_dino, "weight_decay": 0.05})
    opt = torch.optim.AdamW(params)
    total_steps = max(1, len(train_loader) * epochs)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Resume
    start_epoch, best_val = 1, float("inf")
    if resume_path:
        start_epoch, best_val = load_resume(resume_path, projector, dino, opt, sched, map_location=device)

    # Save tokenizer/config + special tokens
    if rank == 0:
        gemma.tokenizer.save_pretrained(output_dir)
        if hasattr(gemma.lm, "config"): gemma.lm.config.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "special_token_ids.json"), "w") as f:
            json.dump({"SOC_G": gemma.id_soc_g, "SOC_L": gemma.id_soc_l, "token_strings": [SOC_G, SOC_L]}, f, indent=2)

    metrics_path = metrics_csv if metrics_csv else os.path.join(log_dir, "metrics.csv")
    if rank == 0: init_metrics_csv(metrics_path)
    img_cache = ImageCache(max_items=2048, verbose=False)

    def to_pdtype(x: torch.Tensor, mdl: nn.Module) -> torch.Tensor:
        p = (mdl.module if isinstance(mdl, DDP) else mdl).parameters()
        dtype = next(p).dtype if p is not None else torch.float32
        return x.to(device=device, dtype=dtype)

    def run_epoch(loader, epoch_idx: int, train: bool, inject_visuals: bool, split_name: str, ablation_mode: str = "both"):
        if train:
            projector.train(True)
            if isinstance(dino, DDP): dino.train(any(p.requires_grad for p in dino.module.parameters()))
            else: dino.train(any(p.requires_grad for p in dino.parameters()))
            if is_dist and hasattr(loader.sampler, "set_epoch"): loader.sampler.set_epoch(epoch_idx)
        else:
            projector.train(False)
            if isinstance(dino, DDP): dino.train(False)
            else: dino.train(False)
        gemma.train(False)

        local_loss_sum, local_tok_sum = 0.0, 0
        running_loss, running_tok = 0.0, 0
        g_has_total = 0; l_has_total = 0

        for step, batch in enumerate(loader):
            if batch["input_ids"].size(0) == 0: continue
            t0 = time.time()

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            B = input_ids.size(0)

            proj_g = None
            proj_l_list: List[Optional[torch.Tensor]] = [None]*B

            if inject_visuals:
                if visual_mode == "frames":
                    # global - skip in local_only mode
                    if ablation_mode != "local_only":
                        flat_g_imgs, g_owner = [], []
                        for b in range(B):
                            paths = batch["global_frames"][b] or []
                            for pth in paths:
                                im = img_cache.get(pth)
                                if im is not None:
                                    flat_g_imgs.append(im); g_owner.append(b)
                        g_has = set()
                        if flat_g_imgs:
                            g_cls_all = dino_encode_images_in_chunks(dino, flat_g_imgs, chunk=dino_local_batch)
                            g_cls_all = to_pdtype(g_cls_all, projector)
                            sum_g = torch.zeros(B, 768, device=device, dtype=g_cls_all.dtype)
                            cnt_g = torch.zeros(B,   1, device=device, dtype=g_cls_all.dtype)
                            for i, b in enumerate(g_owner):
                                sum_g[b] += g_cls_all[i]; cnt_g[b] += 1; g_has.add(b)
                            mean_g = sum_g / torch.clamp(cnt_g, min=1.0)
                            proj_g = projector(mean_g)
                        else:
                            proj_g = projector(torch.zeros(B, 768, device=device, dtype=next(projector.parameters()).dtype))
                        g_has_total += len(g_has)
                    else:
                        proj_g = None

                    # locals - skip in global_only mode
                    if ablation_mode != "global_only":
                        flat_l_imgs, l_owner = [], []
                        for b in range(B):
                            lpaths = batch["local_frames"][b] or []
                            had = 0
                            for pth in lpaths:
                                im = img_cache.get(pth)
                                if im is not None:
                                    flat_l_imgs.append(im); l_owner.append(b); had += 1
                            if had > 0: l_has_total += 1
                        if flat_l_imgs:
                            l_cls_all = dino_encode_images_in_chunks(dino, flat_l_imgs, chunk=dino_local_batch)
                            l_cls_all = to_pdtype(l_cls_all, projector)
                            per_b = {}
                            for i, b in enumerate(l_owner):
                                per_b.setdefault(b, []).append(l_cls_all[i])
                            for b, vecs in per_b.items():
                                l_cls = torch.stack(vecs, dim=0)
                                proj_l_list[b] = projector(l_cls)
                else:
                    # vectors
                    # Skip computing global projections in local_only mode to avoid NaN
                    if ablation_mode != "local_only":
                        g_vecs = []
                        g_has_step = 0
                        for pth in batch["global_vecs"]:
                            if pth and os.path.exists(pth):
                                g = np.load(pth).astype(np.float32); g_has_step += 1
                            else:
                                g = np.zeros((768,), dtype=np.float32)
                            g_vecs.append(torch.from_numpy(g))
                        g_has_total += g_has_step
                        g_cls = torch.stack(g_vecs, dim=0)
                        g_cls = to_pdtype(g_cls, projector)
                        proj_g = projector(g_cls)
                    else:
                        proj_g = None

                    # Skip computing local projections in global_only mode to avoid NaN
                    if ablation_mode != "global_only":
                        order_modes = batch.get("locals_order_mode", [False]*B)

                        for b in range(B):
                            npz_path = batch["locals_npzs"][b]
                            word_ids = batch.get("loc_word_ids", [[]])[b] if isinstance(batch.get("loc_word_ids", None), list) else []
                            order_mode = bool(order_modes[b] if isinstance(order_modes, list) else order_modes)

                            if npz_path and os.path.exists(npz_path) and word_ids:
                                arr = np.load(npz_path, allow_pickle=True)
                                vecs = []
                                if order_mode:
                                    pos2key = _npz_build_pos2key(arr)
                                    for pos in word_ids[:max_locals]:
                                        k = pos2key.get(int(pos))
                                        if k is not None:
                                            v = arr[k]
                                            vecs.append(torch.from_numpy(v.astype(np.float32)))
                                else:
                                    for wid in word_ids[:max_locals]:
                                        if wid in arr: v = arr[wid]
                                        elif str(wid) in arr: v = arr[str(wid)]
                                        else:
                                            v = None
                                            for pat in (f"w_{wid}", f"idx_{wid}", f"word_{wid}", f"{wid:05d}"):
                                                if pat in arr: v = arr[pat]; break
                                            if v is None:
                                                # final fallback: digits in keys
                                                pos2key = _npz_build_pos2key(arr)
                                                k = pos2key.get(int(wid))
                                                if k is not None:
                                                    v = arr[k]
                                        if v is not None:
                                            vecs.append(torch.from_numpy(v.astype(np.float32)))
                                if vecs:
                                    l = torch.stack(vecs, dim=0)
                                    # Check for NaN/Inf in input embeddings
                                    if torch.isnan(l).any() or torch.isinf(l).any():
                                        if rank == 0:
                                            print(f"[ERROR] NaN/Inf detected in local embeddings for batch {b}, skipping")
                                        continue
                                    l = to_pdtype(l, projector)

                                    # Debug: print stats before projection
                                    if rank == 0 and step == 0 and b < 4:
                                        print(f"[DEBUG] Batch {b}: input shape={l.shape}, mean={l.mean().item():.6f}, std={l.std().item():.6f}, min={l.min().item():.6f}, max={l.max().item():.6f}")

                                    proj_l = projector(l)

                                    # Debug: print stats after projection
                                    if rank == 0 and step == 0 and b < 4:
                                        print(f"[DEBUG] Batch {b}: output shape={proj_l.shape}, mean={proj_l.mean().item() if not torch.isnan(proj_l).any() else 'NaN'}, has_nan={torch.isnan(proj_l).any()}, has_inf={torch.isinf(proj_l).any()}")

                                    # Check for NaN/Inf in projector output
                                    if torch.isnan(proj_l).any() or torch.isinf(proj_l).any():
                                        if rank == 0:
                                            print(f"[ERROR] NaN/Inf in projector output for batch {b}, skipping")
                                        continue
                                    proj_l_list[b] = proj_l
                                    l_has_total += 1

            out = gemma(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                proj_global=proj_g, proj_locals=proj_l_list, inject_visuals=inject_visuals,
                ablation_mode=ablation_mode
            )
            loss = out.loss

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(projector.parameters(), grad_clip)
                    if any(p.requires_grad for p in dino.parameters()):
                        torch.nn.utils.clip_grad_norm_(dino.parameters(), grad_clip)
                opt.step(); sched.step()

            n_lab_tok = int((labels != -100).sum().item())
            local_loss_sum += loss.item() * max(n_lab_tok, 1)
            local_tok_sum  += max(n_lab_tok, 1)
            running_loss   += loss.item() * max(n_lab_tok, 1)
            running_tok    += max(n_lab_tok, 1)

            dt = (time.time() - t0)
            toks = int(input_ids.numel())
            if log_interval > 0 and ((step + 1) % log_interval == 0) and (rank == 0):
                avg = (running_loss / max(running_tok, 1))
                ppl = math.exp(min(20.0, max(avg, 1e-8)))
                tok_per_s  = toks / max(1e-6, dt)
                gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                print(f"[rank0] {split_name} ep{epoch_idx} step {step+1}/{len(loader)} "
                      f"avg_loss={avg:.4f} ppl={ppl:.2f} tok={running_tok} tok/s={tok_per_s:.0f} "
                      f"dt={dt:.3f}s mem={gpu_mem_mb:.0f}MB g_has={g_has_total} l_has={l_has_total} mode={ablation_mode}")
                write_metric_row(metrics_path, split=split_name, epoch=epoch_idx, step=step+1, loss=avg, ppl=ppl,
                                 tokens=toks, tok_per_s=tok_per_s, dt_s=dt,
                                 lr_proj=opt.param_groups[0]["lr"],
                                 lr_dino=opt.param_groups[-1]["lr"] if len(opt.param_groups)>1 else 0.0,
                                 gpu_mem_mb=gpu_mem_mb, g_has=g_has_total, l_has=l_has_total,
                                 ablation_mode=ablation_mode)
                running_loss, running_tok = 0.0, 0

            max_steps = limit_train_steps if train else limit_val_steps
            if max_steps and (step + 1) >= max_steps: break

        if is_dist:
            t = torch.tensor([local_loss_sum, float(local_tok_sum), float(g_has_total), float(l_has_total)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss, total_tok = t[0].item(), int(t[1].item())
            g_has_total, l_has_total = int(t[2].item()), int(t[3].item())
        else:
            total_loss, total_tok = local_loss_sum, local_tok_sum

        avg_loss = total_loss / max(total_tok, 1)
        ppl = math.exp(min(20.0, max(avg_loss, 1e-8)))
        if rank == 0:
            write_metric_row(metrics_path, split=split_name, epoch=epoch_idx, step=-1, loss=avg_loss, ppl=ppl,
                             gpu_mem_mb=torch.cuda.max_memory_allocated(device)/(1024**2),
                             g_has=g_has_total, l_has=l_has_total, ablation_mode=ablation_mode)
        return avg_loss, ppl

    best_epoch = start_epoch - 1
    train_ablation = run_config.get("train_ablation_mode", "both")
    eval_ablations = run_config.get("eval_ablations", ["both", "none"])

    # Determine primary mode for metrics (prefer "both", fallback to first available)
    primary_mode = "both" if "both" in eval_ablations else eval_ablations[0]
    training_occurred = True

    # Validate training configuration
    if train_ablation == "none":
        if rank == 0:
            print("[WARNING] Cannot train with ablation_mode='none' (no trainable parameters).")
            print("[WARNING] Switching to evaluation-only mode. Use 'both', 'global_only', or 'local_only' for training.")
        # Skip training, just do evaluation
        start_epoch = epochs + 1
        training_occurred = False

    for ep in range(start_epoch, epochs+1):
        # Training with specified ablation mode
        # Set inject_visuals=False when train_ablation="none" to avoid gradient errors
        inject_train = (train_ablation != "none")
        tr_loss, tr_ppl = run_epoch(train_loader, ep, train=True, inject_visuals=inject_train,
                                     split_name="train", ablation_mode=train_ablation)

        # Evaluation with multiple ablation modes
        val_results = {}
        for abl_mode in eval_ablations:
            inject = (abl_mode != "none")
            split_suffix = abl_mode if abl_mode != "both" else "vis"
            split_suffix = "novis" if abl_mode == "none" else split_suffix
            v_loss, v_ppl = run_epoch(val_loader, ep, train=False, inject_visuals=inject,
                                      split_name=f"val_{split_suffix}", ablation_mode=abl_mode)
            val_results[abl_mode] = (v_loss, v_ppl)

        if rank == 0:
            ppl_str = " | ".join([f"{mode}={val_results[mode][1]:.2f}" for mode in eval_ablations])
            print(f"[Epoch {ep:02d}] train({train_ablation}) ppl {tr_ppl:.2f} | val ppl: {ppl_str}")
            write_metric_row(metrics_path, split="train_summary", epoch=ep, step=-1, loss=tr_loss, ppl=tr_ppl,
                             ablation_mode=train_ablation)
            # Write summary with backward-compatible column names
            summary_kw = {"split": "val_summary", "epoch": ep, "step": -1}
            if "both" in val_results:
                summary_kw["val_ppl_vis"] = val_results["both"][1]
            if "none" in val_results:
                summary_kw["val_ppl_no_vis"] = val_results["none"][1]
            write_metric_row(metrics_path, **summary_kw)

        # Use "both" mode results for checkpointing (or first available mode)
        primary_mode = "both" if "both" in val_results else eval_ablations[0]
        primary_val_ppl = val_results[primary_mode][1]

        if (ep - start_epoch) % max(1, save_every_epochs) == 0:
            save_checkpoint("last", ep, primary_val_ppl, output_dir, projector, dino, opt, sched, run_config, gemma.tokenizer, rank, save_adapter_only)
        if primary_val_ppl < best_val:
            best_val, best_epoch = primary_val_ppl, ep
            save_checkpoint("best", ep, primary_val_ppl, output_dir, projector, dino, opt, sched, run_config, gemma.tokenizer, rank, save_adapter_only)

    # Test with all ablation modes
    test_results = {}
    for abl_mode in eval_ablations:
        inject = (abl_mode != "none")
        split_suffix = abl_mode if abl_mode != "both" else "vis"
        split_suffix = "novis" if abl_mode == "none" else split_suffix
        t_loss, t_ppl = run_epoch(test_loader, epochs+1, train=False, inject_visuals=inject,
                                  split_name=f"test_{split_suffix}", ablation_mode=abl_mode)
        test_results[abl_mode] = (t_loss, t_ppl)

    if rank == 0:
        test_ppl_str = " | ".join([f"{mode}={test_results[mode][1]:.2f}" for mode in eval_ablations])
        print(f"[TEST] ppl: {test_ppl_str}")
        print(f"[best] epoch={best_epoch} val_ppl({primary_mode})={best_val:.4f} | checkpoints → {os.path.join(output_dir,'checkpoints')}")
        # Write summary with backward-compatible column names
        test_summary_kw = {"split": "test_summary", "epoch": epochs+1, "step": -1}
        if "both" in test_results:
            test_summary_kw["test_ppl_vis"] = test_results["both"][1]
        if "none" in test_results:
            test_summary_kw["test_ppl_no_vis"] = test_results["none"][1]
        write_metric_row(metrics_path, **test_summary_kw)
    if is_dist:
        dist.barrier(); dist.destroy_process_group()

# ---------------------- I/O ----------------------
def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--context-turns", type=int, default=2)
    ap.add_argument("--visual-mode", type=str, choices=["frames","vectors"], default="vectors")
    ap.add_argument("--max-locals", type=int, default=50)
    ap.add_argument("--global-nframes", type=int, default=4)
    ap.add_argument("--locals-topup", action="store_true")
    ap.add_argument("--require-visuals", action="store_true")
    ap.add_argument("--lm-name", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--dino-name", type=str, default="vit_base_patch14_dinov2")
    ap.add_argument("--dino-tune-mode", type=str, choices=["frozen","cls_param_only","cls_adapter","last_n","full"], default="cls_adapter")
    ap.add_argument("--dino-last-n", type=int, default=0)
    ap.add_argument("--dino-checkpoint", type=str, default="")
    ap.add_argument("--dino-checkpoint-key", type=str, default="")
    ap.add_argument("--dino-checkpoint-strict", action="store_true")
    ap.add_argument("--col-transcript", type=str, default="")
    ap.add_argument("--col-frames", type=str, default="")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--fallback-split", type=str, choices=["none","fixed_words","fixed_seconds"], default="none")
    ap.add_argument("--split-k", type=int, default=12)
    ap.add_argument("--split-sec", type=float, default=1.5)
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--limit-train-steps", type=int, default=0)
    ap.add_argument("--limit-val-steps", type=int, default=0)
    ap.add_argument("--dino-local-batch", type=int, default=256)
    ap.add_argument("--dino-input-size", type=int, default=518)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-proj", type=float, default=3e-4)
    ap.add_argument("--lr-dino", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--align-eps", type=float, default=0.5,
                    help="(unused in nextword; kept for parity)")
    ap.add_argument("--metrics-csv", type=str, default="")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--save-every-epochs", type=int, default=1)
    ap.add_argument("--save-adapter-only", action="store_true")
    # NEW
    ap.add_argument("--caption-nextword", action="store_true",
                    help="Single-caption mode: emit next-word samples per caption (vectors recommended).")
    ap.add_argument("--train-id-list", type=str, default="",
                    help="Path to text/CSV file listing train clip_id values (one per line or a 'clip_id' column).")
    ap.add_argument("--val-id-list", type=str, default="",
                    help="Optional path listing validation clip_id values.")
    ap.add_argument("--test-id-list", type=str, default="",
                    help="Path to text/CSV file listing test clip_id values.")
    ap.add_argument("--val-frac-of-train", type=float, default=0.1,
                    help="If no --val-id-list provided, take this fraction of train ids to form val.")
    # Ablation study arguments
    ap.add_argument("--train-ablation-mode", type=str, default="both",
                    choices=["both", "global_only", "local_only", "none"],
                    help="Which visual tokens to use during training: 'both' (default), 'global_only', 'local_only', or 'none'.")
    ap.add_argument("--eval-ablations", type=str, nargs="+", default=["both", "none"],
                    choices=["both", "global_only", "local_only", "none"],
                    help="List of ablation modes to evaluate (default: both none). Use to run comprehensive ablation studies.")

    args = ap.parse_args()

    print(f"[whoami] __file__={__file__}")
    print(f"[whoami] parent_dir={args.parent_dir}")
    print(f"[whoami] caption_nextword={getattr(args, 'caption_nextword', False)} col_transcript={args.col_transcript!r}")

    train_and_eval(
        parent_dir=args.parent_dir, output_dir=args.output_dir, seed=args.seed,
        train_frac=args.train_frac, val_frac=args.val_frac, context_turns=args.context_turns,
        visual_mode=args.visual_mode, max_locals=args.max_locals,
        global_nframes=args.global_nframes, locals_topup=args.locals_topup, require_visuals=args.require_visuals,
        lm_name=args.lm_name, dino_name=args.dino_name, dino_tune_mode=args.dino_tune_mode, dino_last_n=args.dino_last_n,
        epochs=args.epochs, batch_size=args.batch_size, lr_proj=args.lr_proj, lr_dino=args.lr_dino, warmup_steps=args.warmup_steps,
        dino_checkpoint=args.dino_checkpoint, dino_checkpoint_key=args.dino_checkpoint_key, dino_checkpoint_strict=args.dino_checkpoint_strict,
        col_transcript=args.col_transcript, col_frames=args.col_frames, max_len=args.max_len, num_workers=args.num_workers,
        fallback_split=args.fallback_split, split_k=args.split_k, split_sec=args.split_sec,
        log_interval=args.log_interval, limit_train_steps=args.limit_train_steps, limit_val_steps=args.limit_val_steps,
        dino_local_batch=args.dino_local_batch, dino_input_size=args.dino_input_size,
        metrics_csv=args.metrics_csv, resume_path=args.resume, save_every_epochs=args.save_every_epochs,
        save_adapter_only=args.save_adapter_only, grad_clip=args.grad_clip, align_eps=args.align_eps,
        caption_nextword=args.caption_nextword, train_id_list=args.train_id_list, val_id_list=args.val_id_list, test_id_list=args.test_id_list, val_frac_of_train=args.val_frac_of_train,
        train_ablation_mode=args.train_ablation_mode, eval_ablations=args.eval_ablations
    )

if __name__ == "__main__":
    main()

