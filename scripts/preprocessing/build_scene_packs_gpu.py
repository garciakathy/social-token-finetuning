#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, glob, cv2, json, csv, time, pickle, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import logging
from logging.handlers import RotatingFileHandler

# ----------------------
# CLI
# ----------------------
import argparse
def get_args():
    p = argparse.ArgumentParser(description="Build & store Social Token Packs (resume-safe, GPU-accelerated).")
    p.add_argument("--root_split_dir", required=True,
                   help="Root of split (e.g., .../naturalistic/train). Expects <CLIP_ID>/{frames,transcripts}")
    p.add_argument("--out_dir", required=True, help="Output directory for packs + manifest/logs")
    p.add_argument("--dino_ckpt", required=True, help="Path to finetuned DINO ViT-B/14 checkpoint")
    p.add_argument("--model_name", default="vit_base_patch14_dinov2.lvd142m",
                   help="timm model name (backbone-only)")
    p.add_argument("--fps", type=float, default=1.0, help="Frame rate at which frames were saved (default 1.0)")
    p.add_argument("--max_global_frames", type=int, default=120, help="Max frames to sample for global vector")
    p.add_argument("--pad_sec", type=float, default=0.75, help="+/- seconds around each noun/verb window")
    p.add_argument("--top_k_locals", type=int, default=12, help="Max noun/verb-local tokens to keep")
    p.add_argument("--limit", type=int, default=0, help="Process at most N clips (0 = all)")
    p.add_argument("--log_file", default="scene_packs.log", help="Main log filename (created under out_dir)")
    p.add_argument("--capture_stdio", action="store_true", help="Also tee stdout/stderr into main log")
    p.add_argument("--dry_run", action="store_true", help="List what would run, but do nothing")
    p.add_argument("--device", default="auto", choices=["auto","cuda","cpu"],
                   help="Device selection; 'auto' picks CUDA if available")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for frame embedding")
    p.add_argument("--overwrite", action="store_true", help="Rebuild packs even if outputs already exist")
    p.add_argument("--resume", action="store_true", help="(Default behavior) Skip clips with existing outputs")
    return p.parse_args()

# ----------------------
# Logging
# ----------------------
def setup_logging(out_dir: str, main_log_name: str, capture_stdio: bool):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    main_log_path = Path(out_dir) / main_log_name
    fail_log_path = main_log_path.with_name(main_log_path.stem + "_failures.log")

    logger = logging.getLogger("scene_packs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Rotating main log
    fh = RotatingFileHandler(main_log_path, maxBytes=10_000_000, backupCount=3)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Console too
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # Separate failures logger
    fail_logger = logging.getLogger("scene_packs.failures")
    fail_logger.setLevel(logging.WARNING)
    fail_logger.handlers.clear()
    fh_fail = RotatingFileHandler(fail_log_path, maxBytes=5_000_000, backupCount=2)
    fh_fail.setFormatter(fmt)
    fh_fail.setLevel(logging.WARNING)
    fail_logger.addHandler(fh_fail)

    # Optional stdio capture
    if capture_stdio:
        class Tee(object):
            def __init__(self, name, stream):
                self.file = open(name, "a", buffering=1)
                self.stream = stream
            def write(self, data):
                self.file.write(data)
                self.stream.write(data)
            def flush(self):
                self.file.flush()
                self.stream.flush()

        sys.stdout = Tee(str(main_log_path), sys.stdout)
        sys.stderr = Tee(str(main_log_path), sys.stderr)

    return logger, fail_logger

# ----------------------
# Data helpers
# ----------------------
def load_frame_paths(frames_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))

def sec_to_frame_idx(t_sec: float, fps: float = 1.0) -> int:
    return int(round(float(t_sec) * float(fps)))

# ----------------------
# Transcript -> words
# (your updated function, unchanged)
# ----------------------
def format_json(data):
    """
    Expecting:
      data["metadata:transcript"] = [
        {"words": [{"word": "I", "start": 1.23, "end": 1.45}, ...]}, ...
      ]
    Keeps only words with valid start/end and ending <= 120s.
    """
    return {
        "words": [
            {
                "text": w["word"],
                "start": float(w["start"]),
                "end": float(w["end"]),
            }
            for seg in data.get("metadata:transcript", [])
            for w in seg.get("words", [])
            if w.get("start") is not None
            and w.get("end") is not None
            and float(w["end"]) <= 120.0
        ]
    }

# ----------------------
# POS tagging
# ----------------------
import spacy
# Use transformer model if available; fallback to sm
try:
    _NLP = spacy.load("en_core_web_trf", disable=["ner","parser"])
except Exception:
    _NLP = spacy.load("en_core_web_sm", disable=["ner","parser"])
KEEP_POS = {"NOUN", "PROPN", "VERB", "AUX"}

def tag_nouns_verbs(words: List[Dict[str, Any]]) -> List[int]:
    """
    Returns indices of words that POS-tag as NOUN/PROPN/VERB/AUX.
    Robust to punctuation-only words; processes per-word for reliable alignment.
    """
    texts = [w["text"] for w in words]
    idxs = []
    for i, doc in enumerate(_NLP.pipe(texts, batch_size=512)):
        tok = next((t for t in doc if not t.is_space), None)
        if tok is None or tok.is_punct:
            continue
        if tok.pos_ in KEEP_POS:
            idxs.append(i)
    return idxs

def window_for_word(words: List[Dict[str, Any]], wi: int, pad: float = 0.75) -> Tuple[float,float]:
    s = max(0.0, float(words[wi]["start"]) - pad)
    e = float(words[wi]["end"]) + pad
    return s, e

# ----------------------
# DINO backbone + CLS extraction
# ----------------------
def select_device(arg: str) -> torch.device:
    if arg == "cuda": return torch.device("cuda")
    if arg == "cpu": return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dino(checkpoint_path: str,
              model_name: str,
              device: torch.device):
    """
    Returns:
      model_backbone (eval, on device),
      preprocess (PIL -> normalized tensor),
      cls_embed(tensor[B,3,H,W]) -> tensor[B,D] normalized
    """
    model_backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    model_backbone.eval().to(device)

    sd = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(sd, dict) and "student_backbone" in sd:
        model_backbone.load_state_dict(sd["student_backbone"], strict=False)
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

    return model_backbone, preprocess, cls_embed

# ----------------------
# Embedding utilities (batched)
# ----------------------
def pil_from_bgr(bgr) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

@torch.no_grad()
def embed_frames_batched(paths: List[str], preprocess, cls_embed, batch_size: int) -> np.ndarray:
    """
    Returns array [N, D] of CLS embeddings for frames at given paths.
    """
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
            vecs.append(v)
            batch_imgs.clear()
    if batch_imgs:
        x = torch.stack(batch_imgs, 0)
        v = cls_embed(x).cpu().numpy()
        vecs.append(v)
    if not vecs:
        return np.zeros((0, 768), dtype=np.float32)
    return np.concatenate(vecs, axis=0)

def mean_pool(arrs: List[np.ndarray]) -> np.ndarray:
    if len(arrs) == 0: return np.zeros((768,), dtype=np.float32)
    if len(arrs) == 1: return arrs[0]
    return np.stack(arrs, 0).mean(0)

@torch.no_grad()
def global_social_vector_from_dir(preprocess, cls_embed, frames_dir: str,
                                  max_frames: Optional[int], batch_size: int) -> np.ndarray:
    paths = load_frame_paths(frames_dir)
    if not paths:
        raise FileNotFoundError(f"No frames in {frames_dir}")
    if max_frames:
        step = max(1, len(paths)//max_frames)
        paths = paths[::step]
    embs = embed_frames_batched(paths, preprocess, cls_embed, batch_size)
    return embs.mean(0) if embs.shape[0] else np.zeros((768,), dtype=np.float32)

@torch.no_grad()
def local_social_vector_for_word(preprocess, cls_embed, frames_dir: str,
                                 words: List[Dict[str,Any]], wi: int,
                                 fps: float, pad: float, batch_size: int) -> Optional[np.ndarray]:
    s, e = window_for_word(words, wi, pad=pad)
    paths = load_frame_paths(frames_dir)
    if not paths:
        return None
    start_idx = max(0, sec_to_frame_idx(s, fps=fps))
    end_idx   = min(len(paths)-1, sec_to_frame_idx(e, fps=fps))
    if end_idx < start_idx:
        return None
    sub = paths[start_idx:end_idx+1]
    embs = embed_frames_batched(sub, preprocess, cls_embed, batch_size)
    if embs.shape[0] == 0:
        return None
    return embs.mean(0)

# ----------------------
# Pack build/save
# ----------------------
def build_social_token_pack(preprocess, cls_embed,
                            clip_id: str, frames_dir: str, transcript_json_path: str,
                            fps: float, max_global_frames: int, pad: float, top_k_locals: int,
                            batch_size: int, logger: logging.Logger):
    with open(transcript_json_path, "r") as f:
        raw = json.load(f)
    trans = format_json(raw)
    words = trans["words"]

    global_vec = global_social_vector_from_dir(
        preprocess, cls_embed, frames_dir, max_frames=max_global_frames, batch_size=batch_size
    )

    nounverb_idxs = tag_nouns_verbs(words)[:top_k_locals]

    locals_list = []
    for wi in nounverb_idxs:
        v = local_social_vector_for_word(
            preprocess, cls_embed, frames_dir, words, wi, fps=fps, pad=pad, batch_size=batch_size
        )
        if v is not None:
            locals_list.append({"wi": wi, "word": words[wi]["text"], "vec": v})

    pack = {
        "clip_id": clip_id,
        "global": global_vec,    # (D,)
        "locals": locals_list,   # list of {wi, word, vec:(D,)}
        "words": words,
    }
    logger.info(f"Built pack for {clip_id}: global={global_vec.shape}, locals={len(locals_list)}")
    return pack

def save_pack(pack: Dict[str,Any], out_dir: str) -> Tuple[str, Dict[str,str]]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(pack["clip_id"].encode()).hexdigest()

    paths = {
        "global": os.path.join(out_dir, f"{key}_global.npy"),
        "locals": os.path.join(out_dir, f"{key}_locals.npz"),
        "meta":   os.path.join(out_dir, f"{key}_meta.pkl"),
    }
    # Write files
    np.save(paths["global"], pack["global"])
    np.savez_compressed(
        paths["locals"],
        **{f"{it['wi']}_{it['word']}": it["vec"] for it in pack["locals"]}
    )
    with open(paths["meta"], "wb") as f:
        pickle.dump({"clip_id": pack["clip_id"], "words": pack["words"]}, f)
    return key, paths

def outputs_exist(clip_id: str, out_dir: str) -> bool:
    key = hashlib.md5(clip_id.encode()).hexdigest()
    g = Path(out_dir) / f"{key}_global.npy"
    l = Path(out_dir) / f"{key}_locals.npz"
    m = Path(out_dir) / f"{key}_meta.pkl"
    return g.exists() and l.exists() and m.exists()

# ----------------------
# Discovery & manifest
# ----------------------
def discover_clips(root_split_dir: str) -> List[Tuple[str,str,str]]:
    """
    Returns list of (clip_id, frames_dir, transcript_json)
    Looks for: <root>/<CLIP_ID>/frames/*.jpg and <root>/<CLIP_ID>/transcripts/<CLIP_ID>.json
    """
    clips = []
    for clip_dir in sorted([p for p in Path(root_split_dir).iterdir() if p.is_dir()]):
        clip_id = clip_dir.name
        frames_dir = clip_dir / "frames"
        trans_dir  = clip_dir / "transcripts"
        json_path  = trans_dir / f"{clip_id}.json"
        if frames_dir.is_dir() and json_path.exists():
            if any(frames_dir.glob("frame_*.jpg")):
                clips.append((clip_id, str(frames_dir), str(json_path)))
    return clips

def ensure_manifest(manifest_csv: str):
    if not Path(manifest_csv).exists():
        with open(manifest_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "clip_id","key","global_vec","locals_npz","meta_pkl",
                "n_locals","n_words","frames_dir","transcript_json",
                "status","error","duration_s","device"
            ])

def append_manifest(manifest_csv: str, row: List[Any]):
    with open(manifest_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)

# ----------------------
# Main
# ----------------------
def main():
    args = get_args()
    device = select_device(args.device)

    logger, fail_logger = setup_logging(args.out_dir, args.log_file, args.capture_stdio)
    manifest_csv = str(Path(args.out_dir) / "scene_packs_manifest.csv")
    ensure_manifest(manifest_csv)

    logger.info("==== Build Social Token Packs ====")
    logger.info(f"root_split_dir={args.root_split_dir}")
    logger.info(f"out_dir={args.out_dir}")
    logger.info(f"dino_ckpt={args.dino_ckpt}")
    logger.info(f"model_name={args.model_name}")
    logger.info(f"device={device}")
    logger.info(f"batch_size={args.batch_size}")

    # Load model once
    model_backbone, preprocess, cls_embed = load_dino(args.dino_ckpt, args.model_name, device)

    # Discover work
    clips = discover_clips(args.root_split_dir)
    if args.limit and args.limit > 0:
        clips = clips[:args.limit]
    logger.info(f"Found {len(clips)} eligible clips.")

    processed = 0
    for clip_id, frames_dir, json_path in clips:
        t0 = time.time()
        try:
            if args.resume and not args.overwrite and outputs_exist(clip_id, args.out_dir):
                logger.info(f"[SKIP existing] {clip_id}")
                # still add a manifest row with status=skipped
                key = hashlib.md5(clip_id.encode()).hexdigest()
                append_manifest(manifest_csv, [
                    clip_id, key,
                    str(Path(args.out_dir)/f"{key}_global.npy"),
                    str(Path(args.out_dir)/f"{key}_locals.npz"),
                    str(Path(args.out_dir)/f"{key}_meta.pkl"),
                    "", "", frames_dir, json_path, "skipped", "", 0.0, str(device)
                ])
                continue

            if args.dry_run:
                logger.info(f"[DRY RUN] Would process {clip_id}")
                continue

            pack = build_social_token_pack(
                preprocess=preprocess,
                cls_embed=cls_embed,
                clip_id=clip_id,
                frames_dir=frames_dir,
                transcript_json_path=json_path,
                fps=args.fps,
                max_global_frames=args.max_global_frames,
                pad=args.pad_sec,
                top_k_locals=args.top_k_locals,
                batch_size=args.batch_size,
                logger=logger,
            )

            key, paths = save_pack(pack, args.out_dir)
            dt = time.time() - t0

            append_manifest(manifest_csv, [
                clip_id, key, paths["global"], paths["locals"], paths["meta"],
                len(pack["locals"]), len(pack["words"]), frames_dir, json_path,
                "ok","", f"{dt:.3f}", str(device)
            ])
            logger.info(f"[OK] {clip_id} key={key} in {dt:.2f}s")
            processed += 1

        except Exception as e:
            dt = time.time() - t0
            err = f"{type(e).__name__}: {e}"
            logger.warning(f"[FAIL] {clip_id} ({dt:.2f}s) → {err}")
            # record to failures log
            fail_logger.warning(f"{clip_id}\t{err}")
            # manifest row with error
            append_manifest(manifest_csv, [
                clip_id, "", "", "", "",
                "", "", frames_dir, json_path,
                "error", err, f"{dt:.3f}", str(device)
            ])
            # continue to next clip

    logger.info(f"Done. Processed={processed} | Total candidates={len(clips)}")
    logger.info(f"Manifest: {manifest_csv}")

if __name__ == "__main__":
    main()

