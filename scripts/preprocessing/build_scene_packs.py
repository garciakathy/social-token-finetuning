#!/usr/bin/env python3
# build_social_packs.py

import os, sys, glob, cv2, json, pickle, hashlib, argparse, logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# ------------------------- Logging -------------------------------------------

def setup_logging(log_file: str, capture_stdio: bool = True) -> logging.Logger:
    """Configure root logger to log INFO+ to console and a log file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear old handlers if re-run
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    if capture_stdio:
        class StreamToLogger:
            def __init__(self, level):
                self.level = level
                self.buf = ""
            def write(self, message):
                message = message.rstrip()
                if message:
                    logging.getLogger("STDIO").log(self.level, message)
            def flush(self):  # needed for compatibility
                pass

        sys.stdout = StreamToLogger(logging.INFO)
        sys.stderr = StreamToLogger(logging.ERROR)

    return logger

# ------------------------- DINO loader ---------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_dino(checkpoint_path: str,
              model_name: str = "vit_base_patch14_dinov2.lvd142m"):
    """
    Returns:
      model_backbone: timm ViT in eval mode (on DEVICE)
      preprocess: torchvision transform matching the model's cfg
      cls_embed(batch): function mapping (B,3,H,W) -> (B, D) normalized CLS
    """
    model_backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    model_backbone.eval().to(DEVICE)

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
    preprocess = create_transform(**cfg)  # PIL -> (3,H,W) normalized tensor

    @torch.no_grad()
    def cls_embed(imgs: torch.Tensor) -> torch.Tensor:
        feats = model_backbone.forward_features(imgs.to(DEVICE))
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
        return F.normalize(x.float(), dim=-1)  # (B, D)
    return model_backbone, preprocess, cls_embed

# ------------------------- Frame utils ---------------------------------------

def load_frame_paths(frames_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

@torch.no_grad()
def dino_cls_from_frame(frame_bgr, preprocess, cls_embed) -> np.ndarray:
    # BGR -> RGB PIL then timm transform
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    x = preprocess(pil).unsqueeze(0)         # (1,3,H,W)
    cls = cls_embed(x)                       # (1,D)
    return cls.squeeze(0).cpu().numpy()      # (D,)

def pool_cls_over_frames(cls_list: List[np.ndarray]) -> np.ndarray:
    return np.stack(cls_list, 0).mean(0) if len(cls_list) > 1 else cls_list[0]

def sec_to_frame_idx(t_sec, fps=1.0) -> int:
    return int(round(float(t_sec) * float(fps)))

@torch.no_grad()
def global_social_vector_from_dir(preprocess, cls_embed,
                                  frames_dir: str, max_frames: Optional[int]=None, stride: int=1) -> np.ndarray:
    """Mean-pool CLS over frames in a directory -> (D,) global social token."""
    paths = load_frame_paths(frames_dir)
    if not paths:
        raise FileNotFoundError(f"No *.jpg frames in {frames_dir}")
    if max_frames:
        step = max(1, len(paths)//max_frames)
        paths = paths[::step]
    cls_vecs = []
    for p in paths[::stride]:
        frame = cv2.imread(p)
        if frame is None:
            continue
        cls_vecs.append(dino_cls_from_frame(frame, preprocess, cls_embed))
    return pool_cls_over_frames(cls_vecs)

# ------------------------- Transcript + POS ----------------------------------

def format_json(data: dict) -> dict:
    """
    Expecting:
      data["metadata:transcript"] = [
        {"words": [{"word": "I", "start": 1.23, "end": 1.45}, ...]}, ...
      ]
    Keeps only words with end <= 120s.
    """
    return {
        "words": [
            {"text": w["word"], "start": float(w["start"]), "end": float(w["end"])}
            for seg in data.get("metadata:transcript", [])
            for w in seg.get("words", [])
            if "word" in w and "start" in w and "end" in w
               and w["start"] is not None and w["end"] is not None
               and float(w["end"]) <= 120.0
        ]
    }

# Load spaCy only when needed (avoid GPU conflicts on cluster init)
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_trf")
    return _nlp

KEEP_POS = {"NOUN", "PROPN", "VERB", "AUX"}  # treat AUX as verb

def tag_nouns_verbs(words: List[dict]) -> List[int]:
    """
    words: list[{"text": str, "start": float, "end": float}, ...]
    returns: list of indices i in `words` whose main token is NOUN/PROPN/VERB/AUX
    Robust to punctuation by per-token processing.
    """
    nlp = get_nlp()
    texts = [w["text"] for w in words]
    idxs = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=256, disable=["ner", "parser"])):
        tok = next((t for t in doc if not t.is_space), None)
        if tok is None or tok.is_punct:
            continue
        if tok.pos_ in KEEP_POS:
            idxs.append(i)
    return idxs

def window_for_word(words: List[dict], wi: int, pad: float=0.75) -> Tuple[float,float]:
    s = max(0.0, float(words[wi]["start"]) - pad)
    e = float(words[wi]["end"]) + pad
    return s, e

@torch.no_grad()
def local_social_vector_for_word(preprocess, cls_embed, frames_dir: str,
                                 words: List[dict], wi: int,
                                 clip_fps: float=1.0, pad: float=0.75) -> Optional[np.ndarray]:
    s, e = window_for_word(words, wi, pad=pad)
    paths = load_frame_paths(frames_dir)
    if not paths:
        return None
    start_idx = max(0, sec_to_frame_idx(s, fps=clip_fps))
    end_idx   = min(len(paths)-1, sec_to_frame_idx(e, fps=clip_fps))
    if end_idx < start_idx:
        return None

    cls_vecs = []
    for p in paths[start_idx:end_idx+1]:
        frame = cv2.imread(p)
        if frame is None:
            continue
        cls_vecs.append(dino_cls_from_frame(frame, preprocess, cls_embed))
    if not cls_vecs:
        return None
    return pool_cls_over_frames(cls_vecs)

# ------------------------- Pack building/saving -------------------------------

def build_social_token_pack(
    preprocess, cls_embed, clip_id: str, frames_dir: str, transcript_json_path: str,
    clip_fps: float=1.0, max_global_frames: int=120, pad: float=0.75, top_k_locals: int=12
) -> dict:
    with open(transcript_json_path, "r") as f:
        raw = json.load(f)
    trans = format_json(raw)
    words = trans["words"]  # list of {text, start, end}

    global_vec = global_social_vector_from_dir(
        preprocess, cls_embed, frames_dir, max_frames=max_global_frames, stride=1
    )  # (D,)

    nounverb_idxs = tag_nouns_verbs(words)
    nounverb_idxs = nounverb_idxs[:top_k_locals]  # simple cap

    locals_list = []
    for wi in nounverb_idxs:
        v = local_social_vector_for_word(preprocess, cls_embed, frames_dir, words, wi, clip_fps, pad)
        if v is not None:
            locals_list.append({"wi": wi, "word": words[wi]["text"], "vec": v})

    pack = {
        "clip_id": clip_id,
        "global": global_vec,          # (D,)
        "locals": locals_list,         # list of {wi, word, vec:(D,)}
        "words": words,                # keep for prompt injection
    }
    return pack

def save_pack(pack: dict, out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(pack["clip_id"].encode()).hexdigest()
    np.save(os.path.join(out_dir, f"{h}_global.npy"), pack["global"])
    # locals as npz (keys like "37_laughed")
    np.savez_compressed(
        os.path.join(out_dir, f"{h}_locals.npz"),
        **{f"{it['wi']}_{it['word']}": it["vec"] for it in pack["locals"]}
    )
    with open(os.path.join(out_dir, f"{h}_meta.pkl"), "wb") as f:
        pickle.dump({"clip_id": pack["clip_id"], "words": pack["words"]}, f)
    return h  # stable key for this clip

# ------------------------- Dataset traversal ---------------------------------

def find_clips(root_split_dir: str) -> List[Tuple[str, str, str]]:
    """
    root_split_dir/train/<CLIP_ID>/frames
    root_split_dir/train/<CLIP_ID>/transcripts/<CLIP_ID>.json
    Returns list of (clip_id, frames_dir, transcript_json)
    """
    out = []
    # Expect subdirs are <CLIP_ID>
    for clip_dir in sorted(glob.glob(os.path.join(root_split_dir, "*"))):
        if not os.path.isdir(clip_dir):
            continue
        clip_id = os.path.basename(clip_dir.rstrip("/"))
        frames_dir = os.path.join(clip_dir, "frames")
        trans_dir  = os.path.join(clip_dir, "transcripts")
        trans_json = os.path.join(trans_dir, f"{clip_id}.json")
        if os.path.isdir(frames_dir) and os.path.isfile(trans_json):
            out.append((clip_id, frames_dir, trans_json))
    return out

# ------------------------- Main / CLI ----------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Build & save Social Token Packs")
    ap.add_argument("--root_split_dir", required=True,
                    help="e.g., /home/.../seamless/full/preprocess/naturalistic/train")
    ap.add_argument("--out_dir", required=True,
                    help="Output dir for social token packs")
    ap.add_argument("--dino_ckpt", required=True,
                    help="Path to finetuned DINOv2 checkpoint")
    ap.add_argument("--model_name", default="vit_base_patch14_dinov2.lvd142m",
                    help="timm model name for ViT-B/14 DINOv2")
    ap.add_argument("--fps", type=float, default=1.0, help="FPS of saved frames")
    ap.add_argument("--max_global_frames", type=int, default=120,
                    help="Max frames to pool for global vector")
    ap.add_argument("--pad_sec", type=float, default=0.75,
                    help="+/- seconds around each noun/verb")
    ap.add_argument("--top_k_locals", type=int, default=12,
                    help="Max local tokens (noun/verb spans) to keep")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N clips (0 = all)")
    ap.add_argument("--log_file", default="build_social_packs.log",
                    help="Log file path")
    ap.add_argument("--capture_stdio", action="store_true",
                    help="Redirect stdout/stderr into the log")
    ap.add_argument("--dry_run", action="store_true",
                    help="List clips but do not process/save")
    return ap.parse_args()

def main():
    args = parse_args()
    logger = setup_logging(args.log_file, capture_stdio=args.capture_stdio)
    logger.info("Starting Social Token Pack build")
    logger.info(f"Args: {args}")

    clips = find_clips(args.root_split_dir)
    if args.limit and len(clips) > args.limit:
        clips = clips[:args.limit]

    if not clips:
        logger.error(f"No valid clips found under {args.root_split_dir}")
        sys.exit(1)

    logger.info(f"Found {len(clips)} clips to process")

    if args.dry_run:
        for clip_id, frames_dir, trans_json in clips:
            logger.info(f"[DRY RUN] {clip_id} | frames={frames_dir} | transcript={trans_json}")
        logger.info("Dry run complete")
        return

    # Load DINO once
    logger.info("Loading DINO backbone + transforms…")
    model_backbone, preprocess, cls_embed = load_dino(args.dino_ckpt, args.model_name)
    logger.info("Model loaded")

    # Process sequentially (simple & robust for spaCy)
    processed, failed = 0, 0
    for clip_id, frames_dir, trans_json in clips:
        try:
            logger.info(f"Processing {clip_id}")
            pack = build_social_token_pack(
                preprocess=preprocess,
                cls_embed=cls_embed,
                clip_id=clip_id,
                frames_dir=frames_dir,
                transcript_json_path=trans_json,
                clip_fps=args.fps,
                max_global_frames=args.max_global_frames,
                pad=args.pad_sec,
                top_k_locals=args.top_k_locals,
            )
            key = save_pack(pack, args.out_dir)
            logger.info(
                f"{clip_id}: saved key={key} | global={pack['global'].shape} | locals={len(pack['locals'])}"
            )
            processed += 1
        except Exception as e:
            logger.exception(f"ERROR processing {clip_id}: {e}")
            failed += 1

    logger.info(f"Done. processed={processed}, failed={failed}, out_dir={args.out_dir}")

if __name__ == "__main__":
    main()

