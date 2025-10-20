#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSA for Gemma with Social-Token Injection & Pooling — logged, pad-safe, debug & semantic decode
===============================================================================================

Key features
- RIGHT padding + padding="longest".
- Pooling masks intersect with attention mask for SOC-only modes.
- Rich logging, timers, CUDA mem snapshots, run meta sidecar.
- Debug tools to verify injection (ΔL2, max|emb - proj|=0) and *semantic decode* of projected vectors.
- Optional SRP toggle (--no-srp). No 4-D mask logic included.
"""
import argparse
import os, re, json, pickle, sys, time, socket, warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, __version__ as transformers_ver
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from tqdm import tqdm

# deepjuice
from deepjuice.extraction import get_feature_maps
from deepjuice.reduction import get_feature_map_srps

# ---------- Logging -----------
import logging
LOGGER_NAME = "sim_rsa"
logger = logging.getLogger(LOGGER_NAME)

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(hostname)s | pid=%(process)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class HostnameFilter(logging.Filter):
        def filter(self, record):
            record.hostname = socket.gethostname()
            return True

    logger.handlers[:] = []
    logger.addFilter(HostnameFilter())

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

def _log(msg: str, level="info"):
    getattr(logger, level)(msg)

def _truncate(s: str, n=160):
    return (s if len(s) <= n else s[:n] + " …")

class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self):
        self.t0 = time.time()
        _log(f"[TIMER] {self.name}…")
        return self
    def __exit__(self, *_):
        _log(f"[TIMER] {self.name} took {time.time()-self.t0:.2f}s")

def cuda_mem(prefix=""):
    if not torch.cuda.is_available(): return {}
    dev = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    stats = {"device": dev, "alloc_MB": round(alloc/1e6,2), "reserved_MB": round(reserved/1e6,2)}
    _log(f"[CUDA] {prefix} device={dev} alloc={stats['alloc_MB']}MB reserved={stats['reserved_MB']}MB")
    return stats

# ========= Defaults (overridden by CLI) =========
VERBOSE = True
LOG_EXAMPLES = 3
LOG_KEEP_UIDS = 25
DRYRUN = False
LOG_INTERVAL = 50
USE_SRP = True

INJECT_MODE = "full"        # full | global | local | none
POOL_MODE   = "all"         # all | exclude_soc | only_soc | locals_only | eos

BATCH = 8
WORKERS = 8

SOC_G = "<SOC_G>"
SOC_L = "<SOC_L>"

TOKENIZER_DIR = None
LM_PATH_OR_ID = "google/gemma-2-2b"

CHECKPOINT = "/path/to/checkpoints/projector_only.pt"
OOO_INDEX  = "/mnt/data/ooo_index_ordered_for_rsa.csv"
SIM_RSM    = "/path/to/sim_judge_train_rsm.csv"
SAVE_FEATURES_DIR = "/tmp/gemma_soc_feats"
RESULTS_CSV       = "/tmp/gemma_soc_ooo_rsa.csv"

MAX_LOCALS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# Load tokenizer/model
# ----------------------
def load_tokenizer_and_lm(tokenizer_dir: Optional[str], lm_src: str):
    src_for_tok = tokenizer_dir if tokenizer_dir else lm_src
    _log(f"[LOAD] tokenizer_src={src_for_tok}")
    _log(f"[LOAD] lm_src={lm_src}")
    tok = AutoTokenizer.from_pretrained(src_for_tok, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    # Force right padding to avoid position-id drift
    tok.padding_side = "right"
    _log(f"[TOKENIZER] pad_token_id={tok.pad_token_id} padding_side={tok.padding_side}")

    lm = AutoModelForCausalLM.from_pretrained(lm_src, torch_dtype=torch.float32)
    lm.resize_token_embeddings(len(tok), mean_resizing=True)
    lm.config.pad_token_id = tok.pad_token_id
    lm.config.output_hidden_states = False
    lm.config.use_cache = False
    lm = lm.to(DEVICE)
    lm.get_input_embeddings().to(DEVICE)
    _log(f"[LOAD] model device={next(lm.parameters()).device}; vocab={len(tok)}")
    return tok, lm

def _strip_pref(k: str) -> str:
    for pref in ("module.", "projector.", "proj.", "model.projector."):
        if k.startswith(pref): return k[len(pref):]
    return k

def load_projector_from_ckpt(projector_path: str, lm_hidden: int):
    _log(f"[LOAD] projector={projector_path}")
    raw = torch.load(projector_path, map_location="cpu")
    state = raw["projector"] if isinstance(raw, dict) and "projector" in raw else raw
    state = {_strip_pref(k): v for k,v in state.items()}
    groups = {}
    for k, v in state.items():
        if k.endswith(".weight") or k.endswith(".bias"):
            prefix = k.rsplit(".", 1)[0]
            groups.setdefault(prefix, {})[k.split(".")[-1]] = v
    layers = []; prev_lin = False
    def nkey(s):
        m = re.search(r"(\d+)", s)
        return (int(m.group(1)) if m else 0, s)
    for p in sorted(groups.keys(), key=nkey):
        W = groups[p].get("weight"); B = groups[p].get("bias")
        if W.ndim == 2:
            lin = nn.Linear(W.shape[1], W.shape[0], bias=B is not None)
            with torch.no_grad():
                lin.weight.copy_(W)
                if B is not None: lin.bias.copy_(B)
            if prev_lin: layers.append(nn.GELU())
            layers.append(lin); prev_lin = True
        elif W.ndim == 1:
            ln = nn.LayerNorm(W.shape[0], elementwise_affine=True)
            with torch.no_grad():
                ln.weight.copy_(W)
                if B is not None: ln.bias.copy_(B)
            layers.append(ln); prev_lin = False
    proj = nn.Sequential(*layers).to(device=DEVICE, dtype=torch.float32).eval()
    with torch.no_grad():
        y = proj(torch.zeros(2, 768, dtype=torch.float32, device=DEVICE))
        assert y.shape[-1] == lm_hidden, f"Projector out {y.shape[-1]} != LM hidden {lm_hidden}"
    _log(f"[LOAD] projector ok; output_dim={y.shape[-1]}")
    return proj

def _positions_from_npz(npz) -> Optional[List[int]]:
    for key in ["indices","word_idx","word_indices","positions","pos"]:
        if key in npz.files:
            return np.array(npz[key]).astype(int).reshape(-1).tolist()
    return None

def _words_from_meta(meta: dict) -> List[str]:
    if "tokens" in meta and isinstance(meta["tokens"], list) and meta["tokens"]:
        return [str(t).strip() for t in meta["tokens"] if str(t).strip()]
    words = []
    for w in meta.get("words", []):
        s = w.get("text") or w.get("word")
        if s: words.append(str(s).strip())
    return words

def _positions_from_meta(meta: dict) -> Optional[List[int]]:
    for key in ["local_indices","locals_idx","locals_index","local_positions","soc_local_indices","word_idx","word_indices"]:
        if key in meta and isinstance(meta[key], (list, tuple)):
            return [int(i) for i in meta[key]]
    return None

def build_inline(words: List[str], local_idx: List[int], fallback_caption: Optional[str]) -> str:
    if (not words) and fallback_caption:
        words = fallback_caption.split()
    if not words:
        return "(no transcript)" if INJECT_MODE == "none" else f"{SOC_G} (no transcript)"
    if INJECT_MODE == "none":
        return " ".join(words)
    if INJECT_MODE == "global":
        return f"{SOC_G} " + " ".join(words)
    if INJECT_MODE == "local":
        # local only: inject SOC_L after words, but no SOC_G
        loc = set(i for i in local_idx if 0 <= i < len(words))
        out = []
        for i, w in enumerate(words):
            out.append(w)
            if i in loc: out.append(SOC_L)
        return " ".join(out)
    # full injection
    loc = set(i for i in local_idx if 0 <= i < len(words))
    out = [SOC_G]
    for i, w in enumerate(words):
        out.append(w)
        if i in loc: out.append(SOC_L)
    return " ".join(out)

def load_pack_row(row: Dict[str, Any], max_locals: Optional[int] = 50):
    g = np.load(row["global_path"]).astype("float32").reshape(-1)
    Lz = np.load(row["locals_path"], allow_pickle=True)
    vec_keys = [k for k in Lz.files if k not in {"indices","word_idx","word_indices","positions","pos"}]
    locals_vecs = [Lz[k].astype("float32").reshape(-1) for k in vec_keys]
    with open(row["meta_path"], "rb") as f:
        meta = pickle.load(f)

    words = _words_from_meta(meta)
    pos_meta = _positions_from_meta(meta)
    pos_npz  = _positions_from_npz(Lz)
    if pos_npz: pos = pos_npz[:len(locals_vecs)]
    elif pos_meta: pos = pos_meta[:len(locals_vecs)]
    else: pos = list(range(len(locals_vecs)))
    pairs = sorted(list(zip(pos, locals_vecs)), key=lambda x: x[0])
    if max_locals and len(pairs) > max_locals:
        pairs = pairs[:max_locals]
    inline = build_inline(words, [p for p,_ in pairs], row.get("caption", ""))
    L_list = [v for _, v in pairs]
    return inline, g, L_list

# ----------------------
# Debug helpers
# ----------------------
def preview_batch_tokens(tok, batch, socg_id, soc_l_id, n=3):
    """Print a few decoded examples, plus positions of <SOC_G>/<SOC_L> and local counts."""
    ids = batch["input_ids"]
    attn = batch["attention_mask"]
    lmask = batch["local_mask"]
    texts = tok.batch_decode(ids, skip_special_tokens=False)  # keep <SOC_*> visible

    B = ids.size(0)
    for b in range(min(n, B)):
        id_list = ids[b].tolist()
        toks = tok.convert_ids_to_tokens(id_list)
        g_pos = [i for i, t in enumerate(id_list) if t == socg_id]
        l_pos = [i for i, t in enumerate(id_list) if (soc_l_id is not None and t == soc_l_id)]
        num_locals = int(lmask[b].sum().item())

        _log(f"[DBG TOKENS b{b}] decoded='{_truncate(texts[b])}'")
        _log(f"[DBG TOKENS b{b}] SOC_G positions={g_pos}  SOC_L positions={l_pos}  locals_in_batch={num_locals}")
        if l_pos:
            lo = max(0, l_pos[0] - 8); hi = l_pos[0] + 9
            _log(f"[DBG TOKENS b{b}] context around first <SOC_L>: {toks[lo:hi]}")
        if g_pos:
            lo = max(0, g_pos[0] - 5); hi = g_pos[0] + 6
            _log(f"[DBG TOKENS b{b}] context around <SOC_G>: {toks[lo:hi]}")
        _log(f"[DBG TOKENS b{b}] last_attended_idx={int(attn[b].sum().item())-1}")

# --------- Semantic decode helpers (nearest-token by cosine) ----------
@torch.no_grad()
def _topk_decode_one(vec: torch.Tensor,
                     emb_weight: torch.Tensor,
                     k: int = 5,
                     exclude_ids: Optional[List[int]] = None,
                     chunk_size: int = 50000) -> Tuple[List[int], List[float]]:
    """
    vec: [H] projected embedding
    emb_weight: [V, H] input-embedding matrix
    returns: (token_ids[topk], scores[topk]) with cosine similarity
    """
    device = emb_weight.device
    v = vec.to(device, dtype=torch.float32)
    v = v / (v.norm() + 1e-9)

    V = emb_weight.size(0)
    top_scores = torch.full((k,), -1e9, device=device)
    top_indices = torch.full((k,), -1, device=device, dtype=torch.long)

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        chunk = emb_weight[start:end].to(device, dtype=torch.float32)
        chunk = chunk / (chunk.norm(dim=1, keepdim=True) + 1e-9)
        sims = torch.mv(chunk, v)  # [chunk]
        if exclude_ids:
            for ex in exclude_ids:
                if start <= ex < end:
                    sims[ex - start] = -1e9
        s, idx = torch.topk(sims, k=min(k, end - start))
        # merge with global top-k
        all_s = torch.cat([top_scores, s])
        all_i = torch.cat([top_indices, idx + start])
        s2, ord2 = torch.topk(all_s, k=k)
        i2 = all_i[ord2]
        top_scores, top_indices = s2, i2

    return top_indices.tolist(), top_scores.tolist()

def _annotate_tokens_with_decodes(tok_ids: List[int],
                                  tok,
                                  g_positions: List[int],
                                  l_positions: List[int],
                                  g_top1: Optional[str],
                                  l_top1_list: List[str],
                                  l_k: int,
                                  max_len: int = 120) -> str:
    tokens = tok.convert_ids_to_tokens(tok_ids)
    if g_positions and g_top1:
        tokens[g_positions[0]] = f"<SOC_G→{g_top1}>"
    for i in range(min(l_k, len(l_positions))):
        tokens[l_positions[i]] = f"<SOC_L→{l_top1_list[i]}>"
    s = " ".join(tokens)
    return _truncate(s, max_len)

# ----------------------
# Dataset / Collate
# ----------------------
class OOODataset(Dataset):
    def __init__(self, index_csv, max_locals: int = 50):
        df = pd.read_csv(index_csv)
        need = {"clip_id","global_path","locals_path","meta_path","caption"}
        missing = sorted(list(need - set(df.columns)))
        if missing:
            raise ValueError(f"Missing required columns in {index_csv}: {missing}")
        self.items = []
        for i, row in df.reset_index(drop=True).iterrows():
            inline, g, L = load_pack_row(row, max_locals=max_locals)
            self.items.append({
                "uid": str(row.get("clip_id", i)),
                "uid_code": i,
                "text": inline,
                "g": g,
                "L": L,
            })
        _log(f"[DATASET] rows={len(self.items)} from {index_csv} (inject={INJECT_MODE}, pool={POOL_MODE})")
        for j in range(min(LOG_EXAMPLES, len(self.items))):
            it = self.items[j]
            gl = it["text"].count(SOC_G); ll = it["text"].count(SOC_L)
            _log(f"[DATASET ex {j}] uid={it['uid']} G={gl} L={ll} text='{_truncate(it['text'])}'")

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def make_collate(tok, soc_g_id: int, soc_l_id: Optional[int]):
    def collate_cpu(batch):
        texts       = [b["text"] for b in batch]
        uid_codes   = [int(b["uid_code"]) for b in batch]
        locals_lists= [b["L"] for b in batch]
        globals_np  = [b["g"] for b in batch]

        # padding="longest" (tokenizer padding_side already set to "right")
        enc = tok(texts, return_tensors="pt", padding="longest", truncation=False)
        input_ids = enc["input_ids"]; attn = enc["attention_mask"]
        g = torch.from_numpy(np.stack(globals_np, 0)).to(torch.float32)

        maxL = max((len(x) for x in locals_lists), default=0)
        if maxL == 0:
            l_pad = torch.zeros((len(locals_lists), 1, 768), dtype=torch.float32)
            l_mask= torch.zeros((len(locals_lists), 1), dtype=torch.bool)
        else:
            l_pad = torch.zeros((len(locals_lists), maxL, 768), dtype=torch.float32)
            l_mask= torch.zeros((len(locals_lists), maxL), dtype=torch.bool)
            for i, arrs in enumerate(locals_lists):
                if len(arrs) == 0: continue
                vv = np.stack(arrs, 0)
                l_pad[i, :vv.shape[0], :] = torch.from_numpy(vv)
                l_mask[i, :vv.shape[0]] = True

        if VERBOSE:
            B = input_ids.size(0)
            _log(f"[COLLATE] input_ids={tuple(input_ids.shape)} attn={tuple(attn.shape)} "
                 f"g={tuple(g.shape)} l_pad={tuple(l_pad.shape)} l_mask.sum={int(l_mask.sum())}")
            for b in range(min(LOG_EXAMPLES, B)):
                ids = input_ids[b].tolist()
                g_pos = [i for i,tokid in enumerate(ids) if tokid == soc_g_id]
                l_pos = [i for i,tokid in enumerate(ids) if (soc_l_id is not None and tokid == soc_l_id)]
                l_count = int(l_mask[b].sum().item())
                _log(f"[COLLATE ex {b}] uid_code={uid_codes[b]} tokens={len(ids)} "
                     f"Gpos={g_pos} Lpos={l_pos} Lvecs={l_count} text='{_truncate(texts[b])}'")
                if soc_l_id is not None and len(l_pos) != l_count and INJECT_MODE != "none":
                    _log(f"[WARN] L positions ({len(l_pos)}) != local vectors ({l_count}) → min-align downstream.", level="warning")

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "global_vec": g,
            "local_vecs_padded": l_pad,
            "local_mask": l_mask,
            "uid_code": torch.tensor(uid_codes, dtype=torch.long),
        }
    return collate_cpu

# ----------------------
# Pooling
# ----------------------
def pool_to_2d(h, ids, attn, pool_mode, socg_id, soc_l_id):
    if h.dim() != 3:
        return h
    B, T, H = h.shape
    m_all = attn.bool()

    if pool_mode == "all":
        m = m_all
    elif pool_mode == "exclude_soc":
        soc = (ids == socg_id)
        if soc_l_id is not None: soc = soc | (ids == soc_l_id)
        m = m_all & (~soc)
        empty = m.sum(dim=1) == 0
        if empty.any(): m[empty] = m_all[empty]
    elif pool_mode == "only_soc":
        m = (ids == socg_id)
        if soc_l_id is not None: m = m | (ids == soc_l_id)
        m = m & m_all
        empty = m.sum(dim=1) == 0
        if empty.any(): m[empty] = m_all[empty]
    elif pool_mode == "locals_only":
        if soc_l_id is None:
            m = m_all
        else:
            m = (ids == soc_l_id) & m_all
            empty = m.sum(dim=1) == 0
            if empty.any(): m[empty] = m_all[empty]
    elif pool_mode == "eos":
        lengths = attn.sum(dim=1).clamp_min(1).long()
        idx = (lengths - 1).to(h.device)
        return h[torch.arange(B, device=h.device), idx, :]
    else:
        m = m_all

    m = m.to(h.dtype)
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (h * m.unsqueeze(-1)).sum(dim=1) / denom

# ----------------------
# SOC wrapper (with debug + semantic decode)
# ----------------------
class SOCWrapper(nn.Module):
    def __init__(self, lm, tok, projector, soc_g_id: int, soc_l_id: Optional[int],
                 debug_injection: bool=False,
                 decode_topk: int=0,
                 decode_show: int=2,
                 decode_exclude_soc: bool=True):
        super().__init__()
        self.lm  = lm
        self.tok = tok
        self.proj= projector
        self.soc_g = soc_g_id
        self.soc_l = soc_l_id
        self.debug_injection = debug_injection
        self.decode_topk = int(decode_topk)
        self.decode_show = int(decode_show)
        self.decode_exclude_soc = decode_exclude_soc
        self._last_meta = None

    def forward(self, *args, **batch):
        if len(args) == 1 and isinstance(args[0], dict):
            batch = args[0]

        dev = next(self.lm.parameters()).device
        emb_dtype = self.lm.get_input_embeddings().weight.dtype
        emb_weight = self.lm.get_input_embeddings().weight  # [V,H]

        input_ids = batch["input_ids"].to(dev)
        attn      = batch["attention_mask"].to(dev)
        g         = batch["global_vec"].to(dev, dtype=torch.float32)
        l_pad     = batch["local_vecs_padded"].to(dev, dtype=torch.float32)
        l_mask    = batch["local_mask"].to(dev)

        uid_code = batch.get("uid_code", None)
        self._last_meta = {"uid_code": uid_code.detach().cpu()} if uid_code is not None else None

        # original -> modified embeddings
        orig_embs = self.lm.get_input_embeddings()(input_ids)
        embs = orig_embs.clone()

        # GLOBAL injection
        if INJECT_MODE in ("full", "global"):
            is_g = (input_ids == self.soc_g)
            if is_g.any():
                proj_g = self.proj(g).to(emb_dtype)
                rows_idx = is_g.any(dim=1).nonzero(as_tuple=True)[0]
                repeats  = is_g.sum(dim=1)[rows_idx]
                embs[is_g] = proj_g[rows_idx].repeat_interleave(repeats, dim=0)

        # LOCAL injection
        if INJECT_MODE in ("full", "local") and self.soc_l is not None:
            B = input_ids.size(0)
            for b in range(B):
                pos = (input_ids[b] == self.soc_l).nonzero(as_tuple=True)[0]
                if pos.numel() == 0: continue
                valid_L = l_mask[b].nonzero(as_tuple=True)[0]
                k = min(pos.numel(), valid_L.numel())
                if k > 0:
                    proj_l = self.proj(l_pad[b, valid_L[:k], :].to(dev)).to(emb_dtype)
                    embs[b, pos[:k], :] = proj_l

        # ---------- DEBUG: confirm actual replacement + show semantics ----------
        if self.debug_injection:
            with torch.no_grad():
                B = input_ids.size(0)
                for b in range(min(LOG_EXAMPLES, B)):
                    # ΔL2 @ <SOC_G>
                    gpos = (input_ids[b] == self.soc_g).nonzero(as_tuple=True)[0]
                    if gpos.numel() > 0 and INJECT_MODE in ("full","global"):
                        delta_g = (embs[b, gpos, :] - orig_embs[b, gpos, :]).norm(dim=1)
                        _log(f"[INJECT-DBG b{b}] <SOC_G> positions={gpos.tolist()} ΔL2={delta_g.tolist()}")

                    # ΔL2 @ <SOC_L> and vector diagnostics
                    if INJECT_MODE in ("full", "local") and self.soc_l is not None:
                        lpos = (input_ids[b] == self.soc_l).nonzero(as_tuple=True)[0]
                        if lpos.numel() > 0:
                            delta_l = (embs[b, lpos, :] - orig_embs[b, lpos, :]).norm(dim=1)
                            _log(f"[INJECT-DBG b{b}] <SOC_L> positions={lpos.tolist()} ΔL2={delta_l.tolist()}")
                            valid_L = l_mask[b].nonzero(as_tuple=True)[0]
                            k = min(lpos.numel(), valid_L.numel())
                            if k > 0:
                                pl = self.proj(l_pad[b, valid_L[:k], :].to(dev)).to(emb_dtype)
                                max_abs = (embs[b, lpos[:k], :] - pl).abs().max().item()
                                _log(f"[INJECT-DBG b{b}] max|emb - proj(local)| over first {k} = {max_abs:.3e}")

                                # --- Added diagnostics: norms, pairwise, peek dims ---
                                pl_cpu = pl.detach().float().cpu()
                                _log(f"[INJECT-DBG b{b}] locals proj norms={pl_cpu.norm(dim=1).tolist()}")
                                if pl_cpu.size(0) > 1:
                                    d = torch.cdist(pl_cpu, pl_cpu, p=2)
                                    d_flat = d[~torch.eye(d.size(0), dtype=torch.bool)]
                                    _log(f"[INJECT-DBG b{b}] locals proj pairwise L2: "
                                         f"min={d_flat.min().item():.4e}, mean={d_flat.mean().item():.4e}, max={d_flat.max().item():.4e}")
                                peek_n = min(2, pl_cpu.size(0))
                                _log(f"[INJECT-DBG b{b}] locals proj first {peek_n} rows (first 8 dims): "
                                     f"{pl_cpu[:peek_n, :8].tolist()}")

                                # --- Optional semantic decode of projected vectors ---
                                if self.decode_topk > 0:
                                    exclude_ids = []
                                    if self.decode_exclude_soc:
                                        if self.soc_g is not None: exclude_ids.append(int(self.soc_g))
                                        if self.soc_l is not None: exclude_ids.append(int(self.soc_l))

                                    # decode global (single)
                                    g_top1_txt = None
                                    if gpos.numel() > 0 and INJECT_MODE in ("full","global"):
                                        g_proj_b = self.proj(g[b].to(dev)).to(emb_dtype)
                                        ids_top, sc_top = _topk_decode_one(g_proj_b, emb_weight,
                                                                           k=self.decode_topk,
                                                                           exclude_ids=exclude_ids)
                                        toks = self.tok.convert_ids_to_tokens(ids_top)
                                        _log(f"[DECODE b{b}] <SOC_G> top{self.decode_topk}={list(zip(toks, [round(s,4) for s in sc_top]))}")
                                        g_top1_txt = toks[0] if toks else None

                                    # decode locals (first decode_show)
                                    l_top1_txts = []
                                    show_m = min(self.decode_show, k)
                                    for i in range(show_m):
                                        ids_top, sc_top = _topk_decode_one(pl[i], emb_weight,
                                                                           k=self.decode_topk,
                                                                           exclude_ids=exclude_ids)
                                        toks = self.tok.convert_ids_to_tokens(ids_top)
                                        _log(f"[DECODE b{b}] <SOC_L[{i}]@pos{int(lpos[i].item())}> top{self.decode_topk}="
                                             f"{list(zip(toks, [round(s,4) for s in sc_top]))}")
                                        l_top1_txts.append(toks[0] if toks else None)

                                    # Annotated caption tokens (limited)
                                    tok_ids = input_ids[b].tolist()
                                    ann = _annotate_tokens_with_decodes(
                                        tok_ids, self.tok,
                                        g_positions=gpos.tolist(),
                                        l_positions=lpos.tolist(),
                                        g_top1=g_top1_txt,
                                        l_top1_list=l_top1_txts,
                                        l_k=show_m,
                                        max_len=200,
                                    )
                                    _log(f"[DECODE b{b}] annotated tokens: {ann}")

        # -------------------------------------------------------
        return self.lm(inputs_embeds=embs, attention_mask=attn, return_dict=True)

    def pop_last_meta(self):
        m = self._last_meta
        self._last_meta = None
        return m

# ----------------------
# CLI
# ----------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    io = p.add_argument_group("Core I/O")
    io.add_argument("--lm", default=LM_PATH_OR_ID,
                    help="HF model id or local path to weights (e.g., google/gemma-2-2b).")
    io.add_argument("--tokenizer-dir", default=None,
                    help="Optional tokenizer directory. If omitted, uses --lm.")
    io.add_argument("--projector-ckpt", required=True,
                    help="Path to projector checkpoint (.pt).")
    io.add_argument("--index", dest="ooo_index", required=True,
                    help="CSV index with columns: clip_id, global_path, locals_path, meta_path, caption.")
    io.add_argument("--sim-rsm", required=True,
                    help="CSV square matrix [N,N] of human similarities.")
    io.add_argument("--save-feats", default=SAVE_FEATURES_DIR,
                    help="Directory to save per-layer features (.npz).")
    io.add_argument("--results-csv", default=RESULTS_CSV,
                    help="Path to save RSA results CSV.")

    beh = p.add_argument_group("Behavior toggles")
    beh.add_argument("--inject", choices=["full","global","local","none"], default=INJECT_MODE,
                     help="Injection mode for social tokens.")
    beh.add_argument("--pool", choices=["all","exclude_soc","only_soc","locals_only","eos"], default=POOL_MODE,
                     help="Pooling mode before RSA.")
    beh.add_argument("--soc-g-token", default=SOC_G, help="String token for global SOC.")
    beh.add_argument("--soc-l-token", default=SOC_L, help="String token for local SOC.")
    beh.add_argument("--max-locals", type=int, default=MAX_LOCALS,
                     help="Max local vectors to inject per item (truncates if longer).")

    rt = p.add_argument_group("Runtime & Logging")
    rt.add_argument("--batch", type=int, default=BATCH, help="Batch size.")
    rt.add_argument("--workers", type=int, default=WORKERS, help="DataLoader workers (0 = main process).")
    rt.add_argument("--device", choices=["auto","cuda","cpu"], default="auto",
                    help="Computation device.")
    rt.add_argument("--dryrun", action="store_true", help="Process one batch then exit.")
    rt.add_argument("--quiet", action="store_true", help="Reduce console logging (INFO->WARNING).")
    rt.add_argument("--log-level", default="INFO", help="Python logging level (DEBUG, INFO, WARNING, ERROR).")
    rt.add_argument("--log-file", default=None, help="Optional path to write a file log.")
    rt.add_argument("--log-examples", type=int, default=LOG_EXAMPLES, help="How many examples to print.")
    rt.add_argument("--log-keep-uids", type=int, default=LOG_KEEP_UIDS,
                    help="How many layer UIDs to print from warmup.")
    rt.add_argument("--log-interval", type=int, default=LOG_INTERVAL,
                    help="Batches between progress logs.")
    rt.add_argument("--no-srp", action="store_true",
                    help="Disable SRP projection; use pooled features directly.")
    # Debug + decode
    rt.add_argument("--debug-injection", action="store_true",
                    help="Log token decode, SOC positions, embedding deltas, and (optional) semantic decode.")
    rt.add_argument("--debug-n", type=int, default=3,
                    help="How many examples to preview when --debug-injection is on.")
    rt.add_argument("--decode-proj-topk", type=int, default=0,
                    help="If >0, perform nearest-token (cosine) decode of projected vectors; prints top-k.")
    rt.add_argument("--decode-proj-show", type=int, default=2,
                    help="Max local vectors per example to decode (limits log spam).")
    rt.add_argument("--decode-annotated", action="store_true",
                    help="Also print an annotated token stream with <SOC_*> replaced by nearest decoded token.")
    rt.add_argument("--decode-exclude-soc", action="store_true",
                    help="Exclude <SOC_G>/<SOC_L> token IDs from nearest-neighbor decode (recommended).")

    return p.parse_args()

def banner(args):
    meta = {
        "when": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers_ver,
        "cuda_available": torch.cuda.is_available(),
        "device": DEVICE,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_ARRAY_TASK_ID": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "args": vars(args),
    }
    _log("[STARTUP] " + json.dumps(meta, indent=2))

def main():
    warnings.filterwarnings("once")
    args = parse_args()

    # Setup logging
    lvl = ("WARNING" if args.quiet else args.log_level)
    setup_logging(level=lvl, log_file=args.log_file)
    global VERBOSE, LOG_EXAMPLES, LOG_KEEP_UIDS, DRYRUN, LOG_INTERVAL, USE_SRP
    global INJECT_MODE, POOL_MODE, BATCH, WORKERS, DEVICE
    global SOC_G, SOC_L, TOKENIZER_DIR, LM_PATH_OR_ID
    global CHECKPOINT, OOO_INDEX, SIM_RSM, SAVE_FEATURES_DIR, RESULTS_CSV, MAX_LOCALS

    VERBOSE = not args.quiet
    LOG_EXAMPLES = args.debug_n if args.debug_injection else args.log_examples
    LOG_KEEP_UIDS = args.log_keep_uids
    LOG_INTERVAL = args.log_interval
    DRYRUN = args.dryrun
    USE_SRP = not args.no_srp

    INJECT_MODE = args.inject.lower()
    POOL_MODE   = args.pool.lower()

    SOC_G = args.soc_g_token
    SOC_L = args.soc_l_token

    BATCH   = int(args.batch)
    WORKERS = int(args.workers)

    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device

    TOKENIZER_DIR = args.tokenizer_dir
    LM_PATH_OR_ID = args.lm

    CHECKPOINT = args.projector_ckpt
    OOO_INDEX  = args.ooo_index
    SIM_RSM    = args.sim_rsm
    SAVE_FEATURES_DIR = args.save_feats
    RESULTS_CSV       = args.results_csv
    MAX_LOCALS        = int(args.max_locals)

    os.makedirs(SAVE_FEATURES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_CSV) or ".", exist_ok=True)

    banner(args)
    cuda_mem("startup")

    try:
        with Timer("load model+tokenizer"):
            tok, lm = load_tokenizer_and_lm(TOKENIZER_DIR, LM_PATH_OR_ID)
            lm.eval()
            cuda_mem("after load model")

        socg_id = tok.convert_tokens_to_ids(SOC_G)
        soc_l_id = tok.convert_tokens_to_ids(SOC_L) if SOC_L in tok.get_vocab() else None

        if INJECT_MODE in ("full", "global"):
            if socg_id is None or socg_id == tok.unk_token_id:
                raise RuntimeError(f"{SOC_G} token not present in tokenizer vocab (source={TOKENIZER_DIR or LM_PATH_OR_ID}).")

        if INJECT_MODE in ("full", "local"):
            if soc_l_id is None or soc_l_id == tok.unk_token_id:
                raise RuntimeError(f"{SOC_L} token not present in tokenizer vocab (source={TOKENIZER_DIR or LM_PATH_OR_ID}).")

        with Timer("load projector"):
            proj = load_projector_from_ckpt(CHECKPOINT, lm.config.hidden_size)
            cuda_mem("after projector")

        wrapped = SOCWrapper(
            lm, tok, proj, socg_id, soc_l_id,
            debug_injection=args.debug_injection,
            decode_topk=args.decode_proj_topk if args.decode_annotated or args.decode_proj_topk > 0 else 0,
            decode_show=args.decode_proj_show,
            decode_exclude_soc=args.decode_exclude_soc
        ).eval()

        with Timer("dataset init"):
            ds = OOODataset(OOO_INDEX, max_locals=MAX_LOCALS)
        collate_cpu = make_collate(tok, socg_id, soc_l_id)

        dl_kwargs = dict(
            dataset=ds,
            batch_size=BATCH,
            shuffle=False,
            num_workers=WORKERS,
            pin_memory=True,
            collate_fn=collate_cpu
        )
        if WORKERS > 0:
            dl_kwargs.update(prefetch_factor=2, persistent_workers=True)
        dl = DataLoader(**dl_kwargs)
        _log(f"[DATALOADER] workers={WORKERS} batch={BATCH} persistent={WORKERS>0}")

        # Warm batch + optional token preview
        warm_batch = next(iter(dl))
        if args.debug_injection:
            preview_batch_tokens(tok, warm_batch, socg_id, soc_l_id, n=LOG_EXAMPLES)

        # Warmup (also triggers debug in wrapper.forward)
        with Timer("warmup feature map discovery"):
            with torch.no_grad():
                warm_maps = get_feature_maps(
                    wrapped, warm_batch,
                    flatten=False, save_inputs=False,
                    remove_duplicates=True, report_irregularities=False
                )
        keep_uids = sorted(warm_maps.keys())
        _log(f"[WARMUP] discovered {len(keep_uids)} module UIDs; showing first {min(LOG_KEEP_UIDS, len(keep_uids))}:")
        for u in keep_uids[:LOG_KEEP_UIDS]:
            fm = warm_maps[u]
            shape = tuple(fm.shape) if hasattr(fm, "shape") else "NA"
            _log(f"  - {u} shape={shape}")

        # Extract
        uid_code_order: List[int] = []
        features_accum: Dict[str, List[np.ndarray]] = {u: [] for u in keep_uids}

        _log("[EXTRACT] starting per-batch deepjuice extraction (var-length safe)…")
        t_last = time.time()
        for bi, batch in enumerate(tqdm(dl, desc="Extracting batches", total=len(dl))):
            with torch.no_grad():
                fmap_dict = get_feature_maps(
                    wrapped, batch,
                    flatten=False, save_inputs=False,
                    remove_duplicates=True, report_irregularities=False
                )
            meta = wrapped.pop_last_meta()
            if meta is not None:
                uid_code_order.extend(meta["uid_code"].tolist())

            ids  = batch["input_ids"]
            attn = batch["attention_mask"].to(dtype=torch.float32)

            for uid, fmap in fmap_dict.items():
                if uid not in features_accum: continue
                t = fmap  # [B,T,H] or [B,D]...
                if t.dim() == 3:
                    t2d = pool_to_2d(t, ids.to(t.device), attn.to(t.device), POOL_MODE, socg_id, soc_l_id)  # [B,H]
                elif t.dim() > 3:
                    t2d = t.flatten(start_dim=1)
                else:
                    t2d = t

                if USE_SRP:
                    X = get_feature_map_srps(t2d, device=DEVICE)
                else:
                    X = t2d

                X = X.detach().float().cpu().numpy()
                features_accum[uid].append(X)

            if (bi % LOG_INTERVAL) == 0:
                dt = time.time()-t_last; t_last = time.time()
                _log(f"[EXTRACT] batch={bi}/{len(dl)} dt={dt:.2f}s "
                     f"uids_collected={(sum(len(v) for v in features_accum.values()))} "
                     f"N_uid_codes={len(uid_code_order)}")
                cuda_mem(f"after batch {bi}")

            if DRYRUN:
                _log("[DRYRUN] stopping after first batch by request (--dryrun).")
                break

        # Concatenate & order
        with Timer("concatenate features"):
            features_by_uid = {uid: np.concatenate(chunks, axis=0)
                               for uid, chunks in features_accum.items() if len(chunks) > 0}
        uid_order = [ds.items[i]["uid"] for i in uid_code_order]
        _log(f"[EXTRACT] collected features for {len(uid_order)} items across {len(features_by_uid)} UIDs.")

        # RSA
        with Timer("RSA computation"):
            sim_rsm = pd.read_csv(SIM_RSM).values
            if sim_rsm.shape[0] != len(uid_order):
                _log(f"[WARN] RSM size {sim_rsm.shape} != N items {len(uid_order)}; will min-align.", level="warning")
            n = min(sim_rsm.shape[0], len(uid_order))

            results = []
            best_layer_uid = None
            best_rho = -np.inf
            best_features = None

            for layer_uid, X in features_by_uid.items():
                # Use actual number of features for this layer (may be less than n if extraction failed)
                layer_n = min(n, X.shape[0])
                Xn = X[:layer_n]

                # Compute triangle indices for this layer's actual size
                tri = np.triu_indices(layer_n, k=1)
                sim_flat = sim_rsm[:layer_n, :layer_n][tri]

                rsm = 1 - pairwise_distances(Xn, metric='correlation')
                model_flat = rsm[tri]
                rho, p = spearmanr(sim_flat, model_flat)
                results.append({"layer_uid": layer_uid, "spearman": float(rho), "p": float(p)})
                _log(f"[RSA] {layer_uid}: r={rho:.4f}, p={p:.2e} (n={layer_n})")

                # Track best layer for saving
                if rho > best_rho:
                    best_rho = rho
                    best_layer_uid = layer_uid
                    best_features = Xn

        df_res = pd.DataFrame(results)
        if len(df_res):
            best = df_res.loc[df_res['spearman'].idxmax()]
            _log(f"[FINISHED] Best layer: {best['layer_uid']}  r={best['spearman']:.4f}  p={best['p']:.2e}")

            # Save only the best layer's features
            if best_features is not None:
                out_npz = os.path.join(SAVE_FEATURES_DIR, f"{INJECT_MODE}_{POOL_MODE}_{best_layer_uid.replace('/', '_')}_best.npz")
                np.savez(out_npz, X=best_features, uids=np.array(uid_order[:n], dtype=object))
                _log(f"[SAVE] Best layer features saved to: {out_npz}  shape={best_features.shape}")
        else:
            _log("[FINISHED] No results computed.", level="warning")
        df_res.to_csv(RESULTS_CSV, index=False)
        _log(f"[DONE] Saved RSA results to: {RESULTS_CSV}")

        # Save a small meta sidecar
        sidecar = RESULTS_CSV.rsplit(".", 1)[0] + ".meta.json"
        with open(sidecar, "w") as f:
            json.dump({
                "args": vars(args),
                "device": DEVICE,
                "cuda": torch.cuda.is_available(),
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "when": datetime.now().isoformat(timespec="seconds"),
                "n_items": len(uid_order),
                "n_layers": len(df_res),
                "best": df_res.loc[df_res['spearman'].idxmax()].to_dict() if len(df_res) else None
            }, f, indent=2)
        _log(f"[DONE] Wrote run meta: {sidecar}")

    except Exception as e:
        logger.exception("FATAL: unhandled exception")
        sys.exit(1)

if __name__ == "__main__":
    main()

