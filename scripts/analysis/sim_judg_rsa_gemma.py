#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSA for Gemma with Social-Token Injection & Pooling
=======================================================

This script computes representational similarity analysis (RSA) between a
language model's internal representations and a human similarity matrix for the
Odd-One-Out (OOO) social video clips. It optionally *injects* social features
into the model via special tokens and provides several *pooling* strategies to
summarize sequence features before RSA.

Pipeline (end-to-end)
---------------------
1) **Load model & tokenizer**
   - Loads a Gemma Causal LM (default: ``google/gemma-2-2b``) and tokenizer.
   - Ensures a ``pad_token`` exists; disables cache; moves to device.

2) **Load social-feature projector**
   - A small MLP/LayerNorm stack loaded from ``SOC_PROJECTOR_CKPT`` that maps
     social features (default 768-D) to the LM hidden size so they can replace
     token embeddings at special positions.

3) **Build dataset from index CSV (``OOO_INDEX``)**
   - Required columns: ``clip_id, global_path, locals_path, meta_path, caption``.
   - Loads per-clip:
     * **Global vector** ``g``: shape ``[768]`` (clip-level).
     * **Local vectors** ``L``: variable-length list of 768-D vectors aligned to words.
     * **Token positions** for locals from npz/meta (if available).
   - Constructs the *inline* text with or without special tokens depending on
     ``SOC_INJECT`` (see below).

4) **Collate (batching)**
   - Tokenizes texts with padding -> ``input_ids``, ``attention_mask``.
   - Packs ``g`` to a dense tensor ``[B, 768]``.
   - Pads local vectors to ``[B, maxL, 768]`` and builds a boolean mask ``l_mask``.
   - **Captions (strings) are not padded**; only token IDs are padded and masked.

5) **Injection wrapper (``SOCWrapper``)**
   - Looks up base token embeddings.
   - If injection is enabled, *replaces* embeddings at:
     * ``<SOC_G>`` with ``projector(g)`` (global signal).
     * ``<SOC_L>`` with ``projector(L[i])`` for the first *k* local positions (local signals).
   - Forwards through the LM; hooks (via **deepjuice**) capture layer feature maps.

6) **Pooling to fixed-size vectors**
   - Most captured maps are ``[B, T, H]``. They are reduced to ``[B, H]`` by
     ``pool_to_2d`` according to ``SOC_POOL`` (see below). Pooling respects the
     attention mask, so pads are ignored. ``eos`` picks the last *attended* token.

7) **Dimensionality reduction & RSA**
   - Applies sparse random projections (``get_feature_map_srps``) for efficiency.
   - Builds a model RSM using correlation similarity (``1 - correlation distance``).
   - Flattens upper triangles and computes **Spearman rho** vs. the human RSM
     from ``SIM_RSM``. Results (per-layer) are saved to CSV; best layer is printed.
   - Optionally saves per-layer pooled features to ``SAVE_FEATURES_DIR``.

Environment Variables (controls)
--------------------------------
**Core I/O**
- ``SOC_OUTPUT_DIR`` / ``SOC_TOKENIZER_DIR`` : tokenizer directory (if separate).
- ``SOC_LM``      : HF model id or local path (default: ``google/gemma-2-2b``).
- ``SOC_PROJECTOR_CKPT`` : projector checkpoint file.
- ``SOC_OOO_INDEX``      : CSV with dataset index.
- ``SOC_SIM_RSM``        : CSV for human similarity RSM.
- ``SOC_SAVE_FEATS``     : directory to save per-layer features (npz).
- ``SOC_RESULTS_CSV``    : output CSV for RSA results.

**Behavior toggles**
- ``SOC_INJECT`` : ``full`` | ``global`` | ``none``  (default: ``full``)
    * ``none``   : no special tokens; no injection.
    * ``global`` : prepend ``<SOC_G>``; inject only the global vector.
    * ``full``   : ``<SOC_G>`` + ``<SOC_L>`` inserted after selected words; inject both.
- ``SOC_POOL``   : ``all`` | ``exclude_soc`` | ``only_soc`` | ``locals_only`` | ``eos`` (default: ``all``)
    * ``all``          : mean over all attended tokens (includes SOC tokens).
    * ``exclude_soc``  : mean over attended tokens **excluding** ``<SOC_G>, <SOC_L>``; falls back to ``all`` if empty.
    * ``only_soc``     : mean over SOC tokens only; falls back to ``all`` if none.
    * ``locals_only``  : mean over ``<SOC_L>`` only; falls back to ``all`` if none.
    * ``eos``          : take the final *attended* hidden state (no averaging).

**Runtime**
- ``SOC_BATCH``   : batch size (default: 8).
- ``SOC_WORKERS`` : DataLoader workers (default: 8).
- ``SOC_VERBOSE`` : ``1`` enables logs (default: ``1``).
- ``SOC_LOG_EXAMPLES`` : how many examples to print (default: 3).
- ``SOC_LOG_KEEP_UIDS``: how many layer UIDs to print from warmup (default: 25).
- ``SOC_DRYRUN``  : ``1`` processes one batch then exits (default: ``0``).

Assumptions & Fallbacks
-----------------------
- The tokenizer must contain ``<SOC_G>`` (and ideally ``<SOC_L>``) when
  ``SOC_INJECT`` is ``global`` or ``full``; otherwise injection is disabled and
  a clear error is raised for the global token.
- If the number of ``<SOC_L>`` tokens does not match the number of available
  local vectors, the first ``k = min(counts)`` positions are injected and a log
  warning is emitted.
- Pooling modes that would select *no* tokens safely fall back to ``all``.
- Padding: token IDs are padded, but attention masks prevent pads from
  influencing attention or pooling. Local vectors are zero-padded and masked.

Inputs
------
- ``OOO_INDEX`` CSV with columns:
  ``clip_id, global_path, locals_path, meta_path, caption``.
- ``global_path`` : npy/npz -> float32 vector ``[768]``.
- ``locals_path`` : npz with multiple arrays of ``[768]`` + optional indices.
- ``meta_path``   : pickle containing tokens/words and optional local positions.
- ``SIM_RSM``     : CSV square matrix ``[N, N]`` of human similarities.

Outputs
-------
- ``SAVE_FEATURES_DIR``: one ``.npz`` per layer containing pooled features and uids.
- ``SOC_RESULTS_CSV``  : per-layer Spearman rho/p-values vs. human RSM.
- Stdout prints the best-performing layer and basic diagnostics.

Quickstart
----------
Baseline (no injection; last-token readout):
    $ export SOC_INJECT=none
    $ export SOC_POOL=eos
    $ python ooo_rsa_gemma_with_pool.py

Global-only injection but *exclude* SOC tokens from pooling (tests indirect effects):
    $ export SOC_INJECT=global
    $ export SOC_POOL=exclude_soc
    $ python ooo_rsa_gemma_with_pool.py

Full injection (global+locals), average all tokens:
    $ export SOC_INJECT=full
    $ export SOC_POOL=all
    $ python ooo_rsa_gemma_with_pool.py

Why these knobs matter
----------------------
- ``SOC_INJECT`` decides **what** social information is written into the sequence
  (none / global / global+local).
- ``SOC_POOL`` decides **where** you **read** the representation from (all tokens,
  exclude SOC, SOC-only, locals-only, or last attended token). This lets you
  distinguish direct readout of injected signals from *indirect propagation*
  into normal tokens—critical for causal claims in RSA.

Dependencies
------------
- ``transformers``, ``torch``, ``numpy``, ``pandas``, ``scikit-learn``,
  ``scipy``, ``tqdm``, and **deepjuice** (``get_feature_maps``,
  ``get_feature_map_srps``).

"""
import os, re, json, pickle
from typing import List, Dict, Any, Tuple, Optional
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from tqdm import tqdm

# deepjuice
from deepjuice.extraction import get_feature_maps
from deepjuice.reduction import get_feature_map_srps

# ========= Logging controls =========
VERBOSE = os.environ.get("SOC_VERBOSE", "1") != "0"
LOG_EXAMPLES = int(os.environ.get("SOC_LOG_EXAMPLES", "3"))
LOG_KEEP_UIDS = int(os.environ.get("SOC_LOG_KEEP_UIDS", "25"))
DRYRUN = os.environ.get("SOC_DRYRUN", "0") == "1"

# ========= Injection + Pooling options =========
INJECT_MODE = os.environ.get("SOC_INJECT", "full").lower()       # full | global | none
POOL_MODE   = os.environ.get("SOC_POOL", "all").lower()          # all | exclude_soc | only_soc | locals_only | eos

def _log(msg: str):
    if VERBOSE:
        print(msg, flush=True)

def _truncate(s: str, n=160):
    return (s if len(s) <= n else s[:n] + " …")

# ======== User paths to set ========
OUTPUT_DIR = os.environ.get("SOC_OUTPUT_DIR", "/path/to/your/tokenizer_and_model_dir")
CHECKPOINT  = os.environ.get("SOC_PROJECTOR_CKPT", "/path/to/checkpoints/projector_only.pt")
OOO_INDEX   = os.environ.get("SOC_OOO_INDEX", "/mnt/data/ooo_index_ordered_for_rsa.csv")
SIM_RSM     = os.environ.get("SOC_SIM_RSM", "/path/to/sim_judge_train_rsm.csv")
SAVE_FEATURES_DIR = os.environ.get("SOC_SAVE_FEATS", "/tmp/gemma_soc_feats")
RESULTS_CSV       = os.environ.get("SOC_RESULTS_CSV", "/tmp/gemma_soc_ooo_rsa.csv")

BATCH = int(os.environ.get("SOC_BATCH", "8"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SOC_G = "<SOC_G>"
SOC_L = "<SOC_L>"

# Separate tokenizer vs model weights
TOKENIZER_DIR = os.environ.get("SOC_TOKENIZER_DIR", os.environ.get("SOC_OUTPUT_DIR", ""))
LM_PATH_OR_ID = os.environ.get("SOC_LM", "google/gemma-2-2b")  # local dir with weights OR HF id

# ----------------------
# Load tokenizer/model
# ----------------------
def load_tokenizer_and_lm(tokenizer_dir: str, lm_src: str):
    _log(f"[LOAD] tokenizer_dir={tokenizer_dir}")
    _log(f"[LOAD] lm_src={lm_src}")
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token

    lm = AutoModelForCausalLM.from_pretrained(lm_src, torch_dtype=torch.float32)
    # Resize (add SOC tokens if tokenizer has them), then move to device
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
    """
    Returns inline text according to INJECT_MODE:
      - 'none'   : plain caption (no SOC tokens)
      - 'global' : <SOC_G> + caption
      - 'full'   : <SOC_G> + <SOC_L> at local positions
    """
    if (not words) and fallback_caption:
        words = fallback_caption.split()
    if not words:
        return "(no transcript)" if INJECT_MODE == "none" else f"{SOC_G} (no transcript)"

    if INJECT_MODE == "none":
        return " ".join(words)
    if INJECT_MODE == "global":
        return f"{SOC_G} " + " ".join(words)

    # full injection
    loc = set(i for i in local_idx if 0 <= i < len(words))
    out = [SOC_G]
    for i, w in enumerate(words):
        out.append(w)
        if i in loc: out.append(SOC_L)
    return " ".join(out)

def load_pack_row(row: Dict[str, Any], max_locals: Optional[int] = 50):
    g = np.load(row["global_path"]).astype("float32").reshape(-1)        # (768,)
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
# Dataset / Collate for deepjuice
# ----------------------
class OOODataset(Dataset):
    def __init__(self, index_csv):
        df = pd.read_csv(index_csv)
        need = {"clip_id","global_path","locals_path","meta_path","caption"}
        missing = sorted(list(need - set(df.columns)))
        if missing:
            raise ValueError(f"Missing required columns in {index_csv}: {missing}")
        self.items = []
        for i, row in df.reset_index(drop=True).iterrows():
            inline, g, L = load_pack_row(row, max_locals=50)
            self.items.append({
                "uid": str(row.get("clip_id", i)),
                "uid_code": i,              # stable integer handle
                "text": inline,
                "g": g,
                "L": L,
            })
        _log(f"[DATASET] loaded {len(self.items)} rows from {index_csv} (inject_mode={INJECT_MODE}, pool={POOL_MODE})")
        for j in range(min(LOG_EXAMPLES, len(self.items))):
            it = self.items[j]
            gl = it["text"].count(SOC_G)
            ll = it["text"].count(SOC_L)
            _log(f"[DATASET example {j}] uid={it['uid']} G={gl} L={ll} text='{_truncate(it['text'])}'")

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def make_collate(tok, soc_g_id: int, soc_l_id: Optional[int]):
    def collate_cpu(batch):
        texts       = [b["text"] for b in batch]
        uid_codes   = [int(b["uid_code"]) for b in batch]
        locals_lists= [b["L"] for b in batch]
        globals_np  = [b["g"] for b in batch]

        enc = tok(texts, return_tensors="pt", padding=True, truncation=False)
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
            _log(f"[COLLATE] input_ids.shape={tuple(input_ids.shape)} "
                 f"attn.shape={tuple(attn.shape)} g.shape={tuple(g.shape)} "
                 f"l_pad.shape={tuple(l_pad.shape)} l_mask.sum={int(l_mask.sum())}")
            for b in range(min(LOG_EXAMPLES, B)):
                ids = input_ids[b].tolist()
                g_pos = [i for i,tokid in enumerate(ids) if tokid == soc_g_id]
                l_pos = [i for i,tokid in enumerate(ids) if soc_l_id is not None and tokid == soc_l_id]
                l_count = int(l_mask[b].sum().item())
                _log(f"[COLLATE ex {b}] uid_code={uid_codes[b]} "
                     f"tokens={len(ids)} Gpos={g_pos} Lpos={l_pos} "
                     f"Lvecs={l_count} text='{_truncate(texts[b])}'")
                if soc_l_id is not None and len(l_pos) != l_count and INJECT_MODE != "none":
                    _log(f"[WARN] L positions ({len(l_pos)}) != local vectors ({l_count}) → min-align downstream.")

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
# Pooling utilities
# ----------------------
def pool_to_2d(h, ids, attn, pool_mode, socg_id, socl_id):
    """
    h:   [B,T,H], ids: [B,T] (long), attn: [B,T] (0/1 floats)
    returns [B,H] pooled according to pool_mode.
    """
    if h.dim() != 3:
        return h  # already 2D or other
    B, T, H = h.shape
    m_all = attn.bool()

    if pool_mode == "all":
        m = m_all
    elif pool_mode == "exclude_soc":
        soc = (ids == socg_id)
        if socl_id is not None:
            soc = soc | (ids == socl_id)
        m = m_all & (~soc)
        empty = m.sum(dim=1) == 0
        if empty.any():  # fallback to all
            m[empty] = m_all[empty]
    elif pool_mode == "only_soc":
        m = (ids == socg_id)
        if socl_id is not None:
            m = m | (ids == socl_id)
        empty = m.sum(dim=1) == 0
        if empty.any():
            m[empty] = m_all[empty]
    elif pool_mode == "locals_only":
        if socl_id is None:
            m = m_all
        else:
            m = (ids == socl_id)
            empty = m.sum(dim=1) == 0
            if empty.any():
                m[empty] = m_all[empty]
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
# SOC wrapper w/ forward() for deepjuice hooks
# ----------------------
class SOCWrapper(nn.Module):
    def __init__(self, lm, tok, projector, soc_g_id: int, soc_l_id: int):
        super().__init__()
        self.lm  = lm
        self.tok = tok
        self.proj= projector
        self.soc_g = soc_g_id
        self.soc_l = soc_l_id
        self._last_meta = None

    def forward(self, *args, **batch):
        # Support positional dict
        if len(args) == 1 and isinstance(args[0], dict):
            batch = args[0]

        dev = next(self.lm.parameters()).device
        emb_dtype = self.lm.get_input_embeddings().weight.dtype

        # Required tensors
        input_ids = batch["input_ids"].to(dev)
        attn      = batch["attention_mask"].to(dev)
        g         = batch["global_vec"].to(dev, dtype=torch.float32)
        l_pad     = batch["local_vecs_padded"].to(dev, dtype=torch.float32)
        l_mask    = batch["local_mask"].to(dev)

        # Optional tracking for RSA ordering
        uid_code = batch.get("uid_code", None)
        self._last_meta = {"uid_code": uid_code.detach().cpu()} if uid_code is not None else None

        # Base token embeddings
        embs = self.lm.get_input_embeddings()(input_ids)

        if INJECT_MODE in ("full", "global"):
            # Inject GLOBALS at <SOC_G>
            is_g = (input_ids == self.soc_g)
            if is_g.any():
                proj_g = self.proj(g).to(emb_dtype)
                rows_idx = is_g.any(dim=1).nonzero(as_tuple=True)[0]
                repeats  = is_g.sum(dim=1)[rows_idx]
                embs[is_g] = proj_g[rows_idx].repeat_interleave(repeats, dim=0)

        if INJECT_MODE == "full" and self.soc_l is not None:
            # Inject LOCALS at <SOC_L> (align first k occurrences)
            B = input_ids.size(0)
            for b in range(B):
                pos = (input_ids[b] == self.soc_l).nonzero(as_tuple=True)[0]
                if pos.numel() == 0:
                    continue
                valid_L = l_mask[b].nonzero(as_tuple=True)[0]
                k = min(pos.numel(), valid_L.numel())
                if k > 0:
                    proj_l = self.proj(l_pad[b, valid_L[:k], :].to(dev)).to(emb_dtype)
                    embs[b, pos[:k], :] = proj_l

        # Return standard HF outputs; deepjuice reads hooks via get_feature_maps
        return self.lm(inputs_embeds=embs, attention_mask=attn, return_dict=True)

    def pop_last_meta(self):
        m = self._last_meta
        self._last_meta = None
        return m


# ----------------------
# Main: deepjuice batchwise + RSA (var-length safe)
# ----------------------
def main():
    os.makedirs(SAVE_FEATURES_DIR, exist_ok=True)

    tok, lm = load_tokenizer_and_lm(TOKENIZER_DIR, LM_PATH_OR_ID)
    lm.eval()

    socg_id = tok.convert_tokens_to_ids(SOC_G)
    socl_id = tok.convert_tokens_to_ids(SOC_L) if SOC_L in tok.get_vocab() else None

    # Only require SOC tokens to exist if we are injecting
    if INJECT_MODE in ("full", "global"):
        if socg_id is None or socg_id == tok.unk_token_id:
            raise RuntimeError(f"{SOC_G} token not present in tokenizer vocab from {TOKENIZER_DIR}")

    proj = load_projector_from_ckpt(CHECKPOINT, lm.config.hidden_size)
    wrapped = SOCWrapper(lm, tok, proj, socg_id, socl_id).eval()

    ds = OOODataset(OOO_INDEX)
    collate_cpu = make_collate(tok, socg_id, socl_id)
    dl = DataLoader(
        ds, batch_size=BATCH, shuffle=False,
        num_workers=int(os.environ.get("SOC_WORKERS", "8")),
        prefetch_factor=2, persistent_workers=True,
        pin_memory=True, collate_fn=collate_cpu
    )

    # ---- Warmup on one real batch (discover module UIDs to keep) ----
    warm_batch = next(iter(dl))
    with torch.no_grad():
        warm_maps = get_feature_maps(
            wrapped, warm_batch,
            flatten=False,              # keep [B, T, D] if sequence exists
            save_inputs=False,
            remove_duplicates=True,
            report_irregularities=False  # quieter logs
        )
    keep_uids = sorted(warm_maps.keys())
    _log(f"[WARMUP] discovered {len(keep_uids)} module UIDs; showing first {min(LOG_KEEP_UIDS, len(keep_uids))}:")
    for u in keep_uids[:LOG_KEEP_UIDS]:
        fm = warm_maps[u]
        shape = tuple(fm.shape) if hasattr(fm, "shape") else "NA"
        _log(f"  - {u} shape={shape}")

    # ---- On-the-fly extraction (no preallocation, handles var-length) ----
    uid_code_order: List[int] = []
    features_accum: Dict[str, List[np.ndarray]] = {u: [] for u in keep_uids}

    _log("[EXTRACT] starting per-batch deepjuice extraction (var-length safe)…")
    for bi, batch in enumerate(tqdm(dl, desc="Extracting batches", total=len(dl))):
        with torch.no_grad():
            fmap_dict = get_feature_maps(
                wrapped, batch,
                flatten=False,          # keep time dim; we'll pool it to 2D
                save_inputs=False,
                remove_duplicates=True,
                report_irregularities=False
            )

        meta = wrapped.pop_last_meta()
        if meta is not None:
            uid_code_order.extend(meta["uid_code"].tolist())

        # Show example texts for first couple of batches
        if VERBOSE and bi < 2:
            texts = tok.batch_decode(batch["input_ids"], skip_special_tokens=False)
            for j in range(min(LOG_EXAMPLES, len(texts))):
                _log(f"[BATCH {bi} ex {j}] decoded='{_truncate(texts[j])}'")

        ids  = batch["input_ids"]
        attn = batch["attention_mask"].to(dtype=torch.float32)

        for uid, fmap in fmap_dict.items():
            if uid not in features_accum:
                continue

            t = fmap  # [B,T,H] or [B,D]...
            if t.dim() == 3:
                # Pool according to SOC_POOL
                t2d = pool_to_2d(t, ids.to(t.device), attn.to(t.device), POOL_MODE, socg_id, socl_id)  # [B,H]
            elif t.dim() > 3:
                t2d = t.flatten(start_dim=1)
            else:
                t2d = t

            X = get_feature_map_srps(t2d, device=DEVICE)  # SRP to 2D proj (defaults)
            X = X.detach().float().cpu().numpy()
            features_accum[uid].append(X)

        if DRYRUN:
            _log("[DRYRUN] stopping after first batch by request (SOC_DRYRUN=1).")
            break

    # ---- Concatenate per-UID feature matrices in dataloader order ----
    features_by_uid = {uid: np.concatenate(chunks, axis=0)
                       for uid, chunks in features_accum.items() if len(chunks) > 0}
    uid_order = [ds.items[i]["uid"] for i in uid_code_order]
    _log(f"[EXTRACT] collected features for {len(uid_order)} items across {len(features_by_uid)} UIDs.")

    # RSA (assumes OOO_INDEX order matches SIM_RSM)
    sim_rsm = pd.read_csv(SIM_RSM).values
    if sim_rsm.shape[0] != len(uid_order):
        _log(f"[WARN] RSM size {sim_rsm.shape} != N items {len(uid_order)}; will min-align.")
    n = min(sim_rsm.shape[0], len(uid_order))
    tri = np.triu_indices(n, k=1)
    sim_flat = sim_rsm[:n, :n][tri]

    results = []
    for layer_uid, X in features_by_uid.items():
        Xn = X[:n]
        # persist per-layer features (optional)
        out_npz = os.path.join(SAVE_FEATURES_DIR, f"{layer_uid.replace('/', '_')}.npz")
        np.savez(out_npz, X=Xn, uids=np.array(uid_order[:n], dtype=object))

        rsm = 1 - pairwise_distances(Xn, metric='correlation')
        model_flat = rsm[tri]
        rho, p = spearmanr(sim_flat, model_flat)
        results.append({"layer_uid": layer_uid, "spearman": float(rho), "p": float(p)})

        _log(f"[RSA] {layer_uid}: r={rho:.4f}, p={p:.2e}")

    df_res = pd.DataFrame(results)
    if len(df_res):
        best = df_res.loc[df_res['spearman'].idxmax()]
        print(f"[FINISHED] Best layer: {best['layer_uid']}  r={best['spearman']:.4f}  p={best['p']:.2e}")
    else:
        print("[FINISHED] No results computed.")
    df_res.to_csv(RESULTS_CSV, index=False)
    print(f"\n[DONE] Saved RSA results to: {RESULTS_CSV}")

if __name__ == "__main__":
    main()

