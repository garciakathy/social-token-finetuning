# -*- coding: utf-8 -*-
# logit_lens_from_jsonl_weights.py
# Logit-lens + compact attention→token grid for next-utterance JSONL.
# - VIS vs NOVIS layer metrics (CE, rank, Δtrue, residual proj)
# - Attention grid that (by default) shows *all* prompt tokens as columns
# - No in-cell token text by default (cleaner)
# - Figure width scales with token count
# - Auto-detect projector layout (ScaleShift vs LayerNorm) to load checkpoint

import os, re, json, argparse
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from transformers import AutoTokenizer, AutoModelForCausalLM

SOC_G = "<SOC_G>"
SOC_L = "<SOC_L>"

# ---------------- Projector ----------------
class ScaleShift(nn.Module):
    def __init__(self, dim: int, init_scale: float = 1.0):
        super().__init__()
        self.g = nn.Parameter(torch.full((dim,), float(init_scale)))
        self.b = nn.Parameter(torch.zeros(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        x = x / rms
        return x * self.g + self.b

class VisualProjector(nn.Module):
    """
    Matches your training-time projector:
      - use_scaleshift=True -> [ScaleShift(in), Linear, GELU, (Dropout), Linear, ScaleShift(out)]
      - use_scaleshift=False -> [LayerNorm, Linear, GELU, Linear]
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 use_scaleshift: bool = True,
                 init_scale_in: float = 1.0,
                 init_scale_out: float = 1.0,
                 dropout: float = 0.0):
        super().__init__()
        mods: List[nn.Module] = []
        if use_scaleshift:
            mods.append(ScaleShift(in_dim, init_scale=init_scale_in))
        else:
            mods.append(nn.LayerNorm(in_dim))
        mods += [nn.Linear(in_dim, out_dim), nn.GELU()]
        if dropout and dropout > 0:
            mods.append(nn.Dropout(dropout))
        mods.append(nn.Linear(out_dim, out_dim))
        if use_scaleshift:
            mods.append(ScaleShift(out_dim, init_scale=init_scale_out))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)

def _unwrap_projector_sd(sd_like):
    sd = sd_like
    if isinstance(sd, dict) and "projector" in sd and isinstance(sd["projector"], dict):
        return sd["projector"]
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        cand = sd["state_dict"]
        has_proj_prefix = any(k.startswith("projector.") for k in cand.keys())
        if has_proj_prefix:
            return {k[len("projector."):] if k.startswith("projector.") else k: v for k, v in cand.items()}
        return cand
    return sd

def _infer_proj_layout_from_keys(keys: list[str]) -> Tuple[bool, float]:
    """
    Infer (use_scaleshift, dropout_value_to_insert) from state_dict keys.
    """
    use_scaleshift = any(k.endswith(".g") or k.endswith(".b") for k in keys) or ("net.0.g" in keys)
    max_idx = -1
    for k in keys:
        if k.startswith("net."):
            try:
                idx = int(k.split(".")[1])
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                pass
    dropout = 0.1 if (use_scaleshift and max_idx >= 5) else 0.0
    return use_scaleshift, dropout

def load_projector(ckpt_path: str, in_dim: int, out_dim: int, device) -> nn.Module:
    try:
        sdfile = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # torch ≥2.4
    except TypeError:
        sdfile = torch.load(ckpt_path, map_location="cpu")
    sd = _unwrap_projector_sd(sdfile)
    if not isinstance(sd, dict):
        raise RuntimeError(f"[projector] unexpected checkpoint format at {ckpt_path}")

    keys = list(sd.keys())
    use_scaleshift, dropout = _infer_proj_layout_from_keys(keys)
    proj = VisualProjector(in_dim, out_dim,
                           use_scaleshift=use_scaleshift,
                           init_scale_in=1.0,
                           init_scale_out=1.0,
                           dropout=dropout).to(device).to(torch.float32)
    try:
        proj.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        print(f"[warn] strict load failed for projector ({e}). Retrying with strict=False.")
        proj.load_state_dict(sd, strict=False)
    proj.eval()
    layout = "ScaleShift(+calibrator)" if use_scaleshift else "LayerNorm"
    print(f"[projector] loaded from {ckpt_path} | layout={layout} dropout={dropout}")
    return proj

# -------------- Tokenizer / LM helpers --------------
def ensure_special_tokens(tokenizer, model, tokens=(SOC_G, SOC_L), zero_init=True, verbose=True):
    vocab = tokenizer.get_vocab()
    to_add = [t for t in tokens if t not in vocab]
    if to_add:
        if verbose: print(f"[tokens] adding {to_add} as additional_special_tokens")
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        if hasattr(tokenizer, "unique_no_split_tokens"):
            for t in to_add:
                if t not in tokenizer.unique_no_split_tokens:
                    tokenizer.unique_no_split_tokens.append(t)
        model.resize_token_embeddings(len(tokenizer))
        if zero_init:
            with torch.no_grad():
                emb = model.get_input_embeddings().weight
                for t in to_add:
                    tid = tokenizer.convert_tokens_to_ids(t)
                    emb[tid].zero_()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.eos_token_id

def _final_norm_module(model):
    for attr in ["model", "transformer", "backbone"]:
        m = getattr(model, attr, None)
        if m is not None:
            for n in ["norm", "ln_f", "final_layernorm"]:
                if hasattr(m, n):
                    return getattr(m, n)
    class _I(torch.nn.Module):
        def forward(self, x): return x
    return _I()

def _lm_head(model):
    return getattr(model, "lm_head", None)

# -------------- JSONL row helpers --------------
def parse_scene_part(clip_id: str) -> Tuple[str, Optional[str]]:
    m = re.match(r"^(.*)_P(\d+)$", str(clip_id))
    if m: return m.group(1), f"P{m.group(2)}"
    return str(clip_id), None

def find_row(jsonl_path: str, clip_id: str, participant: Optional[str] = None) -> Dict[str, Any]:
    want_root, want_part = parse_scene_part(clip_id)
    with open(jsonl_path, "r") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    # exact match
    for r in rows:
        if str(r.get("clip_id","")) == clip_id:
            return r
    # scene_root + participant
    p_need = participant or want_part
    if p_need:
        for r in rows:
            root = str(r.get("scene_root", ""))
            part = str(r.get("participant", ""))
            if root == want_root and part == p_need:
                return r
    # fallback: same scene_root
    for r in rows:
        if str(r.get("scene_root","")) == want_root:
            return r
    raise ValueError(f"Could not find a row for clip_id='{clip_id}' (or scene_root match).")

def build_prompt_from_row(row: Dict[str, Any], use_locals_count: bool, max_locals: int) -> Tuple[str,int]:
    """
    Return (prompt_text, locals_count_inserted_or_existing).
    If prompt already contains <SOC_L>, keep it (training-time count).
    Else, if use_locals_count=True and locals_npz present, append that many <SOC_L>.
    Warn if we expected to insert but had no npz or zero keys (fallback to original).
    """
    prompt = str(row.get("prompt_text","")).strip()
    existing = prompt.count(SOC_L)
    if existing > 0:
        return prompt, existing

    if use_locals_count:
        npz = row.get("locals_npz")
        if isinstance(npz, str) and os.path.exists(npz):
            arr = np.load(npz)
            try:
                keys = list(arr.keys())
            except Exception:
                keys = list(arr.files)
            n_loc = min(len(keys), max_locals if max_locals>0 else len(keys))
            if n_loc > 0:
                prompt2 = (prompt + " " + " ".join([SOC_L]*n_loc)).strip()
                return prompt2, n_loc
            else:
                print("[warn] use-locals-count enabled but locals_npz contains 0 vectors; "
                      "falling back to the original prompt without <SOC_L> insertions.")
        else:
            print("[warn] use-locals-count enabled but no locals_npz on row; "
                  "falling back to the original prompt without <SOC_L> insertions.")
    return prompt, 0

# -------------- Scores / misc --------------
def layerwise_ce(logits, gold_id: int):
    logp = torch.log_softmax(logits, dim=-1)
    return float(-logp[gold_id].item())

def token_rank_for_id(logits_row: torch.Tensor, target_id: int) -> int:
    target_logit = logits_row[target_id]
    return 1 + int((logits_row > target_logit).sum().item())

def _pretty_tok(t: str, max_len: int = 8) -> str:
    if t.startswith("▁"):
        t = "␣" + t[1:]
    t = t.replace("<0x0A>", "↵")
    return t if len(t) <= max_len else (t[:max(1, max_len-1)] + "…")

# -------------- Visual injection --------------
@torch.no_grad()
def inject_visuals_to_embeds(model, tokenizer, input_ids, proj_global, proj_locals):
    emb = model.get_input_embeddings()(input_ids)  # [1,T,H]
    dtype = emb.dtype
    id_g = tokenizer.convert_tokens_to_ids(SOC_G)
    id_l = tokenizer.convert_tokens_to_ids(SOC_L)
    if proj_global is not None:
        pos = (input_ids == id_g).nonzero(as_tuple=False)
        if pos.numel() > 0:
            b = pos[:,0]; t = pos[:,1]
            emb[b, t, :] = proj_global[b, :].to(dtype=dtype)
    if proj_locals is not None and proj_locals.size(0) > 0:
        lpos = (input_ids[0] == id_l).nonzero(as_tuple=False).squeeze(-1)
        L = min(lpos.numel(), proj_locals.size(0))
        if L > 0:
            emb[0, lpos[:L], :] = proj_locals[:L, :].to(dtype=dtype)
    return emb

# -------------- Compact attention→token grid --------------
def compact_attn_token_grid(
    attn_layers: List[torch.Tensor],      # list of [1,H,T,T]
    ids_1d: torch.Tensor,                 # [T]
    tokenizer,
    qpos: int,                            # query position (boundary)
    title: str = "Mean attention (pred pos) → tokens",
    topk: int = 40,
    max_cols: int = -1,                   # <=0 → keep ALL prompt tokens
    min_quantile: float = 0.85,
    log_scale: bool = True,
    annotate: str = "none",               # "none" | "topk"
    ann_topk_per_row: int = 2,
    cmap: str = "magma",
    layer_step: int = 1,
    token_width: float = 0.35,            # inches per token for figure width
    save_path: Optional[str] = None,
):
    if not attn_layers:
        print("[grid] no attentions; skipping plot.")
        return

    # Convert to float32 CPU and (optionally) downsample layers
    A = [a.to(dtype=torch.float32).cpu() for a in attn_layers]
    if layer_step > 1:
        A = A[::layer_step]
    L = len(A)
    T = A[0].shape[-1]

    # Average over heads at the prediction position
    S_list = []
    for li in range(L):
        heads = A[li][0, :, qpos, :]           # [H,T]
        S_list.append(heads.mean(dim=0, keepdim=True))
    S = torch.cat(S_list, dim=0)               # [L,T]

    # ---- Keep only PROMPT tokens (keys ≤ qpos)
    prompt_len = qpos + 1
    prompt_mask = torch.zeros(T, dtype=torch.bool)
    prompt_mask[:prompt_len] = True

    # Determine columns to keep (within prompt)
    if max_cols is None or max_cols <= 0 or max_cols >= prompt_len:
        # keep ALL prompt tokens
        idx_all = torch.arange(0, prompt_len)
    else:
        global_mean = S.mean(dim=0)                           # [T]
        keep_mask = prompt_mask.clone()

        thr = torch.quantile(global_mean[keep_mask], min_quantile).item()
        keep_mask &= (global_mean >= thr)

        # always keep SOCs + a few neighbors near boundary
        ids = ids_1d.cpu()
        is_soc = (ids == tokenizer.convert_tokens_to_ids(SOC_G)) | (ids == tokenizer.convert_tokens_to_ids(SOC_L))
        keep_mask |= (is_soc & prompt_mask)
        for t in range(max(0, qpos-3), qpos+1):
            keep_mask[t] = True

        # explicit global top-k within prompt
        cand = torch.nonzero(prompt_mask, as_tuple=False).squeeze(-1)
        if cand.numel() > 0:
            scores = global_mean[cand]
            k = min(int(topk), cand.numel())
            pick = cand[torch.topk(scores, k=k).indices]
            keep_mask[pick] = True

        # cap number of columns
        idx_all = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
        if idx_all.numel() > max_cols:
            sc = global_mean[idx_all]
            pick = torch.topk(sc, k=max_cols).indices
            idx_all = idx_all[pick]
            idx_all = torch.sort(idx_all).values

    M = S[:, idx_all].numpy()                 # [L,C]
    tok_ids = ids_1d[idx_all].cpu().numpy().tolist()
    toks = [tokenizer.convert_ids_to_tokens(int(tid)) for tid in tok_ids]

    # ---- Plot
    fig_h = 8
    fig_w = max(10, token_width * max(1, len(toks)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vpos = M[M > 0]
    vmin = float(vpos.min()) if vpos.size else 1e-6
    vmax = float(M.max()) if M.size else 1.0
    norm = LogNorm(vmin=max(1e-6, vmin), vmax=max(vmin*1.001, vmax)) if log_scale else None

    im = ax.imshow(M, aspect="auto", interpolation="nearest", norm=norm, cmap=cmap)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("layer", fontsize=10)
    ax.set_xlabel("tokens (prompt only)", fontsize=10)

    ax.set_xticks(range(len(toks)))
    ax.set_xticklabels([_pretty_tok(t) for t in toks], rotation=80, ha="right", fontsize=7)
    ax.set_yticks(range(L))
    ax.set_yticklabels([str(i*layer_step) for i in range(L)], fontsize=8)

    # frame SOC columns
    soc_cols = [i for i, tid in enumerate(tok_ids)
                if tid in (tokenizer.convert_tokens_to_ids(SOC_G), tokenizer.convert_tokens_to_ids(SOC_L))]
    for c in soc_cols:
        ax.add_patch(plt.Rectangle((c-0.5, -0.5), 1, L, fill=False, lw=1.2, edgecolor="black"))

    # optional in-cell text
    if annotate == "topk" and len(toks) > 0:
        for r in range(L):
            row = M[r]
            k = min(ann_topk_per_row, len(row))
            if k <= 0: continue
            cols = np.argpartition(-row, kth=k-1)[:k]
            for c in cols:
                ax.text(c, r, _pretty_tok(toks[c], 8), ha="center", va="center",
                        fontsize=7, color="black", alpha=0.9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean attention", rotation=90)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=250)
        plt.close(fig)
        print(f"[plot] saved {save_path}")
    else:
        plt.show()

# -------------- Core analysis --------------
@torch.no_grad()
def run_all_analyses(
    model, tokenizer,
    input_ids, attention_mask,
    inputs_embeds_novis, inputs_embeds_vis,
    target_id: int,
    qpos: int,                              # boundary (prompt-only)
    out_dir: Optional[str],
    topk: int = 20,
    print_layers: int = 8,
    save_plots: bool = False,
    # grid options
    grid_topk: int = 40,
    grid_max_cols: int = -1,
    grid_min_quantile: float = 0.85,
    grid_ann_topk_per_row: int = 2,
    grid_cmap: str = "magma",
    grid_layer_step: int = 1,
    grid_annotate: str = "none",
    grid_keep_all: bool = True,
    grid_token_width: float = 0.35,
):
    norm = _final_norm_module(model)
    head = _lm_head(model)
    assert head is not None, "Model missing lm_head"

    # Run forward (try with attentions first)
    def forward_with(maybe_attn: bool):
        return model(inputs_embeds=inputs_embeds_novis, attention_mask=attention_mask,
                     output_hidden_states=True, output_attentions=maybe_attn,
                     use_cache=False, return_dict=True), \
               model(inputs_embeds=inputs_embeds_vis,   attention_mask=attention_mask,
                     output_hidden_states=True, output_attentions=maybe_attn,
                     use_cache=False, return_dict=True)

    try:
        outN, outV = forward_with(maybe_attn=True)
        if outN.attentions is None or outN.attentions[0] is None:
            raise RuntimeError("attentions returned None")
    except Exception as e:
        print(f"[note] attention not available ({e}); falling back without attention.")
        outN, outV = forward_with(maybe_attn=False)
        outN.attentions = None
        outV.attentions = None

    hsN: List[torch.Tensor] = list(outN.hidden_states)
    hsV: List[torch.Tensor] = list(outV.hidden_states)
    attN = list(outN.attentions) if outN.attentions is not None else None
    attV = list(outV.attentions) if outV.attentions is not None else None

    W_U = head.weight.detach().to(torch.float32)
    L = len(hsN)

    # ---- Per-layer metrics at the boundary qpos
    rows_ce, rows_rank, rows_delta_true, rows_contrib = [], [], [], []
    for li in range(L):
        hN = norm(hsN[li])[:, qpos, :].to(torch.float32)
        hV = norm(hsV[li])[:, qpos, :].to(torch.float32)
        logitsN = (hN @ W_U.T)[0]
        logitsV = (hV @ W_U.T)[0]

        ceN = layerwise_ce(logitsN, target_id)
        ceV = layerwise_ce(logitsV, target_id)
        rN = token_rank_for_id(logitsN, target_id)
        rV = token_rank_for_id(logitsV, target_id)
        d_true = float((logitsV[target_id] - logitsN[target_id]).item())

        rows_ce.append({"layer_idx": li, "ce_novis": ceN, "ce_vis": ceV})
        rows_rank.append({"layer_idx": li, "rank_novis": rN, "rank_vis": rV})
        rows_delta_true.append({"layer_idx": li, "delta_logit_true": d_true})

        u_y = W_U[target_id]
        hN_u = nn.functional.normalize(hN[0], dim=-1)
        hV_u = nn.functional.normalize(hV[0], dim=-1)
        u_u  = nn.functional.normalize(u_y,   dim=-1)
        rows_contrib.append({
            "layer_idx": li,
            "proj_novis": float(torch.dot(hN_u, u_u).item()),
            "proj_vis":   float(torch.dot(hV_u, u_u).item()),
        })

    df_ce    = pd.DataFrame(rows_ce)
    df_rank  = pd.DataFrame(rows_rank)
    df_dtrue = pd.DataFrame(rows_delta_true)
    df_contr = pd.DataFrame(rows_contrib)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        df_ce.to_csv(os.path.join(out_dir, "layerwise_ce.tsv"), sep="\t", index=False)
        df_rank.to_csv(os.path.join(out_dir, "target_rank.tsv"), sep="\t", index=False)
        df_dtrue.to_csv(os.path.join(out_dir, "delta_true.tsv"), sep="\t", index=False)
        df_contr.to_csv(os.path.join(out_dir, "resid_contrib.tsv"), sep="\t", index=False)
        print(f"[saved] TSVs → {out_dir}")

    # console skim
    Lp = min(print_layers, L)
    idxs = sorted(set(int(i) for i in np.linspace(0, L-1, num=Lp, dtype=int).tolist()))
    for li in idxs:
        rce = df_ce[df_ce.layer_idx==li].iloc[0]
        rrk = df_rank[df_rank.layer_idx==li].iloc[0]
        rdt = df_dtrue[df_dtrue.layer_idx==li].iloc[0]
        print(f"  L{li:02d}  CE: novis={rce.ce_novis:.3f} vis={rce.ce_vis:.3f} | "
              f"rank: novis={rrk.rank_novis} vis={rrk.rank_vis} | Δlogit_true={rdt.delta_logit_true:+.3f}")

    # compact attention grids (prompt-only)
    if (attV is not None) and out_dir:
        compact_attn_token_grid(
            attV, input_ids[0], tokenizer, qpos,
            title="Mean attention (pred pos) → tokens [VIS]",
            topk=grid_topk,
            max_cols=(-1 if grid_keep_all else grid_max_cols),
            min_quantile=grid_min_quantile,
            log_scale=True,
            annotate=grid_annotate,
            ann_topk_per_row=grid_ann_topk_per_row,
            cmap=grid_cmap,
            layer_step=grid_layer_step,
            token_width=grid_token_width,
            save_path=os.path.join(out_dir, "attn_token_grid_vis.png"),
        )
        compact_attn_token_grid(
            attN, input_ids[0], tokenizer, qpos,
            title="Mean attention (pred pos) → tokens [NOVIS]",
            topk=grid_topk,
            max_cols=(-1 if grid_keep_all else grid_max_cols),
            min_quantile=grid_min_quantile,
            log_scale=True,
            annotate=grid_annotate,
            ann_topk_per_row=grid_ann_topk_per_row,
            cmap=grid_cmap,
            layer_step=grid_layer_step,
            token_width=grid_token_width,
            save_path=os.path.join(out_dir, "attn_token_grid_novis.png"),
        )
    elif out_dir:
        print("[plots] attention not available; skipped attention grids.")

    return df_ce, df_rank, df_dtrue, df_contr

# -------------- Runner --------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lm  = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    ensure_special_tokens(tok, lm, tokens=(SOC_G, SOC_L), zero_init=True)
    H = lm.config.hidden_size

    # optional: make attention available on some stacks
    if args.force_math_sdp:
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            print("[attn] forced math SDP kernel.")
        except Exception as e:
            print(f"[attn] could not toggle SDP kernels: {e}")
    if args.force_eager_attn:
        switched = False
        for obj in [lm, getattr(lm, "model", None)]:
            if obj is None: continue
            if hasattr(obj, "set_attn_implementation"):
                try:
                    obj.set_attn_implementation("eager"); switched = True
                    print("[attn] set_attn_implementation('eager') OK")
                except Exception:
                    pass
        if hasattr(lm.config, "attn_implementation"):
            lm.config.attn_implementation = "eager"; switched = True or switched
        if not switched:
            print("[attn] could not switch to eager.")

    # --- JSONL row
    row = find_row(args.jsonl, args.clip_id, participant=args.participant)
    target_text = str(row.get("target_text","")).strip()
    prompt_text, locals_in_prompt = build_prompt_from_row(row, args.use_locals_count, args.max_locals)

    # --- Projector & vectors
    projector = load_projector(args.ckpt, 768, H, device) if (args.ckpt and os.path.exists(args.ckpt)) else None
    proj_g = proj_l = None

    gpath = args.global_vec_override or row.get("global_vec")
    if projector is not None and isinstance(gpath, str) and os.path.exists(gpath):
        g = np.load(gpath).astype(np.float32)
        g = torch.from_numpy(g).unsqueeze(0).to(device)
        proj_g = projector(g)
    elif args.ckpt and not os.path.exists(args.ckpt):
        print(f"[warn] projector checkpoint not found at {args.ckpt}; running NOVIS-only.")

    lpath = args.locals_npz_override or row.get("locals_npz")
    npz_keys = []
    if projector is not None and isinstance(lpath, str) and os.path.exists(lpath):
        arr = np.load(lpath)
        try:    npz_keys = list(arr.keys())
        except: npz_keys = list(arr.files)
        if args.max_locals > 0:
            npz_keys = npz_keys[: args.max_locals]
        if npz_keys:
            l = np.stack([arr[k].astype(np.float32) for k in npz_keys], axis=0)
            proj_l = projector(torch.from_numpy(l).to(device))

    # --- Tokenize
    enc_p = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    enc_t = tok(target_text + tok.eos_token, return_tensors="pt", add_special_tokens=False)

    input_ids = torch.cat([enc_p["input_ids"], enc_t["input_ids"]], dim=1).to(device)  # [1,T]
    attention_mask = torch.ones_like(input_ids).to(device)

    # Boundary & gold-next (use boundary at end of prompt to predict first target token)
    prompt_len = int(enc_p["input_ids"].size(1))
    qpos = prompt_len - 1
    gold_next_id = int(enc_t["input_ids"][0,0].item())

    # Debug: why many <SOC_L>?
    local_frames_count = len(row.get("local_frames") or [])
    npz_count = len(npz_keys)
    print(f"\n=== clip: {row.get('clip_id')} ({row.get('participant')}) ===")
    print(f"prompt: {prompt_text[:200]}{'...' if len(prompt_text)>200 else ''}")
    print(f"target: {target_text[:200]}{'...' if len(target_text)>200 else ''}")
    print(f"[debug] SOC_L in prompt: {prompt_text.count(SOC_L)} | local_frames in row: {local_frames_count} | "
          f"locals in npz (after cap): {npz_count} | max_locals arg: {args.max_locals}")
    if local_frames_count and local_frames_count > args.max_locals:
        print("[note] JSONL likely created with a higher max_locals or top-up; "
              "SOC_L count reflects selected local frames, not POS (nouns/verbs).")

    # --- Embeddings (NOVIS vs VIS)
    E_base = lm.get_input_embeddings()(input_ids)
    E_nov  = E_base.clone()
    E_vis  = inject_visuals_to_embeds(lm, tok, input_ids, proj_g, proj_l) if (proj_g is not None or proj_l is not None) else E_base.clone()

    # --- Analyses
    out_dir = args.out_dir
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    run_all_analyses(
        model=lm, tokenizer=tok,
        input_ids=input_ids, attention_mask=attention_mask,
        inputs_embeds_novis=E_nov, inputs_embeds_vis=E_vis,
        target_id=gold_next_id, qpos=qpos,
        out_dir=out_dir, topk=args.topk, print_layers=args.print_layers, save_plots=args.save_plots,
        grid_topk=args.grid_topk,
        grid_max_cols=args.grid_max_cols,
        grid_min_quantile=args.grid_min_quantile,
        grid_ann_topk_per_row=args.grid_ann_topk_per_row,
        grid_cmap=args.grid_cmap,
        grid_layer_step=args.grid_layer_step,
        grid_annotate=args.grid_annotate,
        grid_keep_all=args.grid_keep_all,
        grid_token_width=args.grid_token_width,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="google/gemma-2-2b-it")
    # RESTORED defaults:
    ap.add_argument("--jsonl", type=str, default="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/nextutt_runs/run_ddp_ooo_02/jsonl/test_nextword.jsonl")
    ap.add_argument("--clip-id", type=str, required=True)
    ap.add_argument("--participant", type=str, default="")

    # projector & vectors (RESTORED)
    ap.add_argument("--ckpt", type=str, default="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/nextutt_runs/run_ddp_ooo_02/checkpoints/projector_only.pt")
    ap.add_argument("--max-locals", type=int, default=50)
    ap.add_argument("--use-locals-count", action="store_true")
    ap.add_argument("--global-vec-override", type=str, default="")
    ap.add_argument("--locals-npz-override", type=str, default="")

    # plotting / outputs (RESTORED)
    ap.add_argument("--out-dir", type=str, default="logits/ddp_8")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--print-layers", type=int, default=10)
    ap.add_argument("--save-plots", action="store_true")

    # attention toggles (RESTORED)
    ap.add_argument("--force-eager-attn", action="store_true")
    ap.add_argument("--force-math-sdp", action="store_true")

    # compact grid defaults (RESTORED values)
    ap.add_argument("--grid-topk", type=int, default=30)
    ap.add_argument("--grid-max-cols", type=int, default=30)
    ap.add_argument("--grid-min-quantile", type=float, default=0.85)
    ap.add_argument("--grid-ann-topk-per-row", type=int, default=2)
    ap.add_argument("--grid-layer-step", type=int, default=1)

    # NEW: cmap / cleaner controls (defaults chosen to match your prior behavior + requested cleanup)
    ap.add_argument("--grid-cmap", type=str, default="cividis")
    ap.add_argument("--grid-annotate", choices=["none","topk"], default="none",
                    help="In-cell text. 'none' disables labels inside cells.")
    # keep-all ON by default, but allow turning it off:
    ap.add_argument("--grid-keep-all", dest="grid_keep_all", action="store_true", default=True,
                    help="Show ALL prompt tokens as columns (ignores max-cols / quantile caps).")
    ap.add_argument("--no-grid-keep-all", dest="grid_keep_all", action="store_false",
                    help="Disable keep-all; use --grid-max-cols/quantile selection instead.")
    ap.add_argument("--grid-token-width", type=float, default=0.35,
                    help="Inches per token for figure width when many tokens are shown.")

    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()

