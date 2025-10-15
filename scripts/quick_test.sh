#!/bin/bash
# ========================================
# Quick Test Script (Minimal Data)
# ========================================
# Ultra-fast test with minimal data for debugging
# Runs only a few batches to verify everything works

set -e

echo "========================================="
echo "QUICK TEST MODE"
echo "Running minimal training for debugging"
echo "========================================="

# --- Environment ---
source /data/lisik3/kgarci18/anaconda3/bin/activate seamless_env

export HF_HOME=~/data_lisik3/kgarci18/hf_cache
mkdir -p "$HF_HOME"
export HF_TOKEN="$(cat ~/.hf_token)"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# --- Paths ---
PARENT_DIR="/home/kgarci18/data_lisik3/kgarci18/ooo/train/social_tokens"
OUT_DIR="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/quick_test"
DINO_CKPT="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/dino_run_20250816_134333/best_dino_vit_base_patch14_dinov2.pt"
TRAIN_IDS="/home/kgarci18/data_lisik3/kgarci18/ooo/train_ids.txt"
TEST_IDS="/home/kgarci18/data_lisik3/kgarci18/ooo/test_ids.txt"

mkdir -p "$OUT_DIR" logs

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUT_DIR"
echo ""

python scripts/llm_finetuning/next_utt_social_ooo_fix.py \
  --parent-dir "$PARENT_DIR" \
  --output-dir "$OUT_DIR" \
  --col-transcript "transcript_json" \
  --visual-mode vectors \
  --caption-nextword \
  --train-frac 0.8 \
  --val-frac 0.1 \
  --epochs 1 \
  --batch-size 2 \
  --lm-name "google/gemma-2-2b-it" \
  --dino-name "vit_base_patch14_dinov2" \
  --dino-checkpoint "$DINO_CKPT" \
  --dino-checkpoint-key "student_backbone" \
  --dino-tune-mode "cls_adapter" \
  --lr-proj 1e-4 \
  --lr-dino 2e-5 \
  --warmup-steps 10 \
  --log-interval 1 \
  --save-every-epochs 1 \
  --dino-local-batch 128 \
  --save-adapter-only \
  --train-id-list "$TRAIN_IDS" \
  --test-id-list "$TEST_IDS" \
  --val-frac-of-train 0.2 \
  --max-locals 20 \
  --train-ablation-mode both \
  --eval-ablations both none \
  --limit-train-steps 5 \
  --limit-val-steps 3

echo ""
echo "========================================="
echo "Quick test complete!"
echo "Check output: ${OUT_DIR}/logs/metrics.csv"
echo "========================================="
