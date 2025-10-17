#!/bin/bash
# ========================================
# Single GPU Interactive Training
# ========================================
# Run this script on an interactive node with GPU access
# Usage: bash scripts/run_single_gpu.sh

set -e  # Exit on error

# --- Environment Setup ---
echo "Setting up environment..."
source /data/lisik3/kgarci18/anaconda3/bin/activate seamless_env

export HF_HOME=~/data_lisik3/kgarci18/hf_cache
mkdir -p "$HF_HOME"
export HF_TOKEN="$(cat ~/.hf_token)"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0

# --- Paths ---
PARENT_DIR="$HOME/data/seamless/outputs/social_tokens_fixed"
OUT_DIR="$HOME/data/seamless/outputs/test_single_gpu"
DINO_CKPT="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/dino_run_20250816_134333/best_dino_vit_base_patch14_dinov2.pt"
TRAIN_IDS="/home/kgarci18/data_lisik3/kgarci18/ooo/train_ids.txt"
TEST_IDS="/home/kgarci18/data_lisik3/kgarci18/ooo/test_ids.txt"

# --- Training Configuration ---
LM_NAME="google/gemma-2-2b-it"
DINO_NAME="vit_base_patch14_dinov2"
DINO_TUNE_MODE="cls_adapter"
EPOCHS=2                    # Fewer epochs for testing
BATCH_SIZE=4                # Smaller batch for single GPU
MAX_LOCALS=50
LR_PROJ=1e-4
LR_DINO=2e-5
WARMUP=100

# --- Ablation Configuration ---
TRAIN_ABLATION="both"                              # both | global_only | local_only | none (NOTE: "none" skips training)
EVAL_ABLATIONS="both none"                         # Space-separated list

# ========================================

mkdir -p "$OUT_DIR" logs

echo "========================================="
echo "Single GPU Training Configuration"
echo "========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUT_DIR"
echo "Train mode: $TRAIN_ABLATION"
echo "Eval modes: $EVAL_ABLATIONS"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "========================================="
echo ""

# Run training (single process, no DDP)
python scripts/llm_finetuning/next_utt_social_ooo_fix.py \
  --parent-dir "$PARENT_DIR" \
  --output-dir "$OUT_DIR" \
  --col-transcript "caption" \
  --visual-mode vectors \
  --caption-nextword \
  --train-frac 0.8 \
  --val-frac 0.1 \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lm-name "$LM_NAME" \
  --dino-name "$DINO_NAME" \
  --dino-checkpoint "$DINO_CKPT" \
  --dino-checkpoint-key "student_backbone" \
  --dino-tune-mode $DINO_TUNE_MODE \
  --lr-proj $LR_PROJ \
  --lr-dino $LR_DINO \
  --warmup-steps $WARMUP \
  --log-interval 10 \
  --save-every-epochs 1 \
  --dino-local-batch 256 \
  --save-adapter-only \
  --train-id-list "$TRAIN_IDS" \
  --test-id-list "$TEST_IDS" \
  --val-frac-of-train 0.2 \
  --max-locals $MAX_LOCALS \
  --train-ablation-mode $TRAIN_ABLATION \
  --eval-ablations $EVAL_ABLATIONS \
  --limit-train-steps 50 \
  --limit-val-steps 20

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo "Results: ${OUT_DIR}/logs/metrics.csv"
echo "Checkpoints: ${OUT_DIR}/checkpoints/"
