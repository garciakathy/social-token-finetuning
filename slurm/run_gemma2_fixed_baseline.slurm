#!/bin/bash
#SBATCH --job-name=gemma2_fixed
#SBATCH --output=logs/gemma2_fixed_%j.out
#SBATCH --error=logs/gemma2_fixed_%j.err
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

# ============================================================================
# FIXED BASELINE TRAINING - Gemma-2-2B (Base Model)
# ============================================================================
# This script trains with the CORRECTED baseline that removes social tokens
# entirely (instead of keeping meaningless token text).
#
# Fix date: 2025-11-09
# See: BASELINE_FIX_SUMMARY.md for details
# ============================================================================

# --- Environment Setup ---
module purge
source /data/lisik3/kgarci18/anaconda3/bin/activate seamless_env

export HF_HOME=~/data_lisik3/kgarci18/hf_cache
mkdir -p "$HF_HOME"
export HF_TOKEN="$(cat ~/.hf_token)"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export NCCL_P2P_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --- Paths ---
PARENT_DIR="$HOME/data/seamless/outputs/social_tokens_fixed"
OUT_DIR="$HOME/data/seamless/outputs/nextutt_runs/run_gemma2_fixed_baseline_$(date +%Y%m%d_%H%M%S)"
DINO_CKPT="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/dino_run_20250816_134333/best_dino_vit_base_patch14_dinov2.pt"

# --- Model Settings ---
LM_NAME="google/gemma-2-2b"          # Base model (NOT instruction-tuned)
DINO_NAME="vit_base_patch14_dinov2"
DINO_TUNE_MODE="cls_adapter"
SEED=42
EPOCHS=5
BATCH_SIZE=8
LR_PROJ=1e-4
LR_DINO=2e-5
WARMUP=888

# --- Create directories ---
mkdir -p "$(dirname "$OUT_DIR")" logs

echo "============================================================================"
echo "TRAINING WITH FIXED BASELINE (Social tokens removed in 'none' mode)"
echo "============================================================================"
echo "Model: $LM_NAME"
echo "Output: $OUT_DIR"
echo "Starting: $(date)"
echo "============================================================================"

# --- Run Training ---
torchrun --nproc_per_node=4 scripts/llm_finetuning/next_utt_social_ooo_fix.py \
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
  --log-interval 25 \
  --save-every-epochs 1 \
  --dino-local-batch 512 \
  --save-adapter-only \
  --train-id-list "/home/kgarci18/data_lisik3/kgarci18/ooo/train_ids.txt" \
  --test-id-list "/home/kgarci18/data_lisik3/kgarci18/ooo/test_ids.txt" \
  --val-frac-of-train 0.2 \
  --eval-ablations both none

EXIT_CODE=$?

echo "============================================================================"
echo "Training completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Output directory: $OUT_DIR"
echo "============================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS! Check results at: $OUT_DIR/logs/"
    echo ""
    echo "Verification checklist:"
    echo "1. Check examples_none.json - should NOT have <SOC_G> or <SOC_L> in prompts"
    echo "2. Check metrics.csv - test_ppl_no_vis should be reasonable (5-50, not 2000+)"
    echo "3. Compare with old results in data/results/ppl/gemma2/"
else
    echo "FAILED! Check error logs for details."
fi

exit $EXIT_CODE
