#!/bin/bash
#SBATCH --job-name=gemma2_local
#SBATCH --output=logs/gemma2_local_%j.out
#SBATCH --error=logs/gemma2_local_%j.err
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

# ============================================================================
# LOCAL-ONLY ABLATION TRAINING - Gemma-2-2B (Base Model)
# ============================================================================
# This script trains with ONLY local social tokens (<SOC_L>), no global token.
# Compares local-only performance against frozen baseline.
#
# Ablation: local_only
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
OUT_DIR="$HOME/data/seamless/outputs/nextutt_runs/run_gemma2_local_only_$(date +%Y%m%d_%H%M%S)"
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
echo "ABLATION TRAINING: LOCAL-ONLY SOCIAL TOKENS"
echo "============================================================================"
echo "Model: $LM_NAME"
echo "Output: $OUT_DIR"
echo "Ablation: Using only <SOC_L> tokens (no global token)"
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
  --train-ablation-mode local_only \
  --eval-ablations local_only frozen_baseline

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
    echo "1. Check examples_local_only.json - should have <SOC_L> but NOT <SOC_G>"
    echo "2. Check examples_frozen_baseline.json - should have NO social tokens"
    echo "3. Check metrics.csv - compare test_ppl_local_only vs test_ppl_frozen_baseline"
    echo "4. Local-only should be worse than 'both' but better than frozen baseline"
else
    echo "FAILED! Check error logs for details."
fi

exit $EXIT_CODE
