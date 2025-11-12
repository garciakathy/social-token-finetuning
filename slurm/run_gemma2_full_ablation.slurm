#!/bin/bash
#SBATCH --job-name=gemma2_full_abl
#SBATCH --output=logs/gemma2_full_abl_%j.out
#SBATCH --error=logs/gemma2_full_abl_%j.err
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

# ============================================================================
# FULL ABLATION STUDY - Train with Both, Evaluate All Modes
# ============================================================================
# This script trains with both global and local tokens (stable training),
# then evaluates in 4 modes at test time:
#   1. both - Full model with both token types
#   2. local_only - Mask out global token at test time
#   3. global_only - Mask out local tokens at test time
#   4. frozen_baseline - No visual information
#
# This approach avoids the numerical instability of training exclusively
# with local tokens while still providing valid ablation metrics.
# ============================================================================

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

PARENT_DIR="$HOME/data/seamless/outputs/social_tokens_fixed"
OUT_DIR="$HOME/data/seamless/outputs/nextutt_runs/run_gemma2_full_ablation_$(date +%Y%m%d_%H%M%S)"
DINO_CKPT="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/dino_run_20250816_134333/best_dino_vit_base_patch14_dinov2.pt"
LM_NAME="google/gemma-2-2b"
DINO_NAME="vit_base_patch14_dinov2"
DINO_TUNE_MODE="cls_adapter"
EPOCHS=5
BATCH_SIZE=8
LR_PROJ=1e-4
LR_DINO=2e-5
WARMUP=888

mkdir -p "$(dirname "$OUT_DIR")" logs

echo "============================================================================"
echo "FULL ABLATION STUDY: Train with Both, Evaluate All Modes"
echo "============================================================================"
echo "Model: $LM_NAME"
echo "Training Mode: both tokens"
echo "Evaluation Modes: both, local_only, global_only, frozen_baseline"
echo "Starting: $(date)"
echo "============================================================================"

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
  --train-ablation-mode both \
  --eval-ablations both local_only global_only frozen_baseline

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
    echo "1. Check examples_both.json - should have BOTH tokens"
    echo "2. Check examples_local_only.json - should have ONLY <SOC_L>"
    echo "3. Check examples_global_only.json - should have ONLY <SOC_G>"
    echo "4. Check examples_frozen_baseline.json - should have NO social tokens"
    echo "5. Check metrics.csv for all 4 perplexity values"
    echo ""
    echo "Expected results:"
    echo "  both:            ~2-4 PPL (best)"
    echo "  global_only:     ~5-10 PPL (good)"
    echo "  local_only:      ~10-20 PPL (moderate)"
    echo "  frozen_baseline: ~163-207 PPL (baseline)"
else
    echo "FAILED! Check error logs for details."
fi

exit $EXIT_CODE
