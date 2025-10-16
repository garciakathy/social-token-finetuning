#!/bin/bash
#SBATCH --job-name=nextutt_abl
#SBATCH --output=logs/nextutt_abl_%j.out
#SBATCH --error=logs/nextutt_abl_%j.err
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --ntasks-per-node=4                 # 4 processes (one per GPU)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

# ========================================
# SINGLE ABLATION TRAINING
# ========================================
# This script runs training with ONE ablation mode.
# Edit TRAIN_ABLATION and EVAL_ABLATIONS below to control behavior.

# --- Env ---
module purge
# module load cuda/12.1  # if your cluster uses modules
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Paths (EDIT) ---
PARENT_DIR="$HOME/data/seamless/outputs/social_tokens_fixed"   # contains social_packs_manifest.csv
OUT_DIR="$HOME/data/seamless/outputs/nextutt_runs/run_ablation_both"
DINO_CKPT="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/dino_run_20250816_134333/best_dino_vit_base_patch14_dinov2.pt"

LM_NAME="google/gemma-2-2b-it"
DINO_NAME="vit_base_patch14_dinov2"
DINO_TUNE_MODE="cls_adapter"                  # frozen | cls_param_only | cls_adapter | last_n
DINO_LAST_N=0
SEED=42
TRAIN_FRAC=0.8
VAL_FRAC=0.1
VISUAL_MODE="vectors"                          # frames | vectors
MAX_LOCALS=50
EPOCHS=4
BATCH_SIZE=8                                   # per-GPU batch size
LR_PROJ=1e-4
LR_DINO=2e-5
WARMUP=888

# ========================================
# ABLATION CONFIGURATION (EDIT THESE!)
# ========================================
# Options: both | global_only | local_only | none
TRAIN_ABLATION="both"

# Eval ablations: space-separated list
# Options: both global_only local_only none
# Examples:
#   "both none"                      → compare with/without
#   "both global_only local_only"    → evaluate all visual modes
#   "both global_only local_only none" → full evaluation
EVAL_ABLATIONS="both global_only local_only none"

# ========================================

mkdir -p "$(dirname "$OUT_DIR")" logs

echo "========================================="
echo "Starting Ablation Training"
echo "Train mode: $TRAIN_ABLATION"
echo "Eval modes: $EVAL_ABLATIONS"
echo "Date: $(date)"
echo "========================================="

torchrun --nproc_per_node=4 \
  /home/kgarci18/kg/code/social-token-finetuning/scripts/llm_finetuning/next_utt_social_ooo_fix.py \
  --parent-dir "$PARENT_DIR" \
  --output-dir "$OUT_DIR" \
  --col-transcript "transcript_json" \
  --visual-mode vectors \
  --caption-nextword \
  --train-frac $TRAIN_FRAC \
  --val-frac $VAL_FRAC \
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
  --max-locals $MAX_LOCALS \
  --train-ablation-mode $TRAIN_ABLATION \
  --eval-ablations $EVAL_ABLATIONS

echo ""
echo "========================================="
echo "Training Complete!"
echo "End date: $(date)"
echo "========================================="
echo ""
echo "Results: ${OUT_DIR}/logs/metrics.csv"
echo "Checkpoints: ${OUT_DIR}/checkpoints/"
