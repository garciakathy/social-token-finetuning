#!/bin/bash
#SBATCH --job-name=ablation_study
#SBATCH --output=logs/ablation_study_%j.out
#SBATCH --error=logs/ablation_study_%j.err
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --ntasks-per-node=4                 # 4 processes (one per GPU)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00                     # Extended time for full ablation study

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
BASE_OUT_DIR="$HOME/data/seamless/outputs/ablation_study"
DINO_CKPT="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/dino_run_20250816_134333/best_dino_vit_base_patch14_dinov2.pt"

LM_NAME="google/gemma-2-2b-it"
DINO_NAME="vit_base_patch14_dinov2"
DINO_TUNE_MODE="cls_adapter"                  # frozen | cls_param_only | cls_adapter | last_n
DINO_LAST_N=0
SEED=42
TRAIN_FRAC=0.8
VAL_FRAC=0.1
VISUAL_MODE="vectors"                          # vectors mode (recommended for OOO)
MAX_LOCALS=50
EPOCHS=4
BATCH_SIZE=8                                   # per-GPU batch size
LR_PROJ=1e-4
LR_DINO=2e-5
WARMUP=888

# Create directories
mkdir -p "$(dirname "$BASE_OUT_DIR")" logs

# Define script path
SCRIPT_PATH="/home/kgarci18/kg/code/social-token-finetuning/scripts/llm_finetuning/next_utt_social_ooo_fix.py"

# ========================================
# COMPREHENSIVE ABLATION STUDY
# ========================================
# This script runs training with all ablation modes and evaluates all combinations

echo "========================================="
echo "Starting Comprehensive Ablation Study"
echo "Date: $(date)"
echo "========================================="

# --- Ablation 1: BOTH tokens (baseline) ---
echo ""
echo "[1/4] Training with BOTH global and local tokens..."
OUT_DIR="${BASE_OUT_DIR}/both_tokens"
mkdir -p "$OUT_DIR"

torchrun --nproc_per_node=4 "$SCRIPT_PATH" \
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
  --train-ablation-mode both \
  --eval-ablations both global_only local_only none

echo "[1/4] BOTH tokens training complete @ $(date)"

# --- Ablation 2: GLOBAL ONLY tokens ---
echo ""
echo "[2/4] Training with GLOBAL ONLY tokens..."
OUT_DIR="${BASE_OUT_DIR}/global_only"
mkdir -p "$OUT_DIR"

torchrun --nproc_per_node=4 "$SCRIPT_PATH" \
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
  --train-ablation-mode global_only \
  --eval-ablations global_only none

echo "[2/4] GLOBAL ONLY training complete @ $(date)"

# --- Ablation 3: LOCAL ONLY tokens ---
echo ""
echo "[3/4] Training with LOCAL ONLY tokens..."
OUT_DIR="${BASE_OUT_DIR}/local_only"
mkdir -p "$OUT_DIR"

torchrun --nproc_per_node=4 "$SCRIPT_PATH" \
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
  --train-ablation-mode local_only \
  --eval-ablations local_only none

echo "[3/4] LOCAL ONLY training complete @ $(date)"

# --- Ablation 4: NO TOKENS (baseline) ---
echo ""
echo "[4/4] Evaluating with NO visual tokens (baseline - no training)..."
echo "Note: This evaluates the frozen LLM without any visual tokens or training."
OUT_DIR="${BASE_OUT_DIR}/no_tokens"
mkdir -p "$OUT_DIR"

# Note: Using --train-ablation-mode none will skip training and only evaluate
# This is correct because there are no trainable parameters with mode "none"
torchrun --nproc_per_node=4 "$SCRIPT_PATH" \
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
  --dino-tune-mode frozen \
  --lr-proj $LR_PROJ \
  --lr-dino 0.0 \
  --warmup-steps $WARMUP \
  --log-interval 25 \
  --save-every-epochs 1 \
  --dino-local-batch 512 \
  --save-adapter-only \
  --train-id-list "/home/kgarci18/data_lisik3/kgarci18/ooo/train_ids.txt" \
  --test-id-list "/home/kgarci18/data_lisik3/kgarci18/ooo/test_ids.txt" \
  --val-frac-of-train 0.2 \
  --max-locals 0 \
  --train-ablation-mode none \
  --eval-ablations none

echo "[4/4] NO TOKENS training complete @ $(date)"

# ========================================
# SUMMARY
# ========================================
echo ""
echo "========================================="
echo "Ablation Study Complete!"
echo "End Date: $(date)"
echo "========================================="
echo ""
echo "Results saved in:"
echo "  - BOTH tokens:   ${BASE_OUT_DIR}/both_tokens/logs/metrics.csv"
echo "  - GLOBAL only:   ${BASE_OUT_DIR}/global_only/logs/metrics.csv"
echo "  - LOCAL only:    ${BASE_OUT_DIR}/local_only/logs/metrics.csv"
echo "  - NO tokens:     ${BASE_OUT_DIR}/no_tokens/logs/metrics.csv"
echo ""
echo "To analyze results, compare perplexity values across:"
echo "  1. BOTH vs GLOBAL_ONLY → contribution of local tokens"
echo "  2. BOTH vs LOCAL_ONLY → contribution of global tokens"
echo "  3. ANY vs NONE → overall benefit of visual grounding"
