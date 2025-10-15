#!/bin/bash
#SBATCH --job-name=nextutt_ddp
#SBATCH --output=logs/nextutt_ddp_%j.out
#SBATCH --error=logs/nextutt_ddp_%j.err
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --ntasks-per-node=4                 # 4 processes (one per GPU)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

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


# --- Paths (EDIT) ---
PARENT_DIR="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/social_tokens/train"   # contains scene_packs_manifest_recovered.csv
OUT_DIR="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/nextutt_runs/run_ddp_08"
DINO_CKPT="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/dino_run_20250816_134333/best_dino_vit_base_patch14_dinov2.pt"

LM_NAME="google/gemma-2-2b-it"
DINO_NAME="vit_base_patch14_dinov2"           # adjust if your ckpt is for another ViT
DINO_TUNE_MODE="cls_adapter"                  # frozen | cls_param_only | cls_adapter | last_n
DINO_LAST_N=0
SEED=42
TRAIN_FRAC=0.8
VAL_FRAC=0.1
CONTEXT_TURNS=1
VISUAL_MODE="vectors"                          # frames | vectors
MAX_LOCALS=50
EPOCHS=4
BATCH_SIZE=8                                  # per-GPU batch size
LR_PROJ=1e-4
LR_DINO=2e-5
WARMUP=888

mkdir -p "$(dirname "$OUT_DIR")" logs

echo "Starting torchrun @ $(date)"
torchrun --nproc_per_node=4 next_utt_social_final.py \
  --parent-dir "$PARENT_DIR" \
  --output-dir "$OUT_DIR" \
  --col-transcript "transcript_json" \
  --col-frames "frames_dir" \
  --seed $SEED \
  --train-frac $TRAIN_FRAC \
  --val-frac $VAL_FRAC \
  --context-turns $CONTEXT_TURNS \
  --fallback-split fixed_words --split-k 12 \
  --visual-mode $VISUAL_MODE \
  --max-locals $MAX_LOCALS \
  --lm-name "$LM_NAME" \
  --dino-name "$DINO_NAME" \
  --dino-checkpoint "$DINO_CKPT" \
  --dino-checkpoint-key "student_backbone" \
  --dino-tune-mode $DINO_TUNE_MODE \
  --dino-last-n $DINO_LAST_N \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr-proj $LR_PROJ \
  --lr-dino $LR_DINO \
  --warmup-steps $WARMUP \
  --log-interval 25 \
  --save-every-epochs 1 \
  --dino-local-batch 512 \
  --save-adapter-only

echo "Done @ $(date)"

