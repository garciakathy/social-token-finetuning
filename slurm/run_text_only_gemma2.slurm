#!/bin/bash
#SBATCH --job-name=text_only_gemma2
#SBATCH --output=logs/text_only_gemma2_%j.out
#SBATCH --error=logs/text_only_gemma2_%j.err
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ============================================================================
# TEXT-ONLY BASELINE TRAINING - Gemma-2-2B (Base)
# ============================================================================
# Train Gemma on caption data WITHOUT visual tokens for baseline comparison.
# This provides a honest baseline for measuring social token contribution.
# ============================================================================

module purge
source /data/lisik3/kgarci18/anaconda3/bin/activate seamless_env

export HF_HOME=~/data_lisik3/kgarci18/hf_cache
mkdir -p "$HF_HOME"
export HF_TOKEN="$(cat ~/.hf_token)"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Configuration
MODEL_ID="google/gemma-2-2b"
CAPTIONS_CSV="data/captions.csv"
OUTPUT_DIR="$HOME/data/seamless/outputs/text_only_baseline/gemma2_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=4
VAL_BATCH_SIZE=8
GRADIENT_ACCUM=2  # Effective batch size = 4 * 2 = 8
EPOCHS=10
LR=5e-5
WARMUP=100
MAX_LENGTH=512
VAL_SPLIT=0.1
SAVE_EVERY=2

mkdir -p "$(dirname "$OUTPUT_DIR")" logs

echo "============================================================================"
echo "TEXT-ONLY BASELINE TRAINING - Gemma-2-2B (Base)"
echo "============================================================================"
echo "Model: $MODEL_ID"
echo "Captions: $CAPTIONS_CSV"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE (effective: $(($BATCH_SIZE * $GRADIENT_ACCUM)))"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Starting: $(date)"
echo "============================================================================"

python scripts/llm_finetuning/train_text_only_baseline.py \
  --model-id "$MODEL_ID" \
  --captions-csv "$CAPTIONS_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size $BATCH_SIZE \
  --val-batch-size $VAL_BATCH_SIZE \
  --gradient-accumulation-steps $GRADIENT_ACCUM \
  --epochs $EPOCHS \
  --lr $LR \
  --warmup-steps $WARMUP \
  --max-length $MAX_LENGTH \
  --val-split $VAL_SPLIT \
  --save-every $SAVE_EVERY \
  --mixed-precision \
  --gradient-checkpointing \
  --num-workers 4 \
  --log-interval 25

EXIT_CODE=$?

echo "============================================================================"
echo "Training completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS! Check results at:"
    echo "  Metrics: $OUTPUT_DIR/metrics.csv"
    echo "  Best model: $OUTPUT_DIR/best_model/"
    echo ""
    echo "Expected validation perplexity: ~200-300 (text-only, no visual grounding)"
else
    echo "FAILED! Check error logs for details."
fi

exit $EXIT_CODE
