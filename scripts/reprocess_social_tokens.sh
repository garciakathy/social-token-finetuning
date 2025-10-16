#!/bin/bash
# ========================================
# Reprocess Social Tokens with Correct DINO Model
# ========================================
# This script re-runs preprocessing to fix the random noise embeddings issue
# Run this on an interactive GPU node
#
# Usage:
#   1. Request interactive GPU:
#      srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=4:00:00 --pty bash
#   2. Run this script:
#      bash scripts/reprocess_social_tokens.sh

set -e  # Exit on error

echo "========================================"
echo "REPROCESSING SOCIAL TOKENS"
echo "========================================"
echo "This will regenerate embeddings with the correct DINO model"
echo ""

# --- Environment Setup ---
echo "Setting up environment..."
source /data/lisik3/kgarci18/anaconda3/bin/activate seamless_env

export HF_HOME=~/data_lisik3/kgarci18/hf_cache
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0

# --- Paths ---
# IMPORTANT: Verify these paths are correct for your setup!

# Input: Directory containing subdirectories with frames (frame_*.jpg)
FRAMES_ROOT="/home/kgarci18/data_lisik3/kgarci18/ooo/train/frames"

# Input: CSV with columns 'video_name' and 'caption'
# TODO: UPDATE THIS PATH! This CSV should contain video names and their captions
CAPTIONS_CSV="/home/kgarci18/data_lisik3/kgarci18/ooo/train/captions.csv"  # ⚠️ VERIFY THIS PATH

# Output: Where to save the corrected social tokens
# Using a NEW directory to avoid overwriting during testing
OUT_DIR="/home/kgarci18/data_lisik3/kgarci18/ooo/train/social_tokens_fixed"

# DINO checkpoint: Fine-tuned DINO model (from your training)
DINO_CKPT="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/dino_run_20250816_134333/best_dino_vit_base_patch14_dinov2.pt"

# --- Configuration ---
MODEL_NAME="vit_base_patch14_dinov2.lvd142m"
MAX_GLOBAL_FRAMES=120  # Number of frames to sample for global vector
TOP_K_LOCALS=1000      # Max number of local tokens
PAD_FRAMES=2           # +/- frames around each token mapping
BATCH_SIZE=64          # Batch size for embedding

# Optional: Limit number of videos for testing (0 = process all)
LIMIT=0  # Set to 5-10 for quick test, 0 for full dataset

# --- Validation ---
echo "Validating paths..."

if [ ! -d "$FRAMES_ROOT" ]; then
    echo "❌ ERROR: FRAMES_ROOT not found: $FRAMES_ROOT"
    exit 1
fi

if [ ! -f "$CAPTIONS_CSV" ]; then
    echo "⚠️  WARNING: CAPTIONS_CSV not found: $CAPTIONS_CSV"
    echo "   You may need to create this CSV with columns: video_name,caption"
    echo ""
    echo "   Example format:"
    echo "   video_name,caption"
    echo "   -YwZOeyAQC8_15.mp4,a man sits playing video games on his tv..."
    echo ""
    echo "   Or extract from manifest:"
    echo "   python -c \"import pandas as pd; df=pd.read_csv('/path/to/manifest.csv'); df[['clip_id','transcript_json']].rename(columns={'clip_id':'video_name','transcript_json':'caption'}).to_csv('captions.csv', index=False)\""
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to abort... "
fi

if [ ! -f "$DINO_CKPT" ]; then
    echo "❌ ERROR: DINO checkpoint not found: $DINO_CKPT"
    echo "   This is the fine-tuned DINO model that should produce real visual features."
    exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"
mkdir -p logs

echo ""
echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "Frames root:     $FRAMES_ROOT"
echo "Captions CSV:    $CAPTIONS_CSV"
echo "Output dir:      $OUT_DIR"
echo "DINO checkpoint: $DINO_CKPT"
echo "Model name:      $MODEL_NAME"
echo "Device:          cuda (GPU $CUDA_VISIBLE_DEVICES)"
echo "Batch size:      $BATCH_SIZE"
echo "Limit:           $LIMIT (0 = all)"
echo "========================================"
echo ""

# Confirm before proceeding
if [ "$LIMIT" -eq 0 ]; then
    read -p "⚠️  This will process ALL videos. Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
else
    echo "Testing mode: Processing only $LIMIT videos"
fi

echo ""
echo "Starting preprocessing at $(date)"
echo ""

# --- Run Preprocessing ---
python scripts/preprocessing/build_scene_packs_ooo.py \
    --frames_root "$FRAMES_ROOT" \
    --captions_csv "$CAPTIONS_CSV" \
    --out_dir "$OUT_DIR" \
    --dino_ckpt "$DINO_CKPT" \
    --model_name "$MODEL_NAME" \
    --max_global_frames $MAX_GLOBAL_FRAMES \
    --top_k_locals $TOP_K_LOCALS \
    --pad_frames $PAD_FRAMES \
    --batch_size $BATCH_SIZE \
    --device cuda \
    --resume \
    ${LIMIT:+--limit $LIMIT}

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ PREPROCESSING COMPLETE!"
    echo "========================================"
    echo ""
    echo "Output saved to: $OUT_DIR"
    echo "Manifest: $OUT_DIR/social_packs_manifest.csv"
    echo "Logs: $OUT_DIR/social_packs_ooo.log"
    echo ""
    echo "Next steps:"
    echo "  1. Verify embeddings are correct:"
    echo "     python scripts/inspect_npz.py $OUT_DIR"
    echo ""
    echo "  2. Run verification script:"
    echo "     python scripts/verify_random_noise.py $OUT_DIR"
    echo ""
    echo "  3. If embeddings look good, update training to use new path:"
    echo "     --parent-dir $OUT_DIR"
else
    echo "❌ PREPROCESSING FAILED"
    echo "========================================"
    echo ""
    echo "Check logs at: $OUT_DIR/social_packs_ooo.log"
    echo ""
    echo "Common issues:"
    echo "  - DINO checkpoint path incorrect or model failed to load"
    echo "  - Captions CSV format wrong (needs 'video_name' and 'caption' columns)"
    echo "  - Frame directories don't exist or frames missing"
    echo "  - Out of GPU memory (reduce --batch_size)"
fi
echo "========================================"

exit $EXIT_CODE
