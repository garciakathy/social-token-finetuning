#!/usr/bin/env bash
#SBATCH --job-name=text_baseline
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log
#SBATCH --account=lisik3_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -e -o pipefail

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# Activate environment
source ~/.bashrc || true
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate seamless_env
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate seamless_env
elif [ -f "$HOME/data_lisik3/kgarci18/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/data_lisik3/kgarci18/anaconda3/etc/profile.d/conda.sh"
  conda activate seamless_env
else
  echo "[WARN] Could not find conda; assuming environment is already active."
fi

echo "Python: $(which python)"
python --version

# Run training
python scripts/train_text_only_baseline.py \
  --captions-csv data/captions.csv \
  --model-id google/gemma-2-2b \
  --batch-size 8 \
  --val-batch-size 16 \
  --epochs 20 \
  --lr 5e-5 \
  --warmup-steps 100 \
  --output-dir ~/data/seamless/outputs/text_only_baseline \
  --save-every 5 \
  --num-workers 4 \
  --device cuda

echo "Job finished at $(date)"
