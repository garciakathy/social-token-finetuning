#!/usr/bin/env bash
#SBATCH --job-name=nosrp_global
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=lisik3_gpu
#SBATCH --partition=ica100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Keep a safe shell and preserve system/module env (no 'set -u')
set -e -o pipefail
export BASHRCSOURCED=1
datetime=$(date +"%Y%m%d_%H%M%S")

########################
# User configuration
########################
export SOC_OUTPUT_DIR="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/nextutt_runs/run_ddp_08"      # tokenizer dir
export SOC_TOKENIZER_DIR="$SOC_OUTPUT_DIR" # for newer runners
export SOC_LM="google/gemma-2-2b"                                                                          # LM weights (HF id or local dir with weights)
export SOC_PROJECTOR_CKPT="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/nextutt_runs/run_ddp_08/checkpoints/projector_only.pt"
export SOC_OOO_INDEX="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/utils/ooo_index_ordered_for_rsa.csv"
export SOC_SIM_RSM="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/utils/sim_judge_train_rsm.csv"
export SOC_SAVE_FEATS="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/features"
export SOC_RESULTS_CSV="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/gemma_global_nosrp_ooo_rsa_${datetime}.csv"
export SOC_BATCH=8

export SOC_VERBOSE=0
export SOC_LOG_EXAMPLES=3
export SOC_LOG_KEEP_UIDS=20
export SOC_DRYRUN=0

# Runner (your existing filename)
RSA_SCRIPT="ooo_rsa_gemma_unl.py"
# If you switched to the new runner, set:
# RSA_SCRIPT="run_ooo_rsa_aligned_v2.py"

########################
# Environment setup
########################
# Re-source your bashrc to restore modules/paths (keeps srun available)
source ~/.bashrc || true

# Conda activation
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
python --version || true
command -v srun >/dev/null 2>&1 && srun --version || echo "[WARN] srun not in PATH"
nvidia-smi || true

########################
# Optional: verify LM weights if SOC_LM is a local dir
########################
if [ -d "$SOC_LM" ]; then
  shopt -s nullglob
  files=( "$SOC_LM"/pytorch_model.bin "$SOC_LM"/*.safetensors )
  if [ ${#files[@]} -eq 0 ]; then
    echo "[ERROR] No model weight files found in SOC_LM='$SOC_LM'."
    echo "        Put 'pytorch_model.bin' or '*.safetensors' there, or set SOC_LM to a HF model id (e.g., google/gemma-2-2b)."
    exit 2
  fi
fi

mkdir -p "$SOC_SAVE_FEATS"

########################
# Run (keep srun; fallback to python if ever missing)
########################
if command -v srun >/dev/null 2>&1; then
  srun python "${RSA_SCRIPT}" --inject global --pool exclude_soc --srp-dim 0
else
  echo "[WARN] 'srun' not found; running with plain python"
  python "${RSA_SCRIPT}"
fi

echo "[SLURM] Job completed."
