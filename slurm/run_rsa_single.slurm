#!/usr/bin/env bash
#SBATCH --job-name=rsa_cli_single
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log
#SBATCH --account=lisik3_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -e -o pipefail
datetime=$(date +"%Y%m%d_%H%M%S")

########## Paths & files (edit if needed) ##########
RSA_SCRIPT="/home/kgarci18/code/social-token-finetuning/scripts/analysis/sim_judg_rsa_gemma_2.py"
LM_ID="google/gemma-2-2b-it"
TOKENIZER_DIR="/home/kgarci18/data/seamless/outputs/test_single_gpu"
PROJECTOR_CKPT="/home/kgarci18/data/seamless/outputs/test_single_gpu/checkpoints/projector_only.pt"
OOO_INDEX="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/utils/ooo_index_ordered_for_rsa.csv"
SIM_RSM="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/utils/sim_judge_train_rsm.csv"
SAVE_FEATS_BASE="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/features"

########## Runtime ##########
BATCH=8
WORKERS=8
QUIET=1           # 1 = pass --quiet ; 0 = verbose
DRYRUN=0          # 1 = pass --dryrun

########## Injection/Pooling from CLI args ##########
INJECT="${1:-global}"        # full | global | none   (default mirrors your last run)
POOL="${2:-exclude_soc}"     # all | exclude_soc | only_soc | locals_only | eos

########## Env setup ##########
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
python --version || true
command -v srun >/dev/null 2>&1 && srun --version || echo "[WARN] srun not in PATH"
nvidia-smi || true

########## Verify local LM dir (if using a local path) ##########
if [ -d "$LM_ID" ]; then
  shopt -s nullglob
  files=( "$LM_ID"/pytorch_model.bin "$LM_ID"/*.safetensors )
  if [ ${#files[@]} -eq 0 ]; then
    echo "[ERROR] No model weights found in LM_ID='$LM_ID'."
    exit 2
  fi
fi

########## Per-combo outputs ##########
SAVE_FEATS="${SAVE_FEATS_BASE}/inj_${INJECT}_pool_${POOL}_${datetime}"
RESULTS_CSV="/home/kgarci18/data/seamless/outputs/rsa/results/gemma_${INJECT}_${POOL}_simjudg_rsa_${datetime}.csv"
mkdir -p "$SAVE_FEATS"

QUIET_FLAG=""
[ "$QUIET" -eq 1 ] && QUIET_FLAG="--quiet"
DRYRUN_FLAG=""
[ "$DRYRUN" -eq 1 ] && DRYRUN_FLAG="--dryrun"

echo "[RUN] inject=${INJECT} pool=${POOL}"
CMD=( python -u "$RSA_SCRIPT"
  --lm "$LM_ID"
  --tokenizer-dir "$TOKENIZER_DIR"
  --projector-ckpt "$PROJECTOR_CKPT"
  --index "$OOO_INDEX"
  --sim-rsm "$SIM_RSM"
  --save-feats "$SAVE_FEATS"
  --results-csv "$RESULTS_CSV"
  --inject "$INJECT"
  --pool "$POOL"
  --batch "$BATCH"
  --workers "$WORKERS"
  --device cuda
  --no-srp
  $QUIET_FLAG
  $DRYRUN_FLAG
)

if command -v srun >/dev/null 2>&1; then
  srun --unbuffered "${CMD[@]}"
else
  echo "[WARN] srun not found; running with plain python"
  "${CMD[@]}"
fi

echo "[SLURM] Done: ${RESULTS_CSV}"

