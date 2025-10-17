#!/usr/bin/env bash
#SBATCH --job-name=rsa_cli_array
#SBATCH --output=%x-%A_%a.log
#SBATCH --error=%x-%A_%a.log
#SBATCH --account=lisik3_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-14   # 3 inject * 5 pool = 15 combos

set -e -o pipefail
datetime=$(date +"%Y%m%d_%H%M%S")

########## Combos ##########
injects=(full global none)
pools=(all exclude_soc only_soc locals_only eos)
Ni=${#injects[@]}
Np=${#pools[@]}
i=$(( SLURM_ARRAY_TASK_ID / Np ))
j=$(( SLURM_ARRAY_TASK_ID % Np ))
INJECT="${injects[$i]}"
POOL="${pools[$j]}"

########## Paths & files ##########
RSA_SCRIPT="../scripts/analysis/sim_judg_rsa_gemma_2.py"
LM_ID="google/gemma-2-2b"
TOKENIZER_DIR="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/nextutt_runs/run_ddp_08"
PROJECTOR_CKPT="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/nextutt_runs/run_ddp_08/checkpoints/projector_only.pt"
OOO_INDEX="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/utils/ooo_index_ordered_for_rsa.csv"
SIM_RSM="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/utils/sim_judge_train_rsm.csv"
SAVE_FEATS_BASE="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/features"

########## Runtime ##########
BATCH=8
WORKERS=8
QUIET=0
DRYRUN=1

########## Env ##########
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
fi

nvidia-smi || true

if [ -d "$LM_ID" ]; then
  shopt -s nullglob
  files=( "$LM_ID"/pytorch_model.bin "$LM_ID"/*.safetensors )
  if [ ${#files[@]} -eq 0 ]; then
    echo "[ERROR] No model weights found in LM_ID='$LM_ID'."
    exit 2
  fi
fi

SAVE_FEATS="${SAVE_FEATS_BASE}/inj_${INJECT}_pool_${POOL}_${datetime}"
RESULTS_PATH="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/${datetime}"
RESULTS_CSV="${RESULTS_PATH}/gemma_${INJECT}_${POOL}_simjudg_rsa_${datetime}.csv"
mkdir -p "$SAVE_FEATS"
mkdir -p "$RESULTS_PATH"

QUIET_FLAG=""
[ "$QUIET" -eq 1 ] && QUIET_FLAG="--quiet"
DRYRUN_FLAG=""
[ "$DRYRUN" -eq 1 ] && DRYRUN_FLAG="--dryrun"

echo "[ARRAY ${SLURM_ARRAY_TASK_ID}] inject=${INJECT} pool=${POOL}"
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
  --debug-injection 
  --debug-n 2 
  --log-level DEBUG
  --decode-proj-topk 5 
  --decode-proj-show 3 
  --decode-annotated 
  --decode-exclude-soc
  $QUIET_FLAG
  $DRYRUN_FLAG
)

srun --unbuffered "${CMD[@]}"
echo "[ARRAY ${SLURM_ARRAY_TASK_ID}] Done: ${RESULTS_CSV}"

