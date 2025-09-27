#!/bin/bash -l
#SBATCH --job-name=dino_dali_ddp
#SBATCH --partition=ica100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- activate the SAME env you preinstalled torch + DALI into ---
source ~/data_lisik3/kgarci18/anaconda3/etc/profile.d/conda.sh
conda activate seamless_env
echo "Python: $(python -V) @ $(which python)"

# env knobs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# quick checks (fail fast if packages missing)
python - <<'PY'
import importlib, sys
for m in ("torch", "nvidia.dali"):
    try:
        importlib.import_module(m)
        print(f"[ok] {m}")
    except Exception as e:
        print(f"[missing] {m}: {e}", file=sys.stderr); sys.exit(1)
PY
# --- your paths ---
MANIFEST=~/data_lisik3/kgarci18/seamless/full/preprocess/naturalistic/train/manifest.csv
OUTDIR=~/data_lisik3/kgarci18/seamless/outputs/dino_run_$(date +%Y%m%d_%H%M%S)

# --- launch DDP with THIS python (no srun, no torchrun from base) ---
$(which python) -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  train_dino_dali_ddp_improved.py \
    --manifest "$MANIFEST" \
    --out_dir "$OUTDIR" \
    --epochs 100 \
    --patience 10 \
    --model_name vit_base_patch14_dinov2 \
    --global_size 518 \
    --local_size 518 \
    --global_area 0.40 1.00 \
    --local_area 0.05 0.25 \
    --global_crops 2 \
    --local_crops 8 \
    --accum_steps 2 \
    --num_workers 8 \
    --teacher_t_warmup 0.04 \
    --teacher_t_end 0.07 \
    --teacher_warmup_epochs 30 \
    --teacher_momentum_start 0.996 \
    --teacher_momentum_end 0.9995 \
    --amp
