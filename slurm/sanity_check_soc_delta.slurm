#!/usr/bin/env bash
#SBATCH --job-name=soc_sanity_check
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log
#SBATCH --account=lisik3_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -e -o pipefail
datetime=$(date +"%Y%m%d_%H%M%S")

########## Configuration ##########
RSA_SCRIPT="../scripts/analysis/sim_judg_rsa_gemma_2.py"
COMPARE_SCRIPT="../scripts/analysis/compare_soc_embeddings.py"

# Model paths
LM_ID="google/gemma-2-2b"
TOKENIZER_DIR="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/nextutt_runs/run_ddp_08"
PROJECTOR_CKPT="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/nextutt_runs/run_ddp_08/checkpoints/projector_only.pt"
OOO_INDEX="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/utils/ooo_index_ordered_for_rsa.csv"
SIM_RSM="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/utils/sim_judge_train_rsm.csv"

# Output directories
BASE_DIR="/home/kgarci18/data_lisik3/kgarci18/seamless/outputs/rsa/sanity_check_${datetime}"
BASELINE_FEATS="${BASE_DIR}/inject_none"
SOC_FEATS="${BASE_DIR}/inject_full"
COMPARISON_OUT="${BASE_DIR}/comparison"

# Runtime
BATCH=8
WORKERS=8
POOL_MODE="all"  # Use same pooling for fair comparison

########## Environment Setup ##########
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

echo "Python: $(which python)"
python --version
nvidia-smi

mkdir -p "$BASELINE_FEATS" "$SOC_FEATS" "$COMPARISON_OUT"

########## Step 1: Extract Baseline Features (no SOC) ##########
echo ""
echo "============================================"
echo "STEP 1: Extracting BASELINE features (inject=none)"
echo "============================================"

srun python -u "$RSA_SCRIPT" \
  --lm "$LM_ID" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --projector-ckpt "$PROJECTOR_CKPT" \
  --index "$OOO_INDEX" \
  --sim-rsm "$SIM_RSM" \
  --save-feats "$BASELINE_FEATS" \
  --results-csv "${BASELINE_FEATS}/rsa_results.csv" \
  --inject none \
  --pool "$POOL_MODE" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --device cuda \
  --no-srp

echo "[DONE] Baseline features saved to: $BASELINE_FEATS"

########## Step 2: Extract SOC Features (full injection) ##########
echo ""
echo "============================================"
echo "STEP 2: Extracting SOC features (inject=full)"
echo "============================================"

srun python -u "$RSA_SCRIPT" \
  --lm "$LM_ID" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --projector-ckpt "$PROJECTOR_CKPT" \
  --index "$OOO_INDEX" \
  --sim-rsm "$SIM_RSM" \
  --save-feats "$SOC_FEATS" \
  --results-csv "${SOC_FEATS}/rsa_results.csv" \
  --inject full \
  --pool "$POOL_MODE" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --device cuda \
  --no-srp

echo "[DONE] SOC features saved to: $SOC_FEATS"

########## Step 3: Compare Embeddings ##########
echo ""
echo "============================================"
echo "STEP 3: Computing embedding deltas"
echo "============================================"

python "$COMPARE_SCRIPT" \
  --baseline-dir "$BASELINE_FEATS" \
  --soc-dir "$SOC_FEATS" \
  --output-dir "$COMPARISON_OUT"

echo ""
echo "============================================"
echo "SANITY CHECK COMPLETE"
echo "============================================"
echo "Results saved to: $COMPARISON_OUT"
echo "  - comparison_stats.csv: Summary statistics"
echo "  - embedding_comparison.png: Visualization plots"
echo "  - embedding_delta.npz: Full delta embeddings"
echo ""

# Print quick summary
if [ -f "${COMPARISON_OUT}/comparison_stats.csv" ]; then
  echo "Quick Summary:"
  python -c "
import pandas as pd
df = pd.read_csv('${COMPARISON_OUT}/comparison_stats.csv')
print(f\"  Mean Cosine Similarity: {df['mean_cosine_sim'].values[0]:.4f}\")
print(f\"  Mean Absolute Delta:    {df['mean_abs_delta'].values[0]:.6f}\")
print(f\"  Frobenius Norm:         {df['frobenius_norm'].values[0]:.2f}\")
print(f\"  Items:                  {df['n_items'].values[0]}\")
print(f\"  Feature Dim:            {df['feature_dim'].values[0]}\")
"
fi
