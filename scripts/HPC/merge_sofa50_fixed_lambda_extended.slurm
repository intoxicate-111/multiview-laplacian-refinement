#!/bin/bash

#SBATCH --job-name=s50_fixed_lambda_report
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
OUTPUT="${REPO}/runs/diagnostics/sofa50_recovery_aware_fixed_lambda_extended/final"
BASE="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/final/shards"
EXT="${REPO}/runs/diagnostics/sofa50_recovery_aware_lambda_extension/final/shards"
LAMBDA_SELECTION="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/lambda_validation/lambda_selection.json"
BETA_SELECTION="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/beta_validation/beta_selection.json"
mkdir -p "${OUTPUT}/shards"
cp "${BASE}/B_lap_plus_refine.json" "${OUTPUT}/shards/B_lap_plus_refine.json"
cp "${BASE}/B_lap_plus_refine_prediction_arrays.npz" \
  "${OUTPUT}/shards/B_lap_plus_refine_prediction_arrays.npz"
for arm in C_lap_plus_refine_lambda1e-3 D_lap_plus_refine_lambda1e-4; do
  cp "${EXT}/${arm}.json" "${OUTPUT}/shards/${arm}.json"
  cp "${EXT}/${arm}_prediction_arrays.npz" \
    "${OUTPUT}/shards/${arm}_prediction_arrays.npz"
done
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
cd "${REPO}"
"${PYTHON}" scripts/evaluate_sofa50_recovery_aware_ablation.py \
  --lambda-selection "${LAMBDA_SELECTION}" --beta-selection "${BETA_SELECTION}" \
  --output-dir "${OUTPUT}" --merge-only --include-fixed-lambda-sweep
