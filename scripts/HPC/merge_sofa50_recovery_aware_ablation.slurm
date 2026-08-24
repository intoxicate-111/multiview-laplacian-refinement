#!/bin/bash

#SBATCH --job-name=s50_sparse_report
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
LAMBDA_SELECTION="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/lambda_validation/lambda_selection.json"
BETA_SELECTION="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/beta_validation/beta_selection.json"
OUTPUT="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/final"
export PYTHONPATH="${REPO}/src"
cd "${REPO}"
"${PYTHON}" scripts/evaluate_sofa50_recovery_aware_ablation.py \
    --lambda-selection "${LAMBDA_SELECTION}" --beta-selection "${BETA_SELECTION}" \
    --output-dir "${OUTPUT}" --merge-only
