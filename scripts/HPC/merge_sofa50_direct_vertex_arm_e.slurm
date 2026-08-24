#!/bin/bash

#SBATCH --job-name=s50_direct_vertex_e_report
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
AD_REPORT="${REPO}/runs/diagnostics/sofa50_recovery_aware_lambda_extension/final"
OUTPUT="${REPO}/runs/diagnostics/sofa50_direct_vertex_arm_e_extension/final"
export PYTHONPATH="${REPO}/src"
cd "${REPO}"
"${PYTHON}" scripts/evaluate_sofa50_direct_vertex_arm_e.py \
  --ad-report-dir "${AD_REPORT}" --output-dir "${OUTPUT}" --merge-only
