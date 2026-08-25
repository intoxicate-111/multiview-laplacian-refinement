#!/bin/bash

#SBATCH --job-name=s50_lambda_objective_report
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
OUTPUT="${REPO}/runs/diagnostics/sofa50_adaptive_lambda_objective_alignment"
REFERENCE="${REPO}/runs/diagnostics/sofa50_adaptive_lambda_oracle/summary.json"
REFERENCE_PER_LAMBDA="${REPO}/runs/diagnostics/sofa50_adaptive_lambda_oracle/per_lambda.csv"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
cd "${REPO}"
"${PYTHON}" scripts/diagnose_sofa50_adaptive_lambda_objective_alignment.py \
  --output-dir "${OUTPUT}" --reference-summary "${REFERENCE}" \
  --reference-per-lambda "${REFERENCE_PER_LAMBDA}" \
  --shard-count 4 --merge-only
