#!/bin/bash

#SBATCH --job-name=s50_old1920_valrep
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
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
SPECIALISTS="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/validation_selection/specialists"
LAMBDA_DIR="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/validation_selection/frozen_lambda"
OUTPUT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/validation_selection"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=2
cd "${REPO}"
"${PYTHON}" scripts/report_sofa50_old_domain_native1920_frozen_validation.py \
  --specialist-summary "${SPECIALISTS}/validation_specialist_summary.json" \
  --specialist-per-sample "${SPECIALISTS}/validation_specialist_per_sample.csv" \
  --lambda-selection "${LAMBDA_DIR}/lambda_selection.json" \
  --lambda-per-sample "${LAMBDA_DIR}/lambda_sweep_per_sample.csv" \
  --output-dir "${OUTPUT}"
