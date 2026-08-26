#!/bin/bash

#SBATCH --job-name=s50_old1920_lock
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
REPORT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1"
RUN="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_continuous_b_e_lr1e-4_20k_seed7_v1"
FINAL_TEST="${REPORT}/final_test"
OUTPUT="${REPORT}/selection_lock/test_authorization.json"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=2
cd "${REPO}"
"${PYTHON}" scripts/lock_sofa50_old_domain_native1920_final_selection.py \
  --contract-audit "${REPORT}/preflight/contract_audit.json" \
  --specialist-summary "${REPORT}/validation_selection/specialists/validation_specialist_summary.json" \
  --lambda-selection "${REPORT}/validation_selection/frozen_lambda/lambda_selection.json" \
  --step0 "${REPORT}/continuous/preflight/step0_validation.json" \
  --continuous-run "${RUN}" \
  --continuous-validation "${REPORT}/continuous/validation/selected_checkpoint.json" \
  --final-test-dir "${FINAL_TEST}" --output "${OUTPUT}"
