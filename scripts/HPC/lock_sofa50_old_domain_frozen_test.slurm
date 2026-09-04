#!/bin/bash

#SBATCH --job-name=s50_old1920_flock
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
REPORT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1"
BENCHMARK="${REPO}/runs/benchmarks/sofa50_same_initial_20260820/benchmark_manifest.json"
OUTPUT="${REPORT}/selection_lock/frozen_test_authorization.json"
FINAL_TEST="${REPORT}/frozen_final_test"

test ! -e "${OUTPUT}"
test ! -e "${FINAL_TEST}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=2
cd "${REPO}"
"${PYTHON}" scripts/lock_sofa50_old_domain_frozen_test.py \
  --contract-audit "${REPORT}/preflight/contract_audit.json" \
  --specialist-summary "${REPORT}/validation_selection/specialists/validation_specialist_summary.json" \
  --lambda-selection "${REPORT}/validation_selection/frozen_lambda/lambda_selection.json" \
  --frozen-validation "${REPORT}/validation_selection/frozen_validation_summary.json" \
  --benchmark-manifest "${BENCHMARK}" --final-test-dir "${FINAL_TEST}" \
  --output "${OUTPUT}"
