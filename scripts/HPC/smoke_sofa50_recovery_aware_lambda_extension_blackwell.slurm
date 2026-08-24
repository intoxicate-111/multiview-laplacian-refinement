#!/bin/bash

#SBATCH --job-name=s50_sparse_lambda_smoke
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
OUTPUT="${REPO}/runs/diagnostics/sofa50_recovery_aware_lambda_extension/smoke"
export PYTHONPATH="${REPO}/src"
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
failures=0
for lambda_value in 1e-3 1e-4; do
  for sample_index in 0 49; do
    lambda_token="${lambda_value//-/m}"
    output_file="${OUTPUT}/float64_lambda_${lambda_token}_tol_1em4_sample_${sample_index}.json"
    rm -f "${output_file}"
    status=0
    "${PYTHON}" scripts/audit_sofa50_differentiable_sparse_recovery.py \
      --manifest "${MANIFEST}" --output "${output_file}" \
      --split validation --sample-index "${sample_index}" \
      --lambda-value "${lambda_value}" --maximum-iterations 2048 \
      --tolerance 1e-4 --pcg-dtype float64 --device cuda || status=$?
    if test -f "${output_file}" && "${PYTHON}" -c \
      'import json,sys; x=json.load(open(sys.argv[1])); assert x["pcg"]["converged"]; assert x["gradient"]["all_finite"] and x["gradient"]["norm"] > 0; assert x["standalone_lsmr"]["all_converged"]' \
      "${output_file}"; then
      echo "AUDIT_SOLVE_PASS dtype=float64 lambda=${lambda_value} tolerance=1e-4 sample=${sample_index} script_status=${status}"
    else
      echo "AUDIT_FAIL dtype=float64 lambda=${lambda_value} tolerance=1e-4 sample=${sample_index}"
      failures=$((failures + 1))
    fi
  done
done
test "${failures}" -eq 0
