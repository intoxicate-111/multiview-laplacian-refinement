#!/bin/bash
#SBATCH --job-name=f2k_mix_final
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
ROOT=${PROJECT}/runs/benchmarks/future2000_mixed_vs_old_external_20260831_v2
ARCHIVE=${PROJECT}/runs/benchmarks/future2000_same_initial_full1000_20260822/results

cd "${PROJECT}"
export PYTHONPATH=${PROJECT}/src
"${PYTHON}" scripts/merge_future2000_external_baseline.py \
    --manifest "${MANIFEST}" \
    --output-dir "${ROOT}/results" \
    --method ours \
    --shard-count 8
"${PYTHON}" scripts/stage_future2000_frozen_comparators.py \
    --archive-results "${ARCHIVE}" \
    --destination-results "${ROOT}/results" \
    --manifest "${MANIFEST}"

set +e
"${PYTHON}" scripts/generate_future2000_same_initial_subset_report.py \
    --manifest "${MANIFEST}" \
    --results-root "${ROOT}/results" \
    --output-dir "${ROOT}/report" \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01 \
    --methods ours old_structure nds nvdiffrec exmesh
REPORT_STATUS=$?
set -e
if [[ "${REPORT_STATUS}" -ne 0 && "${REPORT_STATUS}" -ne 2 ]]; then
    exit "${REPORT_STATUS}"
fi
test -s "${ROOT}/report/FINAL_REPORT.md"
test -s "${ROOT}/report/summary.json"
test -s "${ROOT}/FROZEN_COMPARATOR_PROVENANCE.json"
