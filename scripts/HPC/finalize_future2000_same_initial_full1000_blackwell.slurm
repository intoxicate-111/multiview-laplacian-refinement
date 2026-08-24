#!/bin/bash
#SBATCH --job-name=f2k1000_final
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# The short partition has a one-hour hard limit; this job only merges existing
# shard tables and renders the aggregate report.
#SBATCH --time=0-01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
ROOT=${PROJECT}/runs/benchmarks/future2000_same_initial_full1000_20260822

cd "${PROJECT}"
export PYTHONPATH=${PROJECT}/src
for method in ours nds nds_28v_full nvdiffrec exmesh; do
    "${PYTHON}" scripts/merge_future2000_external_baseline.py \
        --manifest "${MANIFEST}" \
        --output-dir "${ROOT}/results" \
        --method "${method}" \
        --shard-count 8
done
set +e
"${PYTHON}" scripts/generate_future2000_same_initial_subset_report.py \
    --manifest "${MANIFEST}" \
    --results-root "${ROOT}/results" \
    --output-dir "${ROOT}/report" \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01 \
    --methods ours nds nds_28v_full nvdiffrec exmesh
REPORT_STATUS=$?
set -e

# Exit code 2 means the report was generated but an explicitly reported
# contract/completeness check failed (for example, a method produced an invalid
# mesh).  Preserve that scientific outcome in summary.json instead of marking
# the report-generation job itself as failed.
if [[ "${REPORT_STATUS}" -ne 0 && "${REPORT_STATUS}" -ne 2 ]]; then
    exit "${REPORT_STATUS}"
fi
test -s "${ROOT}/report/FINAL_REPORT.md"
test -s "${ROOT}/report/summary.json"
test -s "${ROOT}/report/per_sample.csv"
test -s "${ROOT}/report/paired_ours_vs_external.csv"
