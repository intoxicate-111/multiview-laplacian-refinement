#!/bin/bash
#SBATCH --job-name=f2k_be_report
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
RUN=${PROJECT}/runs/benchmarks/future2000_frozen_be_vs_baselines_20260904_v1
BASELINES=${PROJECT}/runs/benchmarks/future2000_mixed_vs_old_external_20260831_v2/results
REPORT=${PROJECT}/reports/future2000_frozen_be_vs_baselines_20260904_v1
cd "${PROJECT}"
export PYTHONPATH=${PROJECT}/src:${PROJECT}/scripts
"${PYTHON}" scripts/finalize_future2000_frozen_be_report.py \
    --manifest "${MANIFEST}" \
    --run-dir "${RUN}" \
    --baseline-results-root "${BASELINES}" \
    --report-dir "${REPORT}" \
    --shard-count 8 \
    --expected-samples 1000 \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01
test -s "${REPORT}/FINAL_REPORT.md"
test -s "${REPORT}/summary.json"
