#!/bin/bash
#SBATCH --job-name=f2k8_finalize
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
SELECTION=${PROJECT}/configs/baselines/future2000_same_initial_8sample_selection.json
ROOT=${PROJECT}/runs/benchmarks/future2000_same_initial_8sample_20260822
RESULTS=${ROOT}/results
REPORT=${ROOT}/report

cd "${PROJECT}"
export PYTHONPATH=${PROJECT}/src
for method in ours nds nvdiffrec exmesh; do
    "${PYTHON}" scripts/merge_future2000_external_baseline.py \
        --manifest "${MANIFEST}" \
        --selection "${SELECTION}" \
        --output-dir "${RESULTS}" \
        --method "${method}" \
        --shard-count 8
done
"${PYTHON}" scripts/generate_future2000_same_initial_subset_report.py \
    --selection "${SELECTION}" \
    --results-root "${RESULTS}" \
    --output-dir "${REPORT}" \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01
