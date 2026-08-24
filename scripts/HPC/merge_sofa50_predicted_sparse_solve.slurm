#!/bin/bash

#SBATCH --job-name=s50_pred_sparse_merge
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
ROOT=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_seed7/diagnostics/predicted_sparse_solve
cd "${REPO}"
export PYTHONPATH=${REPO}/src

"${PYTHON}" scripts/diagnose_sofa50_predicted_sparse_solve.py \
    --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1/manifest.json \
    --dataset-arm v1_legacy_smoothing --output-dir "${ROOT}/v1_legacy_smoothing" \
    --shard-count 4 --merge-only
"${PYTHON}" scripts/diagnose_sofa50_predicted_sparse_solve.py \
    --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
    --dataset-arm v2_strong_smoothing --output-dir "${ROOT}/v2_strong_smoothing" \
    --shard-count 4 --merge-only
"${PYTHON}" scripts/generate_sofa50_predicted_sparse_solve_report.py \
    --v1-summary "${ROOT}/v1_legacy_smoothing/summary.json" \
    --v2-summary "${ROOT}/v2_strong_smoothing/summary.json" \
    --output "${ROOT}/FINAL_REPORT.md"
