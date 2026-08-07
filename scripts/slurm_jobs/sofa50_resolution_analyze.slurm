#!/bin/bash

#SBATCH --job-name=sofa50_res_analysis
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"

MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"

OUTPUT_ROOT="${REPO}/runs/learned_laplacian/sofa50_image_resolution_ablation_50000step"

CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

cd "${REPO}"

echo "============================================================"
echo "Running F0/F1/F2 unified analysis"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Date:     $(date)"
echo "============================================================"

PYTHONPATH=src "${CONDA}" run \
    --no-capture-output \
    -n test \
    python scripts/run_sofa50_image_resolution_ablation.py \
        --manifest "${MANIFEST}" \
        --output-root "${OUTPUT_ROOT}" \
        --analyze \
        --device cuda

echo
echo "============================================================"
echo "Analysis complete"
echo "Report:"
echo "${OUTPUT_ROOT}/analysis/REPORT.md"
echo "Summary:"
echo "${OUTPUT_ROOT}/analysis/summary.json"
echo "============================================================"