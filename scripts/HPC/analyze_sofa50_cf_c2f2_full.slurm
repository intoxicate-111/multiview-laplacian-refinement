#!/bin/bash
#SBATCH --job-name=cf_full_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/cf_full_eval_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/cf_full_eval_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

GT_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"
EXPANDED_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/expanded_inference_manifest.json"
RESOLUTION_ROOT="${REPO}/runs/learned_laplacian/sofa50_image_resolution_ablation_50000step"
C2F2_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_50000step_3seed"
OUTPUT_DIR="${REPO}/runs/learned_laplacian/sofa50_cf_c2f2_comparison_full"
ANALYZER="${REPO}/scripts/analyze_sofa50_cf_c2f2_full_v2.py"

cd "${REPO}"

echo "============================================================"
echo "Sofa50 full C/F + C2F2 comparison"
echo "Analyzer: ${ANALYZER}"
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "============================================================"

nvidia-smi

test -f "${ANALYZER}"
test -f "${GT_MANIFEST}"
test -f "${EXPANDED_MANIFEST}"

PYTHONPATH="${REPO}/src" "${CONDA}" run \
    --no-capture-output \
    -n test \
    python "${ANALYZER}" \
        --gt-manifest "${GT_MANIFEST}" \
        --expanded-manifest "${EXPANDED_MANIFEST}" \
        --resolution-root "${RESOLUTION_ROOT}" \
        --c2f2-root "${C2F2_ROOT}" \
        --output-dir "${OUTPUT_DIR}" \
        --device cuda

echo "============================================================"
echo "Analysis complete"
echo "Report: ${OUTPUT_DIR}/REPORT.md"
echo "Expanded: ${OUTPUT_DIR}/expanded_comparison.csv"
echo "Renderings: ${OUTPUT_DIR}/renderings"
echo "============================================================"
