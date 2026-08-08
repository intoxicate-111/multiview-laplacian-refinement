#!/bin/bash
#SBATCH --job-name=cf_compare
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/cf_compare_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/cf_compare_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"
RESOLUTION_ROOT="${REPO}/runs/learned_laplacian/sofa50_image_resolution_ablation_50000step"
C2F2_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_50000step_3seed"
OUTPUT_DIR="${REPO}/runs/learned_laplacian/sofa50_cf_c2f2_comparison"

# Use the analyzer that is already present in the user's repo.
ANALYZER="${REPO}/scripts/analyze_sofa50_cf_c2f2_comparison.py"

cd "${REPO}"

echo "============================================================"
echo "Sofa50 C/F + C2F2 3-seed comparison"
echo "Host:       $(hostname)"
echo "SLURM job:  ${SLURM_JOB_ID}"
echo "Repo:       ${REPO}"
echo "Analyzer:   ${ANALYZER}"
echo "Manifest:   ${MANIFEST}"
echo "Resolution: ${RESOLUTION_ROOT}"
echo "C2F2:       ${C2F2_ROOT}"
echo "Output:     ${OUTPUT_DIR}"
echo "============================================================"

nvidia-smi

test -f "${ANALYZER}"
test -f "${MANIFEST}"
test -d "${RESOLUTION_ROOT}/arms/image_resolution_f0"
test -d "${RESOLUTION_ROOT}/arms/image_resolution_f1"
test -d "${RESOLUTION_ROOT}/arms/image_resolution_f2"
test -d "${C2F2_ROOT}/seed_7"
test -d "${C2F2_ROOT}/seed_17"
test -d "${C2F2_ROOT}/seed_27"

PYTHONPATH="${REPO}/src" "${CONDA}" run \
    --no-capture-output \
    -n test \
    python "${ANALYZER}" \
        --manifest "${MANIFEST}" \
        --resolution-root "${RESOLUTION_ROOT}" \
        --c2f2-root "${C2F2_ROOT}" \
        --output-dir "${OUTPUT_DIR}" \
        --device cuda

echo "============================================================"
echo "Analysis complete"
echo "Report: ${OUTPUT_DIR}/REPORT.md"
echo "Table:  ${OUTPUT_DIR}/main_comparison.csv"
echo "Seeds:  ${OUTPUT_DIR}/c2f2_3seed_stats.csv"
echo "============================================================"
