#!/bin/bash
#SBATCH --job-name=c2f2_view_ablation
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/c2f2_view_ablation_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/c2f2_view_ablation_%A_%a.err

set -euo pipefail

# ============================================================
# C2_F2 14 / 28 / 56 view-count ablation
#
# One SLURM array task per view count:
#   task 0 -> 14 views
#   task 1 -> 28 views
#   task 2 -> 56 views
#
# Keep C2_F2, seeds, image resolution, recovery settings, split,
# GT graph target, and expanded graph fixed. Only view count changes.
# ============================================================

# ============================================================
# Paths
# ============================================================

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

SOFA_ROOT="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement"
NESTED_ROOT="${SOFA_ROOT}/multiview_nested_14_28_56_cpu_v3"

# The existing analyzer used by the C2_F2 full evaluation.
ANALYZER="${REPO}/scripts/analyze_sofa50_cf_c2f2_full.py"

# The analyzer currently expects the existing resolution-ablation root.
# This is held fixed across 14/28/56 and is NOT the ablated variable here.
RESOLUTION_ROOT="${REPO}/runs/learned_laplacian/sofa50_image_resolution_ablation_50000step"

# Expected C2_F2 training layout:
#
# ${C2F2_ROOT_BASE}/views_14/seed_7
# ${C2F2_ROOT_BASE}/views_14/seed_17
# ${C2F2_ROOT_BASE}/views_14/seed_27
# ${C2F2_ROOT_BASE}/views_28/seed_7
# ...
# ${C2F2_ROOT_BASE}/views_56/seed_27
#
C2F2_ROOT_BASE="${REPO}/runs/learned_laplacian/sofa50_c2_f2_14_28_56_view_ablation_20000step_3seed"

OUTPUT_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_14_28_56_view_ablation_eval"

# ============================================================
# Select view count from SLURM array index
# ============================================================

VIEW_COUNTS=(14 28 56)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: this script is intended to be submitted with sbatch as a job array."
    exit 2
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#VIEW_COUNTS[@]} )); then
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 2
fi

VIEWS="${VIEW_COUNTS[$SLURM_ARRAY_TASK_ID]}"

GT_MANIFEST="${NESTED_ROOT}/gt_query_views_${VIEWS}_manifest.json"

# IMPORTANT:
# The nested-view preparation script you showed creates the GT-query manifests,
# but not these expanded inference manifests. Keep one expanded manifest per
# view count so the expanded query graph is fixed while the observations follow
# the corresponding 14/28/56 camera subset.
#
# If your actual filenames differ, edit only this line.
EXPANDED_MANIFEST="${NESTED_ROOT}/expanded_inference_views_${VIEWS}_manifest.json"

C2F2_ROOT="${C2F2_ROOT_BASE}/views_${VIEWS}"
OUTPUT_DIR="${OUTPUT_ROOT}/views_${VIEWS}"

# ============================================================
# Setup
# ============================================================

cd "${REPO}"
mkdir -p "${REPO}/slurm_logs"
mkdir -p "${OUTPUT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ============================================================
# Job information
# ============================================================

printf '%s\n' \
"============================================================" \
"Sofa50 C2_F2 14/28/56 view-count ablation" \
"Host:              $(hostname)" \
"SLURM job:         ${SLURM_JOB_ID}" \
"Array task:        ${SLURM_ARRAY_TASK_ID}" \
"Views:             ${VIEWS}" \
"Analyzer:          ${ANALYZER}" \
"GT manifest:       ${GT_MANIFEST}" \
"Expanded manifest: ${EXPANDED_MANIFEST}" \
"Resolution root:   ${RESOLUTION_ROOT}" \
"C2_F2 root:        ${C2F2_ROOT}" \
"Output:            ${OUTPUT_DIR}" \
"============================================================"

nvidia-smi

# ============================================================
# Preflight checks
# ============================================================

if [[ ! -f "${ANALYZER}" ]]; then
    echo "ERROR: analyzer not found:"
    echo "  ${ANALYZER}"
    exit 2
fi

if [[ ! -f "${GT_MANIFEST}" ]]; then
    echo "ERROR: ${VIEWS}-view GT manifest not found:"
    echo "  ${GT_MANIFEST}"
    exit 2
fi

if [[ ! -f "${EXPANDED_MANIFEST}" ]]; then
    echo "ERROR: ${VIEWS}-view expanded inference manifest not found:"
    echo "  ${EXPANDED_MANIFEST}"
    echo
    echo "The nested 14/28/56 preparation script creates GT-query manifests only."
    echo "Generate the expanded inference manifest for this view count first,"
    echo "or change EXPANDED_MANIFEST above to its actual path."
    exit 2
fi

# Existing resolution-ablation arms required by the current analyzer.
for ARM in image_resolution_f0 image_resolution_f1 image_resolution_f2; do
    ARM_DIR="${RESOLUTION_ROOT}/arms/${ARM}"

    if [[ ! -d "${ARM_DIR}" ]]; then
        echo "ERROR: resolution arm not found:"
        echo "  ${ARM_DIR}"
        exit 2
    fi
done

# Three C2_F2 seeds for this view-count arm.
for SEED in 7 17 27; do
    SEED_DIR="${C2F2_ROOT}/seed_${SEED}"

    if [[ ! -d "${SEED_DIR}" ]]; then
        echo "ERROR: C2_F2 ${VIEWS}-view seed directory not found:"
        echo "  ${SEED_DIR}"
        exit 2
    fi

    if [[ ! -f "${SEED_DIR}/launch_config.json" ]]; then
        echo "WARNING: launch_config.json not found:"
        echo "  ${SEED_DIR}/launch_config.json"
    fi
done

echo
echo "C2_F2 ${VIEWS}-view runs:"
for SEED in 7 17 27; do
    echo "  seed_${SEED}: ${C2F2_ROOT}/seed_${SEED}"
done
echo

# ============================================================
# Full prediction / recovery / render evaluation
# ============================================================

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

# ============================================================
# Done
# ============================================================

printf '%s\n' \
"============================================================" \
"C2_F2 ${VIEWS}-view evaluation complete" \
"C2_F2 root: ${C2F2_ROOT}" \
"Report:      ${OUTPUT_DIR}/REPORT.md" \
"Recovery:    ${OUTPUT_DIR}/expanded_comparison.csv" \
"Per mesh:    ${OUTPUT_DIR}/expanded_per_mesh.csv" \
"Renderings:  ${OUTPUT_DIR}/renderings/cross_model_main_confidence" \
"============================================================"
