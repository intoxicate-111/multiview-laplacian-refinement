#!/bin/bash
#SBATCH --job-name=c2f2_1920_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/c2f2_1920_eval_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/c2f2_1920_eval_%j.err

set -euo pipefail


# ============================================================
# Paths
# ============================================================

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

# 1920 Sofa50 manifests
GT_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_1920/gt_query_manifest.json"
EXPANDED_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_1920/expanded_inference_manifest.json"

# Existing F0/F1/F2 resolution-ablation runs.
# analyze_sofa50_cf_c2f2_full.py expects this root as well.
RESOLUTION_ROOT="${REPO}/runs/learned_laplacian/sofa50_image_resolution_ablation_50000step"

# Three completed C2F2 @ 1920 runs:
#
#   seed_7
#   seed_17
#   seed_27
#
C2F2_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_1920_20000step_3seed"

# Keep 1920 evaluation outputs separate from the previous full evaluation.
OUTPUT_DIR="${REPO}/runs/learned_laplacian/sofa50_cf_c2f2_1920_comparison_full"

ANALYZER="${REPO}/scripts/analyze_sofa50_cf_c2f2_full.py"


# ============================================================
# Setup
# ============================================================

cd "${REPO}"
mkdir -p "${REPO}/slurm_logs"
mkdir -p "${OUTPUT_DIR}"

# Helpful for the memory-heavy 1920/F2 inference.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


# ============================================================
# Job information
# ============================================================

printf '%s\n' \
  "============================================================" \
  "Sofa50 C2F2 1920 full prediction/recovery/render evaluation" \
  "Host:              $(hostname)" \
  "SLURM job:         ${SLURM_JOB_ID}" \
  "Analyzer:          ${ANALYZER}" \
  "GT manifest:       ${GT_MANIFEST}" \
  "Expanded manifest: ${EXPANDED_MANIFEST}" \
  "Resolution root:   ${RESOLUTION_ROOT}" \
  "C2F2 1920 root:    ${C2F2_ROOT}" \
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
    echo "ERROR: 1920 GT manifest not found:"
    echo "  ${GT_MANIFEST}"
    exit 2
fi

if [[ ! -f "${EXPANDED_MANIFEST}" ]]; then
    echo "ERROR: 1920 expanded manifest not found:"
    echo "  ${EXPANDED_MANIFEST}"
    exit 2
fi


# Existing resolution-ablation arms required by the analyzer.
for ARM in image_resolution_f0 image_resolution_f1 image_resolution_f2; do
    ARM_DIR="${RESOLUTION_ROOT}/arms/${ARM}"

    if [[ ! -d "${ARM_DIR}" ]]; then
        echo "ERROR: resolution arm not found:"
        echo "  ${ARM_DIR}"
        exit 2
    fi
done


# Three new C2F2 @ 1920 runs.
for SEED in 7 17 27; do
    SEED_DIR="${C2F2_ROOT}/seed_${SEED}"

    if [[ ! -d "${SEED_DIR}" ]]; then
        echo "ERROR: C2F2 1920 seed directory not found:"
        echo "  ${SEED_DIR}"
        exit 2
    fi

    if [[ ! -f "${SEED_DIR}/launch_config.json" ]]; then
        echo "WARNING: launch_config.json not found:"
        echo "  ${SEED_DIR}/launch_config.json"
    fi
done


echo
echo "1920 C2F2 runs:"
for SEED in 7 17 27; do
    echo "  seed_${SEED}: ${C2F2_ROOT}/seed_${SEED}"
done
echo


# ============================================================
# Full analysis
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
  "C2F2 1920 full evaluation complete" \
  "C2F2 root: ${C2F2_ROOT}" \
  "Report:    ${OUTPUT_DIR}/REPORT.md" \
  "Recovery:  ${OUTPUT_DIR}/expanded_comparison.csv" \
  "Per mesh:  ${OUTPUT_DIR}/expanded_per_mesh.csv" \
  "Renderings:${OUTPUT_DIR}/renderings/cross_model_main_confidence" \
  "============================================================"