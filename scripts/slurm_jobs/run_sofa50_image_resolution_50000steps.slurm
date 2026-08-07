#!/bin/bash

#SBATCH --job-name=sofa50_res50k
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

# ============================================================
# Paths
# ============================================================

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"

MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"

BASE_CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"

OUTPUT_ROOT="${REPO}/runs/learned_laplacian/sofa50_image_resolution_ablation_50000step"

CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

STEPS=50000
DEVICE="cuda"


# ============================================================
# Job information
# ============================================================

cd "${REPO}"

mkdir -p "${REPO}/slurm_logs"

echo "============================================================"
echo "Sofa50 Image Resolution Ablation"
echo "============================================================"
echo "Date:               $(date)"
echo "Hostname:           $(hostname)"
echo "SLURM job ID:       ${SLURM_JOB_ID}"
echo "SLURM job name:     ${SLURM_JOB_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repository:         ${REPO}"
echo "Manifest:           ${MANIFEST}"
echo "Base config:        ${BASE_CONFIG}"
echo "Output root:        ${OUTPUT_ROOT}"
echo "Optimizer steps:    ${STEPS}"
echo "Arms:               F0 F1 F2"
echo "============================================================"

echo
echo "GPU:"
nvidia-smi
echo


# ============================================================
# Train F0 / F1 / F2 sequentially
# ============================================================

for ARM in F0 F1 F2; do

    echo
    echo "============================================================"
    echo "Starting ${ARM}"
    echo "Optimizer steps: ${STEPS}"
    echo "Date: $(date)"
    echo "============================================================"

    PYTHONPATH=src "${CONDA}" run \
        --no-capture-output \
        -n test \
        python scripts/run_sofa50_image_resolution_ablation.py \
            --manifest "${MANIFEST}" \
            --base-config "${BASE_CONFIG}" \
            --output-root "${OUTPUT_ROOT}" \
            --arm "${ARM}" \
            --max-optimizer-steps "${STEPS}" \
            --device "${DEVICE}"

    echo
    echo "============================================================"
    echo "Finished ${ARM}"
    echo "Date: $(date)"
    echo "============================================================"

done


# ============================================================
# Final analysis
# ============================================================

echo
echo "============================================================"
echo "Running final F0/F1/F2 analysis"
echo "Date: $(date)"
echo "============================================================"

PYTHONPATH=src "${CONDA}" run \
    --no-capture-output \
    -n test \
    python scripts/run_sofa50_image_resolution_ablation.py \
        --manifest "${MANIFEST}" \
        --output-root "${OUTPUT_ROOT}" \
        --analyze \
        --device "${DEVICE}"


# ============================================================
# Done
# ============================================================

echo
echo "============================================================"
echo "All experiments finished"
echo "Date: $(date)"
echo
echo "Results:"
echo "  ${OUTPUT_ROOT}/analysis/summary.json"
echo "  ${OUTPUT_ROOT}/analysis/REPORT.md"
echo "============================================================"