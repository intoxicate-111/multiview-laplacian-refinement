#!/bin/bash

#SBATCH --job-name=sofa50_50k
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

# ============================================================
# Paths
# ============================================================

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"

MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"

CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"

OUTPUT="${REPO}/runs/learned_laplacian/sofa50_50mesh_50000steps_hpc"

CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

# ============================================================
# Job information
# ============================================================

cd "${REPO}"

echo "============================================================"
echo "Sofa50 Learned Laplacian Training"
echo "============================================================"
echo "Date:               $(date)"
echo "Hostname:           $(hostname)"
echo "SLURM job ID:       ${SLURM_JOB_ID}"
echo "SLURM job name:     ${SLURM_JOB_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repository:         ${REPO}"
echo "Manifest:           ${MANIFEST}"
echo "Config:             ${CONFIG}"
echo "Output:             ${OUTPUT}"
echo "============================================================"

echo
echo "GPU:"
nvidia-smi

echo
echo "Disk:"
df -h "${REPO}"

echo
echo "Starting training..."
echo

# ============================================================
# Training
# ============================================================

"${CONDA}" run \
    --no-capture-output \
    -n test \
    python scripts/train_multi_mesh_laplacian.py \
        --manifest "${MANIFEST}" \
        --config "${CONFIG}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --epochs 5000 \
        --max-optimizer-steps 50000

EXIT_CODE=$?

echo
echo "============================================================"
echo "Training finished"
echo "Date:      $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================================"

exit "${EXIT_CODE}"