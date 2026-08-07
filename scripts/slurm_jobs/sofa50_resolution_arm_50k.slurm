#!/bin/bash

#SBATCH --job-name=sofa50_res
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"

MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"

BASE_CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"

OUTPUT_ROOT="${REPO}/runs/learned_laplacian/sofa50_image_resolution_ablation_50000step"

CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

STEPS=50000
DEVICE="cuda"

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 F0|F1|F2"
    exit 1
fi

ARM="$1"

case "${ARM}" in
    F0|F1|F2)
        ;;
    *)
        echo "Invalid arm: ${ARM}"
        exit 1
        ;;
esac

cd "${REPO}"
mkdir -p slurm_logs

echo "============================================================"
echo "Sofa50 image-resolution ablation"
echo "Arm:             ${ARM}"
echo "Optimizer steps: ${STEPS}"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Hostname:        $(hostname)"
echo "Date:            $(date)"
echo "============================================================"

nvidia-smi

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
echo "${ARM} finished successfully"
echo "Date: $(date)"
echo "============================================================"