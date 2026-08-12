#!/bin/bash

#SBATCH --job-name=cur28_h2abc
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7"

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
    ARM="B_direct_raw_laplacian"
    CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_synthetic_current_28view_direct_raw_20k.json"
else
    ARM="C_normalized_output_raw_loss"
    CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_synthetic_current_28view_normalized_output_raw_loss_20k.json"
fi
OUTPUT="${ROOT}/${ARM}"

test -f "${MANIFEST}"
test -f "${CONFIG}"
if [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty run: ${OUTPUT}"
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
cd "${REPO}"
echo "job_id=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} arm=${ARM}"
echo "manifest=${MANIFEST} config=${CONFIG} output=${OUTPUT} steps=20000"
nvidia-smi

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/train_multi_mesh_laplacian.py \
        --manifest "${MANIFEST}" \
        --config "${CONFIG}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --max-optimizer-steps 20000
