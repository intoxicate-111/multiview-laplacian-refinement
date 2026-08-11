#!/bin/bash

#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=eval_current_ab
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current/manifest.json"
A_RUN="${REPO}/runs/learned_laplacian/sofa50_c2_f2_50000step_3seed/seed_7"
B_RUN="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_c2f2_14view_20k_seed7"
B_CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_synthetic_current_c2f2_14view_20k.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_ab_comparison_seed7"

for path in \
    "${MANIFEST}" \
    "${A_RUN}/best.pt" \
    "${A_RUN}/config.json" \
    "${B_RUN}/best.pt" \
    "${B_CONFIG}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required comparison input: ${path}"
        exit 2
    fi
done
if [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty A/B comparison: ${OUTPUT}"
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
cd "${REPO}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "comparison=existing_frozen_A_vs_new_B"
echo "a_checkpoint=${A_RUN}/best.pt"
echo "b_checkpoint=${B_RUN}/best.pt"
echo "shared_test_manifest=${MANIFEST}"
echo "output=${OUTPUT}"

nvidia-smi

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/evaluate_sofa50_synthetic_current_ab.py \
        --manifest "${MANIFEST}" \
        --a-checkpoint "${A_RUN}/best.pt" \
        --a-config "${A_RUN}/config.json" \
        --b-checkpoint "${B_RUN}/best.pt" \
        --b-config "${B_CONFIG}" \
        --output-dir "${OUTPUT}" \
        --device cuda
