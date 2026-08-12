#!/bin/bash

#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=eval_current_50k
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
OLD_COMPARISON="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_ab_comparison_seed7/comparison.json"
CURRENT50_RUN="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_c2f2_14view_50k_resume_from_20k_seed7"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_50k_downstream_evaluation_seed7"

for path in \
    "${OLD_COMPARISON}" \
    "${CURRENT50_RUN}/best.pt" \
    "${CURRENT50_RUN}/config.json" \
    "${CURRENT50_RUN}/metrics.json" \
    "${CURRENT50_RUN}/training_history.json"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required evaluation artifact: ${path}"
        exit 2
    fi
done
if [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty evaluation output: ${OUTPUT}"
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
cd "${REPO}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "old_comparison=${OLD_COMPARISON}"
echo "current50_run=${CURRENT50_RUN}"
echo "output=${OUTPUT}"

nvidia-smi

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/evaluate_sofa50_synthetic_current_50k_downstream.py \
        --old-comparison "${OLD_COMPARISON}" \
        --current50-run "${CURRENT50_RUN}" \
        --output-dir "${OUTPUT}" \
        --device cuda
