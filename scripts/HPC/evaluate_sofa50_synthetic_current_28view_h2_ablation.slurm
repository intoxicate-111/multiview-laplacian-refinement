#!/bin/bash

#SBATCH --job-name=eval_cur28_h2
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
TRAIN_ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7"
A_RUN="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7/A_no_jitter"
B_RUN="${TRAIN_ROOT}/B_direct_raw_laplacian"
C_RUN="${TRAIN_ROOT}/C_normalized_output_raw_loss"
OUTPUT="${TRAIN_ROOT}/analysis"

for run in "${A_RUN}" "${B_RUN}" "${C_RUN}"; do
    test -f "${run}/checkpoint_latest.pt"
    test -f "${run}/metrics.json"
done
if [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty analysis: ${OUTPUT}"
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
cd "${REPO}"
echo "job_id=${SLURM_JOB_ID} manifest=${MANIFEST} output=${OUTPUT}"
nvidia-smi

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/evaluate_sofa50_synthetic_current_28view_h2_ablation.py \
        --manifest "${MANIFEST}" \
        --arm-a-run "${A_RUN}" \
        --arm-b-run "${B_RUN}" \
        --arm-c-run "${C_RUN}" \
        --output-dir "${OUTPUT}" \
        --device cuda
