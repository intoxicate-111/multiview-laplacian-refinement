#!/bin/bash

#SBATCH --job-name=eval_cur28_h2_3g
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
TRAIN_ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7"
A_RUN="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7/A_no_jitter"
B_RUN="${TRAIN_ROOT}/B_direct_raw_laplacian"
C_RUN="${TRAIN_ROOT}/C_normalized_output_raw_loss"
OUTPUT="${TRAIN_ROOT}/analysis"
SCRIPT="${REPO}/scripts/evaluate_sofa50_synthetic_current_28view_h2_ablation.py"

for path in \
    "${MANIFEST}" \
    "${SCRIPT}" \
    "${A_RUN}/checkpoint_latest.pt" \
    "${B_RUN}/checkpoint_latest.pt" \
    "${C_RUN}/checkpoint_latest.pt"
do
    test -f "${path}"
done
if [[ -f "${OUTPUT}/REPORT.md" ]]; then
    echo "Refusing to overwrite completed analysis: ${OUTPUT}" >&2
    exit 2
fi
if [[ -e "${OUTPUT}/shards/shard_${SLURM_ARRAY_TASK_ID}.json" ]]; then
    echo "Refusing to overwrite completed shard ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${OUTPUT}/shards"
cd "${REPO}"

echo "job_id=${SLURM_JOB_ID} shard=${SLURM_ARRAY_TASK_ID}/3 output=${OUTPUT}"
nvidia-smi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${SCRIPT}" \
        --manifest "${MANIFEST}" \
        --arm-a-run "${A_RUN}" \
        --arm-b-run "${B_RUN}" \
        --arm-c-run "${C_RUN}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --shard-index "${SLURM_ARRAY_TASK_ID}" \
        --shard-count 3
