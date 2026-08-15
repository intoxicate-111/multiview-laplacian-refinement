#!/bin/bash

#SBATCH --job-name=eval_s50_loss4
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
HUBER_RUN="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/B_direct_raw_laplacian"
MSE_RUN="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_loss_ablation_20k_seed7/MSE_raw_3gpu"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_loss_ablation_20k_seed7/analysis"
SCRIPT="${REPO}/scripts/evaluate_sofa50_synthetic_current_28view_loss_ablation.py"

for path in \
    "${MANIFEST}" \
    "${SCRIPT}" \
    "${HUBER_RUN}/checkpoint_latest.pt" \
    "${MSE_RUN}/checkpoint_latest.pt"
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
echo "job_id=${SLURM_JOB_ID} shard=${SLURM_ARRAY_TASK_ID}/4 output=${OUTPUT}"
nvidia-smi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${SCRIPT}" \
        --manifest "${MANIFEST}" \
        --huber-run "${HUBER_RUN}" \
        --mse-run "${MSE_RUN}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --shard-index "${SLURM_ARRAY_TASK_ID}" \
        --shard-count 4
