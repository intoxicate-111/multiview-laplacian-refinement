#!/bin/bash
#SBATCH --job-name=f2k_learn_eval
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="${MANIFEST:-/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json}"
LAPLACIAN_RUN="${LAPLACIAN_RUN:-${REPO}/runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k_seed7}"
DISPLACEMENT_RUN="${DISPLACEMENT_RUN:-${REPO}/runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_displacement_20k_seed7}"
OUTPUT="${OUTPUT:-${LAPLACIAN_RUN}/analysis}"

for path in \
    "${MANIFEST}" \
    "${LAPLACIAN_RUN}/run_config.json" \
    "${DISPLACEMENT_RUN}/run_config.json"
do
    test -f "${path}"
done
test -f "${LAPLACIAN_RUN}/checkpoint_best.pt" || test -f "${LAPLACIAN_RUN}/checkpoint_latest.pt"
test -f "${DISPLACEMENT_RUN}/checkpoint_best.pt" || test -f "${DISPLACEMENT_RUN}/checkpoint_latest.pt"

SHARD="${OUTPUT}/shards/per_sample_shard_$(printf '%03d' "${SLURM_ARRAY_TASK_ID}").csv"
if [[ -f "${SHARD}" ]]; then
    echo "Refusing to overwrite completed shard: ${SHARD}" >&2
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${OUTPUT}/shards"
cd "${REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "job_id=${SLURM_JOB_ID} shard=${SLURM_ARRAY_TASK_ID}/3"
echo "manifest=${MANIFEST}"
echo "laplacian_run=${LAPLACIAN_RUN}"
echo "displacement_run=${DISPLACEMENT_RUN}"
echo "output=${OUTPUT}"
nvidia-smi

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/evaluate_future2000_laplacian_vs_displacement.py \
        --manifest "${MANIFEST}" \
        --laplacian-run "${LAPLACIAN_RUN}" \
        --displacement-run "${DISPLACEMENT_RUN}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --shard-index "${SLURM_ARRAY_TASK_ID}" \
        --shard-count 3 \
        --surface-samples 3000 \
        --metric-seed 7 \
        --fscore-threshold 0.01
