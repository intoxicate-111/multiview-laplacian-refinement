#!/bin/bash

#SBATCH --job-name=eval_s50_img4
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
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
BASELINE="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/B_direct_raw_laplacian"
ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_image_feature_ablation_20k_seed7"
GAUSSIAN="${ROOT}/B_gaussian_feature_2gpu"
HIGH="${ROOT}/C_original_plus_high_frequency_2gpu"
OUTPUT="${ROOT}/analysis"
SCRIPT="${REPO}/scripts/evaluate_sofa50_synthetic_current_image_feature_ablation.py"

for path in \
    "${MANIFEST}" \
    "${SCRIPT}" \
    "${BASELINE}/checkpoint_latest.pt" \
    "${GAUSSIAN}/checkpoint_latest.pt" \
    "${HIGH}/checkpoint_latest.pt"
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${SCRIPT}" \
        --manifest "${MANIFEST}" \
        --baseline-run "${BASELINE}" \
        --gaussian-run "${GAUSSIAN}" \
        --high-frequency-run "${HIGH}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --shard-index "${SLURM_ARRAY_TASK_ID}" \
        --shard-count 4
