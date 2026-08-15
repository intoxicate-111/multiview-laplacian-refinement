#!/bin/bash

#SBATCH --job-name=eval_s50_hf1920
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
MANIFEST_960="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
MANIFEST_1920="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json"
RUN_960="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_image_feature_ablation_20k_seed7/C_original_plus_high_frequency_2gpu"
ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_hf_resolution_1920_20k_seed7"
RUN_1920="${ROOT}/HF_native1920_4gpu"
OUTPUT="${ROOT}/analysis"
SCRIPT="${REPO}/scripts/evaluate_sofa50_hf_resolution_ablation.py"

for path in \
    "${MANIFEST_960}" \
    "${MANIFEST_1920}" \
    "${SCRIPT}" \
    "${RUN_960}/checkpoint_latest.pt" \
    "${RUN_1920}/checkpoint_latest.pt"
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
mkdir -p "${OUTPUT}/shards" "${REPO}/slurm_logs"
cd "${REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${SCRIPT}" \
        --manifest-960 "${MANIFEST_960}" \
        --manifest-1920 "${MANIFEST_1920}" \
        --run-960 "${RUN_960}" \
        --run-1920 "${RUN_1920}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --shard-index "${SLURM_ARRAY_TASK_ID}" \
        --shard-count 4
