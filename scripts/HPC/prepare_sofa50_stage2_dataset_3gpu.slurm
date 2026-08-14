#!/bin/bash

#SBATCH --job-name=prep_B_s2
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
B_RUN="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/B_direct_raw_laplacian"
EXPERIMENT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_b_stage2_adaptation_20k_seed7"
OUTPUT="${EXPERIMENT}/dataset"
SCRIPT="${REPO}/scripts/prepare_sofa50_synthetic_current_stage2_adaptation.py"

for path in "${MANIFEST}" "${B_RUN}/checkpoint_latest.pt" "${B_RUN}/run_config.json" "${SCRIPT}"; do
    test -f "${path}"
done
if [[ -e "${OUTPUT}/shards/dataset_shard_${SLURM_ARRAY_TASK_ID}.json" ]]; then
    echo "Refusing to overwrite completed dataset shard ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi
mkdir -p "${REPO}/slurm_logs" "${OUTPUT}/shards"
cd "${REPO}"
nvidia-smi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYOPENGL_PLATFORM="egl"
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${SCRIPT}" \
        --manifest "${MANIFEST}" \
        --b-run-dir "${B_RUN}" \
        --output-dir "${OUTPUT}" \
        --shard-index "${SLURM_ARRAY_TASK_ID}" \
        --shard-count 3 \
        --visibility-size 960 \
        --device cuda
