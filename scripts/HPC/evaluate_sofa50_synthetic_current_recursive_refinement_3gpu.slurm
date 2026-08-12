#!/bin/bash

#SBATCH --job-name=rec_refine_b3
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
H2_ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7"
B_RUN="${H2_ROOT}/B_direct_raw_laplacian"
BASELINE="${H2_ROOT}/analysis"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_b_recursive_refinement_3round_seed7"
SCRIPT="${REPO}/scripts/evaluate_sofa50_synthetic_current_recursive_refinement.py"

for path in \
    "${MANIFEST}" \
    "${SCRIPT}" \
    "${B_RUN}/checkpoint_latest.pt" \
    "${B_RUN}/run_config.json" \
    "${BASELINE}/topk_oracle_replacement_per_sample.csv" \
    "${BASELINE}/prediction_per_sample.csv"
do
    test -f "${path}"
done
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
export PYOPENGL_PLATFORM="egl"
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${SCRIPT}" \
        --manifest "${MANIFEST}" \
        --run-dir "${B_RUN}" \
        --baseline-analysis-dir "${BASELINE}" \
        --output-dir "${OUTPUT}" \
        --rounds 3 \
        --shard-index "${SLURM_ARRAY_TASK_ID}" \
        --shard-count 3 \
        --visibility-size 960 \
        --device cuda
