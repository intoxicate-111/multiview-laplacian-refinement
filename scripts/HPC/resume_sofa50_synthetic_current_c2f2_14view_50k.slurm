#!/bin/bash

#SBATCH --job-name=c2f2_current_50k
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current/manifest.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_synthetic_current_c2f2_14view_20k.json"
SOURCE_RUN="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_c2f2_14view_20k_seed7"
RESUME_CHECKPOINT="${SOURCE_RUN}/checkpoint_latest.pt"
OUTPUT_DIR="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_c2f2_14view_50k_resume_from_20k_seed7"
TARGET_STEPS=50000
GPUS_PER_NODE=2

for path in "${MANIFEST}" "${CONFIG}" "${RESUME_CHECKPOINT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing input file: ${path}"
        exit 2
    fi
done
if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty continuation run: ${OUTPUT_DIR}"
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${OUTPUT_DIR}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="${REPO}/src"

cd "${REPO}"
echo "job_id=${SLURM_JOB_ID}"
echo "resume_after_analysis_job=15649"
echo "resume_checkpoint=${RESUME_CHECKPOINT}"
echo "output_dir=${OUTPUT_DIR}"
echo "target_optimizer_steps=${TARGET_STEPS}"
echo "world_size=${GPUS_PER_NODE}"
nvidia-smi

srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    "${CONDA}" run --no-capture-output -n test \
    torchrun \
        --nnodes=1 \
        --nproc-per-node="${GPUS_PER_NODE}" \
        --node-rank=0 \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        scripts/train_multi_mesh_laplacian.py \
            --manifest "${MANIFEST}" \
            --config "${CONFIG}" \
            --output-dir "${OUTPUT_DIR}" \
            --device cuda \
            --resume-checkpoint "${RESUME_CHECKPOINT}" \
            --max-optimizer-steps "${TARGET_STEPS}"
