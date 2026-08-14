#!/bin/bash

#SBATCH --job-name=mlr_ddp
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
    echo "Usage: sbatch $0 MANIFEST CONFIG OUTPUT_DIR [RESUME_CHECKPOINT]"
    exit 2
fi

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="$1"
CONFIG="$2"
OUTPUT_DIR="$3"
RESUME_CHECKPOINT="${4:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

if [[ ! "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS_PER_NODE must be a positive integer."
    exit 2
fi
for path in "${MANIFEST}" "${CONFIG}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing input file: ${path}"
        exit 2
    fi
done
if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "Missing resume checkpoint: ${RESUME_CHECKPOINT}"
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${OUTPUT_DIR}"

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
export REPO CONDA MANIFEST CONFIG OUTPUT_DIR RESUME_CHECKPOINT
export GPUS_PER_NODE MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-slurm}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-slurm}"

echo "nodes=${SLURM_NNODES}"
echo "gpus_per_node=${GPUS_PER_NODE}"
echo "world_size=$((SLURM_NNODES * GPUS_PER_NODE))"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "nccl_socket_ifname=${NCCL_SOCKET_IFNAME}"
echo "manifest=${MANIFEST}"
echo "config=${CONFIG}"
echo "output_dir=${OUTPUT_DIR}"

srun --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    bash -c '
        set -euo pipefail
        cd "${REPO}"
        arguments=(
            --manifest "${MANIFEST}"
            --config "${CONFIG}"
            --output-dir "${OUTPUT_DIR}"
            --device cuda
        )
        if [[ -n "${RESUME_CHECKPOINT}" ]]; then
            arguments+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
        fi
        PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
            torchrun \
                --nnodes="${SLURM_NNODES}" \
                --nproc-per-node="${GPUS_PER_NODE}" \
                --node-rank="${SLURM_PROCID}" \
                --master-addr="${MASTER_ADDR}" \
                --master-port="${MASTER_PORT}" \
                scripts/train_multi_mesh_laplacian.py \
                "${arguments[@]}"
    '
