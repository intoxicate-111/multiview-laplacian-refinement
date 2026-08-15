#!/bin/bash

#SBATCH --job-name=smoke_s50_hf1920
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L40:4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_synthetic_current_28view_direct_raw_original_plus_high_frequency_1920_20k_4gpu.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_hf_resolution_1920_20k_seed7/smoke_4gpu"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

test -f "${MANIFEST}"
test -f "${CONFIG}"
if [[ -e "${OUTPUT}/metrics.json" ]]; then
    echo "Refusing to overwrite completed smoke: ${OUTPUT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export OMP_NUM_THREADS=8
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${CONDA}" run --no-capture-output -n test torchrun \
    --nnodes=1 \
    --nproc-per-node=4 \
    --master-addr=127.0.0.1 \
    --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py \
        --manifest "${MANIFEST}" \
        --config "${CONFIG}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --max-optimizer-steps 1 \
        --report-every-optimizer-steps 1
