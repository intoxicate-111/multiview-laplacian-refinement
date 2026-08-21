#!/bin/bash

#SBATCH --job-name=s50_gt_raw_smoke
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:L40:2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3/gt_query_views_28_manifest.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_gt_query_28view_direct_raw_hf_20k_2gpu.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_gt_query_direct_raw_zero_shot_20k_seed7/smoke_2gpu_${SLURM_JOB_ID}"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

test -f "${MANIFEST}"
test -f "${CONFIG}"
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export OMP_NUM_THREADS=6
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH=src "${CONDA}" run --no-capture-output -n test torchrun \
    --nnodes=1 \
    --nproc-per-node=2 \
    --master-addr=127.0.0.1 \
    --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py \
        --manifest "${MANIFEST}" \
        --config "${CONFIG}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --max-optimizer-steps 1 \
        --report-every-optimizer-steps 1
