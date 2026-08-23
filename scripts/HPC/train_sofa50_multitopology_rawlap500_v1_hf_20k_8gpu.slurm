#!/bin/bash

#SBATCH --job-name=s50_mt500_hf
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1/manifest.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_multitopology_rawlap500_v1_hf_20k_8gpu.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v1_hf_20k_seed7"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
WORLD_SIZE=8
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

for path in "${PYTHON}" "${MANIFEST}" "${CONFIG}"; do
    test -f "${path}"
done
if [[ -e "${OUTPUT}/checkpoint_latest.pt" || -e "${OUTPUT}/metrics.json" ]]; then
    echo "Refusing to overwrite existing training output: ${OUTPUT}" >&2
    exit 2
fi

export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=7
export PYTHONPATH="${REPO}/src"
export OMP_NUM_THREADS=8
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
{
    echo "hardware=nvidia_rtx_pro_6000_blackwell_server_edition"
    echo "initialization=from_scratch"
    echo "resume_checkpoint=none"
    echo "world_size=8 per_gpu_batch=1 accumulation=1 effective_global_batch=8"
    echo "comparison_batch_contract=authorized_difference_from_960_HF_baseline_global_batch_2"
} > "${OUTPUT}/launch_contract_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
"${PYTHON}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node="${WORLD_SIZE}" \
    --master-addr=127.0.0.1 \
    --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py \
        --manifest "${MANIFEST}" \
        --config "${CONFIG}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --num-workers 0 \
        --no-persistent-workers
