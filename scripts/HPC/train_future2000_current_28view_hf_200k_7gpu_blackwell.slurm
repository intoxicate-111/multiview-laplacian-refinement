#!/bin/bash
#SBATCH --job-name=f2k_hf_200k_bw7
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:7
#SBATCH --mem=448G
#SBATCH --time=3-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json"
RGB_SOURCE="/networkhome/WMGDS/zhou_c/future2000_compact/multiview_960/rendered"
CONFIG="${REPO}/configs/learned_laplacian/train_future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_hf_200k_7gpu_blackwell.json"
OUTPUT="${REPO}/runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_hf_200k_seed7_bw7"
WORLD_SIZE=7
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

for path in "${PYTHON}" "${MANIFEST}" "${CONFIG}"; do
    test -e "${path}"
done
test -d "${RGB_SOURCE}"
if [[ -e "${OUTPUT}/checkpoint_latest.pt" || -e "${OUTPUT}/metrics.json" || -e "${OUTPUT}/run_config.json" ]]; then
    echo "Refusing to overwrite or resume an existing run: ${OUTPUT}" >&2
    exit 2
fi

NODE_TMP_ROOT="${SLURM_TMPDIR:-/tmp/mlr-${SLURM_JOB_ID}}"
RGB_LOCAL="${NODE_TMP_ROOT}/future2000_compact/multiview_960/rendered"
mkdir -p "${RGB_LOCAL}" "${OUTPUT}" "${REPO}/slurm_logs"
rsync -a "${RGB_SOURCE}/" "${RGB_LOCAL}/"
SOURCE_COUNT="$(find "${RGB_SOURCE}" -type f | wc -l)"
LOCAL_COUNT="$(find "${RGB_LOCAL}" -type f | wc -l)"
test "${SOURCE_COUNT}" = "${LOCAL_COUNT}"

export MLR_IMAGE_PATH_REMAP_FROM="${RGB_SOURCE}"
export MLR_IMAGE_PATH_REMAP_TO="${RGB_LOCAL}"
export OMP_NUM_THREADS=8
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=7
export PYTHONPATH="${REPO}/src"
ulimit -n "$(ulimit -Hn)"

cd "${REPO}"
{
    echo "initialization=from_scratch"
    echo "resume_checkpoint=none"
    echo "world_size=${WORLD_SIZE} per_gpu_batch=1 accumulation=1 global_batch=${WORLD_SIZE}"
    echo "workers_per_rank=0 persistent_workers=false cuda_prefetch=true"
    echo "max_optimizer_steps=200000 report_every_optimizer_steps=2000"
    echo "manifest=${MANIFEST} config=${CONFIG} output=${OUTPUT}"
    echo "rgb_source=${RGB_SOURCE} rgb_local=${RGB_LOCAL} rgb_files=${LOCAL_COUNT}"
    echo "open_file_limit_soft=$(ulimit -Sn) open_file_limit_hard=$(ulimit -Hn)"
    df -h "${NODE_TMP_ROOT}" /dev/shm
} | tee "${OUTPUT}/launch_contract_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader | tee "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"

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
