#!/bin/bash

#SBATCH --job-name=s50_anchor_bp
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
B0_RUN="${REPO}/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7"
E_RUN="${REPO}/runs/learned_laplacian/sofa50_v2_direct_vertex_arm_e_20k_seed7"
E_REPORT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_bp_positional_anchor_20k_seed7"
CACHE="${OUTPUT}/frozen_arm_e_anchor_cache"
CONFIG="${OUTPUT}/launch_config.json"
E_SHA="6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1"
WORLD_SIZE=4
MASTER_PORT="$((22000 + SLURM_JOB_ID % 18000))"

for path in \
    "${MANIFEST}" \
    "${B0_RUN}/config.json" \
    "${E_RUN}/config.json" \
    "${E_RUN}/checkpoint_best.pt" \
    "${E_REPORT}/shards/E_direct_vertex_residual.json" \
    "${E_REPORT}/shards/E_direct_vertex_residual_prediction_arrays.npz"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required input: ${path}"
        exit 2
    fi
done
if [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty controlled run: ${OUTPUT}"
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export PYTHONHASHSEED=7
export OMP_NUM_THREADS=8
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURM_JOB_NODELIST}"
echo "controlled_arm=B_P"
echo "world_size=${WORLD_SIZE}"
echo "effective_global_batch=8"
echo "output=${OUTPUT}"
nvidia-smi
"${PYTHON}" - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(4)]
assert all("L40" in name and "RTX" not in name for name in names), names
print({"visible_gpus": names}, flush=True)
PY
test "$(sha256sum "${E_RUN}/checkpoint_best.pt" | cut -d' ' -f1)" = "${E_SHA}"

"${PYTHON}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node="${WORLD_SIZE}" \
    --master-addr=127.0.0.1 \
    --master-port="${MASTER_PORT}" \
    scripts/cache_sofa50_frozen_arm_e_anchors.py \
        --manifest "${MANIFEST}" \
        --arm-e-config "${E_RUN}/config.json" \
        --arm-e-checkpoint "${E_RUN}/checkpoint_best.pt" \
        --arm-e-report "${E_REPORT}" \
        --output-dir "${CACHE}" \
        --device cuda

"${PYTHON}" scripts/build_sofa50_anchor_conditioning_config.py \
    --arm-b0-config "${B0_RUN}/config.json" \
    --arm-e-checkpoint "${E_RUN}/checkpoint_best.pt" \
    --anchor-cache-metadata "${CACHE}/metadata.json" \
    --output "${CONFIG}" \
    --world-size 4 \
    --global-batch 8

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/audit_sofa50_anchor_conditioning_gradient.py \
    --manifest "${MANIFEST}" \
    --bp-config "${CONFIG}" \
    --arm-e-config "${E_RUN}/config.json" \
    --arm-e-checkpoint "${E_RUN}/checkpoint_best.pt" \
    --anchor-cache-metadata "${CACHE}/metadata.json" \
    --output "${OUTPUT}/pretraining_gradient_audit.json" \
    --device cuda

"${PYTHON}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node="${WORLD_SIZE}" \
    --master-addr=127.0.0.1 \
    --master-port="${MASTER_PORT}" \
    scripts/train_sofa50_anchor_conditioned_laplacian.py \
        --manifest "${MANIFEST}" \
        --config "${CONFIG}" \
        --anchor-cache-metadata "${CACHE}/metadata.json" \
        --output-dir "${OUTPUT}" \
        --device cuda

echo "Arm-B_P training completed. No evaluation or follow-up job was launched."
