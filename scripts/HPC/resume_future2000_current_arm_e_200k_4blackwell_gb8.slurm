#!/bin/bash

#SBATCH --job-name=f2k_E_200k_bw4
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:4
#SBATCH --mem=240G
#SBATCH --time=3-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
RGB_SOURCE=/networkhome/WMGDS/zhou_c/future2000_compact/multiview_960/rendered
OUTPUT=${REPO}/runs/learned_laplacian/future2000_current28view_arm_e_200k_seed7_gb8_v1
RESUME=${OUTPUT}/checkpoint_scaleup_2to4_step009000.pt
SOURCE_CONFIG=${OUTPUT}/config_pre_scaleup_2gpu.json
CONFIG=${OUTPUT}/resume_config_bw4.json
WORLD_SIZE=4
ACCUMULATION=2
EFFECTIVE_GLOBAL_BATCH=8
MASTER_PORT="$((24000 + SLURM_JOB_ID % 16000))"

for path in "${PYTHON}" "${MANIFEST}" "${RESUME}" "${SOURCE_CONFIG}"; do
    test -f "${path}"
done
test -d "${RGB_SOURCE}"
test ! -e "${OUTPUT}/metrics.json"

export PYTHONPATH=${REPO}/src
read -r RESUME_STEP RESUME_EPOCH NEXT_EPOCH OLD_WORLD_SIZE < <(
    "${PYTHON}" -c '
import sys, torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
state = payload.get("training_state", {})
step = int(payload["optimizer_steps"])
epoch = int(payload["epoch"])
next_epoch = int(state.get("next_epoch", -1))
groups = int(state.get("groups_completed_in_epoch", -1))
old_world = int(state.get("distributed_world_size", -1))
assert step == 9000
assert epoch == 9 and next_epoch == 10
assert groups == 0
assert old_world == 2
print(step, epoch, next_epoch, old_world)
' "${RESUME}"
)

"${PYTHON}" -c '
import json, sys
source, output = sys.argv[1:]
config = json.load(open(source))
multi = config["multi_object_training"]
multi["gradient_accumulation_meshes"] = 2
multi["max_optimizer_steps"] = 200000
metadata = config["experiment_metadata"]
metadata.update({
    "distributed_world_size": 4,
    "gradient_accumulation_meshes_per_rank": 2,
    "effective_global_batch_meshes": 8,
    "initialization": "resume",
    "resume_optimizer_step": 9000,
    "resume_old_world_size": 2,
    "execution_contract_change": (
        "world_size_2_to_4_at_epoch_boundary;"
        "accumulation_per_rank_4_to_2;effective_global_batch_remains_8"
    ),
})
with open(output, "w") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\n")
' "${SOURCE_CONFIG}" "${CONFIG}"

"${PYTHON}" -c '
import json, sys
config = json.load(open(sys.argv[1]))
multi = config["multi_object_training"]
metadata = config["experiment_metadata"]
assert multi["gradient_accumulation_meshes"] == 2
assert multi["max_optimizer_steps"] == 200000
assert metadata["distributed_world_size"] == 4
assert metadata["per_gpu_batch_meshes"] == 1
assert metadata["gradient_accumulation_meshes_per_rank"] == 2
assert metadata["effective_global_batch_meshes"] == 8
assert metadata["checkpoint_selection"] == "validation_direct_vertex_mse_only"
assert metadata["sealed_test_policy"] == "test_unavailable_to_training_and_checkpoint_selection"
' "${CONFIG}"

NODE_TMP_ROOT=${SLURM_TMPDIR:-/tmp/mlr-${SLURM_JOB_ID}}
RGB_LOCAL=${NODE_TMP_ROOT}/future2000_compact/multiview_960/rendered
mkdir -p "${RGB_LOCAL}" "${REPO}/slurm_logs"
rsync -a "${RGB_SOURCE}/" "${RGB_LOCAL}/"
SOURCE_COUNT="$(find "${RGB_SOURCE}" -type f | wc -l)"
LOCAL_COUNT="$(find "${RGB_LOCAL}" -type f | wc -l)"
test "${SOURCE_COUNT}" = "${LOCAL_COUNT}"

export MLR_IMAGE_PATH_REMAP_FROM=${RGB_SOURCE}
export MLR_IMAGE_PATH_REMAP_TO=${RGB_LOCAL}
export OMP_NUM_THREADS=8
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export MLR_DISTRIBUTED_TIMEOUT_SECONDS=600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=7
ulimit -n "$(ulimit -Hn)"

cd "${REPO}"
{
    echo "arm=E initialization=resume checkpoint=${RESUME}"
    echo "resume_optimizer_step=${RESUME_STEP} resume_epoch=${RESUME_EPOCH} next_epoch=${NEXT_EPOCH}"
    echo "old_world_size=${OLD_WORLD_SIZE} new_world_size=${WORLD_SIZE} epoch_boundary=true"
    echo "per_gpu_batch=1 accumulation_per_rank=${ACCUMULATION} effective_global_batch=${EFFECTIVE_GLOBAL_BATCH}"
    echo "execution_difference=DDP_sharding_and_sample_grouping_change_after_step_${RESUME_STEP}"
    echo "max_optimizer_steps=200000 checkpoint_selection=validation_direct_vertex_mse_only"
    echo "test_access=false overlapping_jobs=false"
    sha256sum "${RESUME}" "${SOURCE_CONFIG}" "${CONFIG}" "${MANIFEST}"
} | tee "${OUTPUT}/resume_contract_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader | tee "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
test "$(wc -l < "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq "${WORLD_SIZE}"
test "$(grep -c 'RTX PRO 6000 Blackwell' "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq "${WORLD_SIZE}"

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
        --no-persistent-workers \
        --gradient-accumulation-meshes "${ACCUMULATION}" \
        --max-optimizer-steps 200000 \
        --report-every-optimizer-steps 2000 \
        --resume-checkpoint "${RESUME}"
