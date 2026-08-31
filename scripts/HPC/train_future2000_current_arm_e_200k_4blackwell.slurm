#!/bin/bash
#SBATCH --job-name=f2k_E_200k_bw4
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:4
#SBATCH --mem=320G
#SBATCH --time=3-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
RGB_SOURCE=/networkhome/WMGDS/zhou_c/future2000_compact/multiview_960/rendered
FUTURE_BASELINE=${REPO}/configs/learned_laplacian/train_future2000_current28view_arm_b_mixed_loss_200k_4blackwell.json
SOFA_E_REFERENCE=${REPO}/configs/learned_laplacian/train_sofa50_old_domain_native1920_arm_e_20k_4blackwell.json
OUTPUT=${REPO}/runs/learned_laplacian/future2000_current28view_arm_e_200k_seed7_bw4_v1
CONFIG=${OUTPUT}/run_config.json
WORLD_SIZE=4
ACCUMULATION=2
MASTER_PORT="$((24000 + SLURM_JOB_ID % 16000))"

for path in "${PYTHON}" "${MANIFEST}" "${FUTURE_BASELINE}" "${SOFA_E_REFERENCE}"; do
    test -f "${path}"
done
test -d "${RGB_SOURCE}"
if [[ -e "${OUTPUT}/checkpoint_latest.pt" || -e "${OUTPUT}/metrics.json" || -e "${OUTPUT}/run_config.json" ]]; then
    echo "Refusing to overwrite or resume an existing run: ${OUTPUT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
export PYTHONPATH=${REPO}/src
"${PYTHON}" "${REPO}/scripts/build_future2000_current_arm_e_config.py" \
    --future2000-baseline "${FUTURE_BASELINE}" \
    --sofa50-arm-e-reference "${SOFA_E_REFERENCE}" \
    --output "${CONFIG}" \
    --distributed-world-size "${WORLD_SIZE}" \
    --effective-global-batch 8 \
    --max-optimizer-steps 200000

"${PYTHON}" -c '
import json, sys
from mlr.learned_laplacian.multi_trainer import _build_model
candidate = json.load(open(sys.argv[1]))
reference = json.load(open(sys.argv[2]))
for key in (
    "input_mode", "image_encoder", "model", "confidence", "recovery",
    "prediction_semantics", "target_mode", "target_scaling", "query_training",
    "local_query_jitter", "renderer_visibility", "training",
):
    if candidate.get(key) != reference.get(key):
        raise SystemExit(f"Sofa50 Arm-E architecture/objective mismatch: {key}")
assert candidate["dataset"]["expected_split_counts"] == {
    "train": 8000, "validation": 1000, "test": 1000
}
assert candidate["multi_object_training"]["max_optimizer_steps"] == 200000
assert candidate["multi_object_training"]["gradient_accumulation_meshes"] == 2
assert candidate["experiment_metadata"]["checkpoint_selection"] == "validation_direct_vertex_mse_only"
assert candidate["experiment_metadata"]["sealed_test_policy"] == "test_unavailable_to_training_and_checkpoint_selection"
model = _build_model(candidate, None, False)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
assert parameter_count == 826115, parameter_count
print(f"future2000_arm_e_contract=true parameter_count={parameter_count}")
' "${CONFIG}" "${SOFA_E_REFERENCE}"

NODE_TMP_ROOT=${SLURM_TMPDIR:-/tmp/mlr-${SLURM_JOB_ID}}
RGB_LOCAL=${NODE_TMP_ROOT}/future2000_compact/multiview_960/rendered
SMOKE_ROOT=${NODE_TMP_ROOT}/future2000_arm_e_smoke
mkdir -p "${RGB_LOCAL}" "${SMOKE_ROOT}"
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
"${PYTHON}" scripts/prepare_future2000_training_smoke.py \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --output-dir "${SMOKE_ROOT}/prepared"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/train_multi_mesh_laplacian.py \
    --manifest "${SMOKE_ROOT}/prepared/manifest.json" \
    --config "${SMOKE_ROOT}/prepared/config_phase1.json" \
    --output-dir "${SMOKE_ROOT}/run" \
    --device cuda \
    --num-workers 0 \
    --no-persistent-workers \
    --gradient-accumulation-meshes 2 \
    --max-optimizer-steps 5 \
    --report-every-optimizer-steps 1
test -f "${SMOKE_ROOT}/run/checkpoints/checkpoint_step_000005.pt"
test -f "${SMOKE_ROOT}/run/metrics.json"
"${PYTHON}" -c '
import json, math, sys
metrics = json.load(open(sys.argv[1]))
assert int(metrics["optimizer_steps"]) == 5
assert math.isfinite(float(metrics["train_loss"]))
' "${SMOKE_ROOT}/run/metrics.json"

{
    echo "arm=E initialization=from_scratch checkpoint=none"
    echo "architecture_contract=completed_Sofa50_Arm_E_direct_vertex"
    echo "objective=mean_same_index_vertex_residual_squared_l2"
    echo "inference=V_input+delta_v_pred analytic_recovery=none"
    echo "dataset=Future2000_gt_adaptive_v2 views=28 input_resolution=960"
    echo "world_size=${WORLD_SIZE} accumulation=${ACCUMULATION} effective_global_batch=8"
    echo "max_optimizer_steps=200000 checkpoint_selection=validation_direct_vertex_mse_only"
    echo "future_role=frozen_V_E_anchor_for_B_plus_E_fusion"
    echo "test_access=false"
    echo "in_allocation_five_step_smoke=passed"
    sha256sum "${MANIFEST}" "${CONFIG}" "${SOFA_E_REFERENCE}"
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
        --no-persistent-workers \
        --gradient-accumulation-meshes "${ACCUMULATION}" \
        --max-optimizer-steps 200000 \
        --report-every-optimizer-steps 2000
