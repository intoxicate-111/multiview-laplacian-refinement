#!/bin/bash
#SBATCH --job-name=f2k_Bmix_200k_bw4
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

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json"
RGB_SOURCE="/networkhome/WMGDS/zhou_c/future2000_compact/multiview_960/rendered"
CONFIG="${REPO}/configs/learned_laplacian/train_future2000_current28view_arm_b_mixed_loss_200k_4blackwell.json"
ARM_B_REFERENCE="${REPO}/configs/learned_laplacian/train_sofa50_old_domain_native1920_arm_b_20k_4blackwell.json"
ARM_B_CHECKPOINT="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_arm_b_20k_seed7_v1/checkpoint_best.pt"
OUTPUT="${REPO}/runs/learned_laplacian/future2000_current28view_arm_b_mixed_loss_200k_seed7_bw4_v1"
SOLVER="${REPO}/src/mlr/learned_laplacian/differentiable_sparse_recovery.py"
WORLD_SIZE=4
ACCUMULATION=2
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

for path in "${PYTHON}" "${MANIFEST}" "${CONFIG}" "${ARM_B_REFERENCE}" \
    "${ARM_B_CHECKPOINT}" "${SOLVER}"; do
    test -f "${path}"
done
test -d "${RGB_SOURCE}"
if [[ -e "${OUTPUT}/checkpoint_latest.pt" || -e "${OUTPUT}/metrics.json" || -e "${OUTPUT}/run_config.json" ]]; then
    echo "Refusing to overwrite or resume an existing run: ${OUTPUT}" >&2
    exit 2
fi

# Fail closed unless the architecture, inputs, representation and recovery
# operator match the completed current Arm-B run. The explicitly audited
# difference is the requested Future2000 dataset/horizon; the objective is the established mixed Arm-B loss.
"${PYTHON}" -c '
import json, sys
candidate = json.load(open(sys.argv[1]))
reference = json.load(open(sys.argv[2]))
keys = (
    "input_mode", "image_encoder", "model", "confidence",
    "recovery", "prediction_semantics", "target_mode", "target_scaling",
    "query_training", "local_query_jitter", "renderer_visibility",
)
for key in keys:
    if candidate.get(key) != reference.get(key):
        raise SystemExit(f"Current Arm-B architecture/operator mismatch: {key}")
candidate_training = candidate["training"]
reference_training = reference["training"]
candidate_without_recovery = {
    key: value for key, value in candidate_training.items()
    if key != "recovery_aware_geometry_loss"
}
reference_without_recovery = {
    key: value for key, value in reference_training.items()
    if key != "recovery_aware_geometry_loss"
}
assert candidate_without_recovery == reference_without_recovery
loss = candidate_training["recovery_aware_geometry_loss"]
reference_loss = reference_training["recovery_aware_geometry_loss"]
assert loss["enabled"] is True
assert loss["prediction_loss_weight"] == 1.0
assert loss["beta"] == 0.01
assert float(reference_loss.get("prediction_loss_weight", 1.0)) == 1.0
assert reference_loss["beta"] == 0.01
for key in (
    "clean_vertex_use", "compute_dtype", "differentiation", "lambda",
    "maximum_iterations", "runtime_diagnostics", "tolerance",
):
    assert loss[key] == reference_loss[key], key
assert candidate["multi_object_training"]["gradient_accumulation_meshes"] == 2
assert candidate["multi_object_training"]["max_optimizer_steps"] == 200000
assert candidate["dataset"]["expected_split_counts"] == {
    "train": 8000, "validation": 1000, "test": 1000
}
assert candidate["experiment_metadata"]["loss_contract"] == "raw_laplacian_huber_plus_1e-2_recovered_vertex_mse_implicit_through_laplacian_solve"
print("current_arm_b_architecture_mixed_loss_contract=true")
' "${CONFIG}" "${ARM_B_REFERENCE}"

NODE_TMP_ROOT="${SLURM_TMPDIR:-/tmp/mlr-${SLURM_JOB_ID}}"
RGB_LOCAL="${NODE_TMP_ROOT}/future2000_compact/multiview_960/rendered"
SMOKE_ROOT="${NODE_TMP_ROOT}/future2000_arm_b_mixed_loss_smoke"
mkdir -p "${RGB_LOCAL}" "${SMOKE_ROOT}" "${REPO}/slurm_logs"
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
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export MLR_DISTRIBUTED_TIMEOUT_SECONDS=600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=7
export PYTHONPATH="${REPO}/src"
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
test -f "${SMOKE_ROOT}/run/training_step_history.json"
"${PYTHON}" -c '
import json, math, sys
metrics = json.load(open(sys.argv[1]))
history = json.load(open(sys.argv[2]))
assert int(metrics["optimizer_steps"]) == 5
assert history and int(history[-1]["optimizer_steps"]) == 5
assert all(int(row["pcg_failed_solves"]) == 0 for row in history)
assert all(int(row["nan_inf_count"]) == 0 for row in history)
assert all(
    math.isfinite(float(row["pcg_relative_residual_max"]))
    and float(row["pcg_relative_residual_max"]) <= 1.05e-4
    for row in history
)
assert all(math.isfinite(float(row["train_recovery_refine_loss"])) for row in history)
assert all(
    math.isclose(
        float(row["train_objective"]),
        float(row["train_loss"]) + 0.01 * float(row["train_recovery_refine_loss"]),
        rel_tol=2e-5,
        abs_tol=1e-10,
    )
    for row in history
)
' "${SMOKE_ROOT}/run/metrics.json" "${SMOKE_ROOT}/run/training_step_history.json"

mkdir -p "${OUTPUT}"
{
    echo "initialization=from_scratch resume_checkpoint=none"
    echo "architecture_contract=completed_current_Arm_B"
    echo "architecture_reference=${ARM_B_REFERENCE}"
    echo "loss_contract=raw_laplacian_huber_plus_0.01_recovered_vertex_mse prediction_loss_weight=1 beta=0.01 implicit_laplacian_backward=true"
    echo "dataset=Future2000_gt_adaptive_v2 views=28 input_resolution=960"
    echo "world_size=${WORLD_SIZE} per_gpu_batch=1 accumulation=${ACCUMULATION} global_batch=8"
    echo "max_optimizer_steps=200000 checkpoint_selection=validation_objective_only"
    echo "test_access=false external_evaluation_submitted=false"
    echo "manifest=${MANIFEST} config=${CONFIG} output=${OUTPUT}"
    echo "rgb_source=${RGB_SOURCE} rgb_local=${RGB_LOCAL} rgb_files=${LOCAL_COUNT}"
    echo "in_allocation_five_step_smoke=passed"
    sha256sum "${MANIFEST}" "${CONFIG}" "${ARM_B_REFERENCE}" \
        "${ARM_B_CHECKPOINT}" "${SOLVER}"
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
        --no-persistent-workers \
        --gradient-accumulation-meshes "${ACCUMULATION}" \
        --max-optimizer-steps 200000 \
        --report-every-optimizer-steps 2000
