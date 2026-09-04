#!/bin/bash
#SBATCH --job-name=s50_Bvonly_bw4
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
BASELINE="${REPO}/configs/learned_laplacian/train_sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_2gpu_l40.json"
BUILDER="${REPO}/scripts/build_sofa50_recovery_aware_ablation_config.py"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_v2_arm_b_recovery_only_lambda1e-2_20k_seed7_bw4_v1"
CONFIG="${OUTPUT}/training_config.json"
WORLD_SIZE=4
ACCUMULATION=2
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
SMOKE_PORT="$((MASTER_PORT + 1))"

for path in "${PYTHON}" "${MANIFEST}" "${BASELINE}" "${BUILDER}"; do
    test -f "${path}"
done
if [[ -e "${OUTPUT}/checkpoint_latest.pt" || -e "${OUTPUT}/metrics.json" || -e "${OUTPUT}/run_config.json" ]]; then
    echo "Refusing to overwrite or resume an existing run: ${OUTPUT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src"
"${PYTHON}" "${BUILDER}" \
    --baseline "${BASELINE}" --output "${CONFIG}" --arm B \
    --lambda-value 0.01 --beta 1.0 --prediction-loss-weight 0.0 \
    --max-optimizer-steps 20000 --run-kind final \
    --distributed-world-size "${WORLD_SIZE}" --effective-global-batch 8 \
    --training-gpu-model "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
    --initialization from_scratch
"${PYTHON}" -c '
import json, sys
c = json.load(open(sys.argv[1]))
c["training"]["recovery_aware_geometry_loss"]["runtime_diagnostics"] = True
c["experiment_metadata"]["runtime_diagnostics"] = "enabled_for_gradient_and_pcg_audit_only"
json.dump(c, open(sys.argv[1], "w"), indent=2, sort_keys=True)
' "${CONFIG}"

"${PYTHON}" -c '
import json, sys
candidate = json.load(open(sys.argv[1]))
baseline = json.load(open(sys.argv[2]))
for key in (
    "seed", "dataset", "input_mode", "target_mode", "target_semantics",
    "target_definition", "target_scaling", "query_training",
    "local_query_jitter", "renderer_visibility", "image_encoder", "model",
    "data_loading",
):
    assert candidate[key] == baseline[key], key
loss = candidate["training"]["recovery_aware_geometry_loss"]
assert loss["enabled"] is True
assert loss["prediction_loss_weight"] == 0.0
assert loss["beta"] == 1.0
assert loss["lambda"] == 0.01
assert loss["differentiation"] == "implicit_custom_autograd"
assert loss["runtime_diagnostics"] is True
assert candidate["multi_object_training"]["max_optimizer_steps"] == 20000
assert candidate["multi_object_training"]["gradient_accumulation_meshes"] == 2
assert candidate["experiment_metadata"]["training_objective"] == "recovered_vertex_mse_only"
' "${CONFIG}" "${BASELINE}"

export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export MLR_DISTRIBUTED_TIMEOUT_SECONDS=600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8
cd "${REPO}"

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
test "$(wc -l < "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 4
test "$(grep -c 'RTX PRO 6000 Blackwell' "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 4

SMOKE_ROOT="${SLURM_TMPDIR:-/tmp}/sofa50-v2-b-recovery-only-${SLURM_JOB_ID}"
mkdir -p "${SMOKE_ROOT}"
SMOKE_CONFIG="${SMOKE_ROOT}/smoke_config.json"
"${PYTHON}" -c '
import json, sys
c = json.load(open(sys.argv[1]))
m = c["multi_object_training"]
m["max_optimizer_steps"] = 5
m["checkpoint_optimizer_steps"] = [5]
m["report_every_optimizer_steps"] = 1
json.dump(c, open(sys.argv[2], "w"), indent=2, sort_keys=True)
' "${CONFIG}" "${SMOKE_CONFIG}"
"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node="${WORLD_SIZE}" \
    --master-addr=127.0.0.1 --master-port="${SMOKE_PORT}" \
    scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${SMOKE_CONFIG}" \
    --output-dir "${SMOKE_ROOT}" --device cuda --num-workers 0 --no-persistent-workers \
    --gradient-accumulation-meshes "${ACCUMULATION}" \
    --max-optimizer-steps 5 --report-every-optimizer-steps 1
"${PYTHON}" -c '
import json, math, sys
metrics = json.load(open(sys.argv[1]))
history = json.load(open(sys.argv[2]))
assert int(metrics["optimizer_steps"]) == 5
assert history and int(history[-1]["optimizer_steps"]) == 5
for row in history:
    assert int(row["pcg_failed_solves"]) == 0
    assert int(row["nan_inf_count"]) == 0
    assert math.isclose(float(row["train_objective"]), float(row["train_recovery_refine_loss"]), rel_tol=2e-5, abs_tol=1e-10)
    assert float(row["delta_pred_gradient_norm"]) > 0
    assert float(row["prediction_head_gradient_norm"]) > 0
' "${SMOKE_ROOT}/metrics.json" "${SMOKE_ROOT}/training_step_history.json"

{
    echo "arm=B initialization=from_scratch checkpoint=none"
    echo "dataset=Sofa50MultiTopologyRawLap500_v2 recipes=A1_to_D2 views=28 resolution=960"
    echo "loss=recovered_vertex_mse_only prediction_loss_weight=0 beta=1"
    echo "operator=uniform_random_walk lambda=0.01 implicit_backward=true"
    echo "world_size=${WORLD_SIZE} accumulation_per_rank=${ACCUMULATION} effective_global_batch=8"
    echo "max_optimizer_steps=20000 checkpoint_selection=validation_recovered_vertex_mse_only"
    echo "sealed_test_access=false external_evaluation_submitted=false"
    echo "in_allocation_five_step_real_data_gate=passed"
    sha256sum "${MANIFEST}" "${BASELINE}" "${CONFIG}" "${BUILDER}"
} > "${OUTPUT}/launch_contract_${SLURM_JOB_ID}.txt"

"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node="${WORLD_SIZE}" \
    --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
    --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
    --gradient-accumulation-meshes "${ACCUMULATION}"
