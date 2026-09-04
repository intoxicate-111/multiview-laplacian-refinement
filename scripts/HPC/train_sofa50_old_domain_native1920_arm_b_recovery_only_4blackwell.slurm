#!/bin/bash
#SBATCH --job-name=s50_old_Bvonly_bw4
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --time=2-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json"
REFERENCE="${REPO}/configs/learned_laplacian/train_sofa50_old_domain_native1920_arm_b_20k_4blackwell.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_arm_b_recovery_only_20k_seed7_bw4_v1"
CONFIG="${OUTPUT}/training_config.json"
SOLVER="${REPO}/src/mlr/learned_laplacian/differentiable_sparse_recovery.py"
WORLD_SIZE=4
ACCUMULATION=2
MASTER_PORT="$((21000 + SLURM_JOB_ID % 18000))"
SMOKE_PORT="$((MASTER_PORT + 1))"

for path in "${PYTHON}" "${MANIFEST}" "${REFERENCE}" "${SOLVER}"; do
    test -f "${path}"
done
if [[ -e "${OUTPUT}/checkpoint_latest.pt" || -e "${OUTPUT}/metrics.json" || -e "${OUTPUT}/run_config.json" ]]; then
    echo "Refusing to overwrite or resume an existing run: ${OUTPUT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"

"${PYTHON}" -c '
import copy, json, sys
reference_path, output_path = sys.argv[1:]
reference = json.load(open(reference_path))
candidate = copy.deepcopy(reference)
loss = candidate["training"]["recovery_aware_geometry_loss"]
loss["prediction_loss_weight"] = 0.0
loss["beta"] = 1.0
candidate["method"] = "sofa50_old_domain_native1920_arm_b_recovery_only"
candidate["experiment_metadata"].update({
    "experiment": "Sofa50_old_domain_native1920_Arm_B_recovered_geometry_only",
    "beta": 1.0,
    "prediction_loss_weight": 0.0,
    "loss": "recovered_vertex_mse_only_implicit_through_laplacian_solve",
    "checkpoint_selection": "validation_recovered_vertex_mse_only",
    "initialization": "from_scratch",
})
json.dump(candidate, open(output_path, "w"), indent=2, sort_keys=True)
' "${REFERENCE}" "${CONFIG}"

"${PYTHON}" -c '
import copy, json, sys
candidate = json.load(open(sys.argv[1]))
reference = json.load(open(sys.argv[2]))
expected = copy.deepcopy(reference)
loss = expected["training"]["recovery_aware_geometry_loss"]
loss["prediction_loss_weight"] = 0.0
loss["beta"] = 1.0
expected["method"] = "sofa50_old_domain_native1920_arm_b_recovery_only"
expected["experiment_metadata"].update({
    "experiment": "Sofa50_old_domain_native1920_Arm_B_recovered_geometry_only",
    "beta": 1.0,
    "prediction_loss_weight": 0.0,
    "loss": "recovered_vertex_mse_only_implicit_through_laplacian_solve",
    "checkpoint_selection": "validation_recovered_vertex_mse_only",
    "initialization": "from_scratch",
})
assert candidate == expected
loss = candidate["training"]["recovery_aware_geometry_loss"]
assert loss["enabled"] is True and loss["prediction_loss_weight"] == 0.0 and loss["beta"] == 1.0
assert loss["lambda"] == 0.01 and loss["differentiation"] == "implicit_custom_autograd"
assert candidate["dataset"]["expected_split_counts"] == {"train": 200, "validation": 25, "test": 25}
assert candidate["multi_object_training"]["max_optimizer_steps"] == 20000
assert candidate["multi_object_training"]["gradient_accumulation_meshes"] == 2
' "${CONFIG}" "${REFERENCE}"

export PYTHONPATH="${REPO}/src"
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export MLR_DISTRIBUTED_TIMEOUT_SECONDS=600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8
cd "${REPO}"

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
test "$(wc -l < "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 4
test "$(grep -c 'RTX PRO 6000 Blackwell' "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 4

SMOKE_ROOT="${SLURM_TMPDIR:-/tmp}/sofa50-old-b-recovery-only-${SLURM_JOB_ID}"
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
    echo "dataset=sofa50_old_domain_native1920_v1 views=28 resize=none"
    echo "loss=recovered_vertex_mse_only prediction_loss_weight=0 beta=1"
    echo "operator=uniform_random_walk lambda=0.01 pcg=float32,tol1e-4,max256 implicit_backward=true"
    echo "world_size=${WORLD_SIZE} accumulation_per_rank=${ACCUMULATION} effective_global_batch=8"
    echo "max_optimizer_steps=20000 checkpoint_selection=validation_recovered_vertex_mse_only"
    echo "sealed_test_access=false external_evaluation_submitted=false"
    echo "in_allocation_five_step_real_data_gate=passed"
    sha256sum "${MANIFEST}" "${REFERENCE}" "${CONFIG}" "${SOLVER}"
} > "${OUTPUT}/launch_contract_${SLURM_JOB_ID}.txt"

"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node="${WORLD_SIZE}" \
    --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
    --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
    --gradient-accumulation-meshes "${ACCUMULATION}"
