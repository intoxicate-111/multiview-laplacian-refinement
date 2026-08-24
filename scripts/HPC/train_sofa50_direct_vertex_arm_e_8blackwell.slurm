#!/bin/bash

#SBATCH --job-name=s50_direct_vertex_e_bw8
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
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
BASELINE="${REPO}/configs/learned_laplacian/train_sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_2gpu_l40.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_v2_direct_vertex_arm_e_20k_seed7"
CONFIG="${OUTPUT}/run_config.json"
WORLD_SIZE=8
MASTER_PORT="$((22000 + SLURM_JOB_ID % 18000))"

for path in "${PYTHON}" "${MANIFEST}" "${BASELINE}"; do test -f "${path}"; done
test ! -e "${OUTPUT}/checkpoint_latest.pt"
test ! -e "${OUTPUT}/metrics.json"
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src"
"${PYTHON}" "${REPO}/scripts/build_sofa50_direct_vertex_arm_e_config.py" \
  --baseline "${BASELINE}" --output "${CONFIG}" \
  --distributed-world-size 8 --effective-global-batch 8

"${PYTHON}" - "${CONFIG}" "${OUTPUT}/preflight_contract.json" <<'PY'
import json
import sys
from mlr.learned_laplacian.multi_trainer import _build_model

config = json.load(open(sys.argv[1], encoding="utf-8"))
assert config["prediction_semantics"] == "direct_vertex_displacement"
assert config["training"]["loss"] == "mse"
assert config["training"]["prediction_loss_space"] == "output_representation"
assert config["training"]["direct_vertex_runtime_diagnostics"] is True
assert config["training"]["recovery_aware_geometry_loss"]["enabled"] is False
assert config["confidence"]["enabled"] is False
assert config["recovery"]["mode"] == "direct_vertex_addition"
assert config["recovery"]["laplacian_operator_used"] is False
assert config["recovery"]["sparse_integration"] is False
assert config["recovery"]["lambda"] is None
assert config["recovery"]["visibility_gate"] is False
assert config["recovery"]["confidence_weighting"] is False
assert config["recovery"]["robust_loss"] is None
assert config["recovery"]["optimizer"] is None
assert config["multi_object_training"]["max_optimizer_steps"] == 20000
assert config["multi_object_training"]["gradient_accumulation_meshes"] == 1
model = _build_model(config, None, False)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
assert parameter_count == 826115, parameter_count
audit = {
    "implementation_audit": True,
    "parameter_count": parameter_count,
    "forward": "delta_v_pred=f_theta(I,C,V_input,F);V_refined=V_input+delta_v_pred",
    "loss": "mean_i(||delta_v_pred_i-(V_clean_i-V_input_i)||_2^2)",
    "no_gt_in_model_input": True,
    "no_laplacian_or_recovery_path": True,
}
json.dump(audit, open(sys.argv[2], "w", encoding="utf-8"), indent=2, sort_keys=True)
PY

export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8
cd "${REPO}"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
{
  echo "arm=E initialization=from_scratch checkpoint=none"
  echo "world_size=8 accumulation_per_rank=1 effective_global_batch=8"
  echo "target=V_clean-V_input objective=mean_vertex_squared_l2"
  echo "inference=V_input+delta_v_pred analytic_recovery=none"
} > "${OUTPUT}/launch_contract_${SLURM_JOB_ID}.txt"

"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node="${WORLD_SIZE}" \
  --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
  scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
  --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
  --gradient-accumulation-meshes 1
