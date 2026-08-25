#!/bin/bash

#SBATCH --job-name=s50_hard_anchor_lambda0_bw8
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
SOURCE_CONFIG="${REPO}/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7/run_config.json"
CONNECTIVITY="${REPO}/runs/diagnostics/sofa50_hard_anchor_lambda0/connectivity/summary.json"
SOLVER_AUDIT="${REPO}/runs/diagnostics/sofa50_hard_anchor_lambda0/solver_audit/audit.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_i_lambda0_hard_anchor_20k_seed7"
CONFIG="${OUTPUT}/run_config.json"
MASTER_PORT="$((22000 + SLURM_JOB_ID % 18000))"
for path in "${PYTHON}" "${MANIFEST}" "${SOURCE_CONFIG}" "${CONNECTIVITY}" "${SOLVER_AUDIT}"; do test -f "${path}"; done
test ! -e "${OUTPUT}/checkpoint_latest.pt"
test ! -e "${OUTPUT}/metrics.json"
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
cd "${REPO}"
"${PYTHON}" - "${CONNECTIVITY}" "${SOLVER_AUDIT}" <<'PY'
import json, sys
assert json.load(open(sys.argv[1], encoding="utf-8"))["contract_audit"] is True
assert json.load(open(sys.argv[2], encoding="utf-8"))["passed"] is True
PY
"${PYTHON}" scripts/build_sofa50_hard_anchor_lambda0_config.py \
  --arm-b-run-config "${SOURCE_CONFIG}" --output "${CONFIG}"
"${PYTHON}" - "${CONFIG}" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
r = c["training"]["recovery_aware_geometry_loss"]
assert r["solver"] == "hard_anchor_lambda0" and r["lambda"] == 0.0
assert r["beta"] == 1e-2 and r["compute_dtype"] == "float64"
assert r["maximum_iterations"] == 2048 and r["tolerance"] == 1e-4
assert c["multi_object_training"]["max_optimizer_steps"] == 20000
assert c["multi_object_training"]["gradient_accumulation_meshes"] == 1
assert c["recovery"]["solver"] == "reduced_hard_anchor_undamped_normal_sparse_lu_float64"
assert c["recovery"]["centroid_constraint"] is False
assert c["recovery"]["soft_positional_penalty"] is False
assert c["recovery"]["hidden_tikhonov_damping"] is False
PY
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader \
  > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
  scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
  --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
  --gradient-accumulation-meshes 1
