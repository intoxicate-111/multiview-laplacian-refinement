#!/bin/bash

#SBATCH --job-name=s50_old1920_ctBE
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
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_old_domain_native1920_continuous_b_e_4blackwell.json"
RUN_B="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_arm_b_20k_seed7_v1"
RUN_E="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_arm_e_20k_seed7_v1"
LAMBDA_SELECTION="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/validation_selection/frozen_lambda/lambda_selection.json"
PREFLIGHT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/continuous/preflight/step0_validation.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_continuous_b_e_lr1e-4_20k_seed7_v1"
MASTER_PORT="$((23000 + SLURM_JOB_ID % 16000))"

for path in \
  "${PYTHON}" "${MANIFEST}" "${CONFIG}" "${PREFLIGHT}" "${LAMBDA_SELECTION}" \
  "${RUN_B}/checkpoint_best.pt" "${RUN_E}/checkpoint_best.pt"; do
  test -f "${path}"
done
"${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["contract_audit"] is True; assert p["split"]=="validation"; assert p["test_accessed"] is False; assert p["parameter_count"]==1652230; assert p["gradient_audit"]["passed"] is True' "${PREFLIGHT}"
test ! -e "${OUTPUT}/checkpoint_latest.pt"
test ! -e "${OUTPUT}/metrics.json"
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src"
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8
cd "${REPO}"

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
test "$(wc -l < "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 4
test "$(grep -c 'RTX PRO 6000 Blackwell' "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 4

SELECTED_LAMBDA="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_lambda"])' "${LAMBDA_SELECTION}")"
{
  echo "initialization=complete_validation_selected_Arm_B_plus_complete_Arm_E"
  echo "shared_backbone=false parameter_count=1652230"
  echo "domain=historical_native1920 views=28 resize=none"
  echo "world_size=4 accumulation_per_rank=2 effective_global_batch=8"
  echo "optimizer=fresh_Adam learning_rate=1e-4"
  echo "objective=mean_i_squared_l2_final_hybrid_vertices_minus_clean_only"
  echo "operator=uniform_random_walk lambda=${SELECTED_LAMBDA} pcg=float64,tol1e-8,max2048"
  echo "checkpoint_selector=validation_final_hybrid_unified_surface_chamfer_only"
  echo "sealed_test_access=false"
  sha256sum "${CONFIG}" "${RUN_B}/checkpoint_best.pt" "${RUN_E}/checkpoint_best.pt" \
    "${LAMBDA_SELECTION}" "${PREFLIGHT}"
} > "${OUTPUT}/launch_contract_${SLURM_JOB_ID}.txt"

"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
  scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
  --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
  --gradient-accumulation-meshes 2
