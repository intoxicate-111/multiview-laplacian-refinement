#!/bin/bash

#SBATCH --job-name=s50_old1920_Br
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
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_old_domain_native1920_arm_b_20k_4blackwell.json"
PREFLIGHT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/preflight/runtime_preflight.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_arm_b_20k_seed7_v1"
RESUME="${OUTPUT}/checkpoint_resume_from_17498_step000175.pt"
SOLVER="${REPO}/src/mlr/learned_laplacian/differentiable_sparse_recovery.py"
MASTER_PORT="$((22000 + SLURM_JOB_ID % 18000))"

for path in "${PYTHON}" "${MANIFEST}" "${CONFIG}" "${PREFLIGHT}" "${RESUME}" "${SOLVER}"; do
  test -f "${path}"
done
test ! -e "${OUTPUT}/metrics.json"
"${PYTHON}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["contract_audit"] is True' "${PREFLIGHT}"
read -r RESUME_STEP RESUME_EPOCH NEXT_EPOCH < <(
  "${PYTHON}" -c 'import sys,torch; p=torch.load(sys.argv[1],map_location="cpu",weights_only=False); s=p.get("training_state",{}); step=int(p["optimizer_steps"]); epoch=int(p["epoch"]); next_epoch=int(s.get("next_epoch",-1)); assert 0 < step < 20000; assert int(s.get("groups_completed_in_epoch",-1)) == 0; assert int(s.get("distributed_world_size",-1)) == 4; assert next_epoch == epoch + 1; print(step,epoch,next_epoch)' "${RESUME}"
)

mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src"
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000 MLR_DISTRIBUTED_TIMEOUT_SECONDS=600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8
cd "${REPO}"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
{
  echo "arm=B initialization=resume checkpoint=${RESUME}"
  echo "resume_optimizer_step=${RESUME_STEP} resume_epoch=${RESUME_EPOCH} next_epoch=${NEXT_EPOCH}"
  echo "world_size=4 accumulation_per_rank=2 effective_global_batch=8"
  echo "objective=huber_raw_laplacian+0.01*recovered_vertex_mse lambda_B=0.01"
  echo "solver_contract=float32_pcg tolerance=1e-4 maximum_iterations=256 true_residual_recheck=true"
  echo "distributed_timeout_seconds=600 sealed_test_access=false"
  sha256sum "${RESUME}" "${CONFIG}" "${SOLVER}"
} > "${OUTPUT}/resume_contract_${SLURM_JOB_ID}.txt"

"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
  scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
  --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
  --gradient-accumulation-meshes 2 --resume-checkpoint "${RESUME}"
