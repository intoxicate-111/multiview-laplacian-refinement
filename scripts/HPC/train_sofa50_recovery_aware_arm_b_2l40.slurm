#!/bin/bash

#SBATCH --job-name=s50_sparse_B
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
BASELINE="${REPO}/configs/learned_laplacian/train_sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_2gpu_l40.json"
LAMBDA_SELECTION="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/lambda_validation/lambda_selection.json"
BETA_SELECTION="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/beta_validation/beta_selection.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7"
CONFIG="${OUTPUT}/run_config.json"
MASTER_PORT="$((24000 + SLURM_JOB_ID % 16000))"
for path in "${PYTHON}" "${MANIFEST}" "${BASELINE}" "${LAMBDA_SELECTION}" "${BETA_SELECTION}"; do test -f "${path}"; done
test ! -e "${OUTPUT}/checkpoint_latest.pt"
read -r LAMBDA BETA < <("${PYTHON}" -c 'import json,sys; l=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); assert l["contract_audit"]["passed"] and b["contract_audit"]["passed"]; assert l["selected_lambda"] == b["lambda"]; print(l["selected_lambda"], b["selected_beta"])' "${LAMBDA_SELECTION}" "${BETA_SELECTION}")
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src"
"${PYTHON}" "${REPO}/scripts/build_sofa50_recovery_aware_ablation_config.py" \
    --baseline "${BASELINE}" --output "${CONFIG}" --arm B \
    --lambda-value "${LAMBDA}" --beta "${BETA}" --max-optimizer-steps 20000 --run-kind final
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8
cd "${REPO}"
{
  echo "arm=B initialization=from_scratch checkpoint=none"
  echo "world_size=2 accumulation_per_rank=4 effective_global_batch=8"
  echo "lambda=${LAMBDA} beta=${BETA} recovery_loss=enabled"
} > "${OUTPUT}/launch_contract_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node=2 \
    --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
    --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers
