#!/bin/bash

#SBATCH --job-name=s50_cot_full_bw8
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_v2_cotangent_hybrid_selected_8blackwell.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_v2_cotangent_hybrid_selected_20k_seed7"
PREFLIGHT="${OUTPUT}/preflight_audit.json"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

for path in "${PYTHON}" "${MANIFEST}" "${CONFIG}"; do
    test -f "${path}"
done
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
test "$(wc -l < "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 8
test "$(grep -c 'RTX PRO 6000 Blackwell' "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 8

"${PYTHON}" scripts/audit_sofa50_cotangent_hybrid_preflight.py \
    --manifest "${MANIFEST}" --config "${CONFIG}" \
    --output "${PREFLIGHT}" --representative-samples 3
"${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["contract_audit"] and p["gradient_audit"]["all_finite"] and all(x["pcg_converged"] for x in p["solver_audit"])' "${PREFLIGHT}"

"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node=8 \
    --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
    --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
    --gradient-accumulation-meshes 1

