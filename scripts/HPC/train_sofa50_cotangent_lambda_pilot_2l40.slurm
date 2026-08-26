#!/bin/bash

#SBATCH --job-name=s50_cot_pilot
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-6%2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
VALUES=("1e-3" "3e-3" "1e-2" "3e-2" "1e-1" "3e-1" "1")
VALUE="${VALUES[${SLURM_ARRAY_TASK_ID}]}"
CONFIG="${REPO}/configs/learned_laplacian/cotangent_pilot_lambda_${VALUE}.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_v2_cotangent_lambda_pilot/lambda_${VALUE}"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

for path in "${PYTHON}" "${MANIFEST}" "${CONFIG}"; do
    test -f "${path}"
done
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="$(dirname "${PYTHON}"):${PATH}"
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
test "$(wc -l < "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 2
test "$(grep -c 'L40' "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 2
"${PYTHON}" -c 'from pathlib import Path; import torch; expected=Path("/networkhome/WMGDS/zhou_c/miniconda3/envs/test"); actual=Path(torch.__file__).resolve(); assert str(actual).startswith(str(expected)), (actual, expected); assert torch.cuda.is_available(); assert torch.cuda.device_count()==2; assert all("L40" in torch.cuda.get_device_name(i) for i in range(2)); print(torch.__version__, torch.version.cuda, actual, flush=True)'

{
    echo "pilot=true operator=symmetric_cotangent_stiffness mass_normalization=false"
    echo "lambda=${VALUE} optimizer_steps=2000"
    echo "world_size=2 accumulation_per_rank=4 effective_global_batch=8"
    echo "selection=validation_final_VH_unified_v2_chamfer_only"
    sha256sum "${CONFIG}"
} > "${OUTPUT}/launch_contract_${SLURM_JOB_ID}.txt"

"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node=2 \
    --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
    --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
    --gradient-accumulation-meshes 4

"${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["optimizer_steps"] == 2000; assert len(p["per_object_metrics"]["validation"]) == 50' "${OUTPUT}/metrics.json"
