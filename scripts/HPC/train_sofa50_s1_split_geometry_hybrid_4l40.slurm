#!/bin/bash

#SBATCH --job-name=s50_s1_split_l40
#SBATCH --partition=long
#SBATCH --nodelist=gpu-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --time=2-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
S0_CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_v2_end_to_end_hybrid_single_loss_8blackwell.json"
S1_CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_v2_s1_split_geometry_hybrid_4l40.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_v2_s1_split_geometry_hybrid_lambda3e-2_20k_seed7"
REPORT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/s1_split_geometry_hybrid"
PREFLIGHT="${REPORT}/preflight_audit.json"
WORLD_SIZE=4
ACCUMULATION_PER_RANK=2
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

for path in "${PYTHON}" "${MANIFEST}" "${S0_CONFIG}" "${S1_CONFIG}"; do
    test -f "${path}"
done
test ! -e "${OUTPUT}/checkpoint_latest.pt"
mkdir -p "${OUTPUT}" "${REPORT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="$(dirname "${PYTHON}"):${PATH}"
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
test "$(wc -l < "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 4
test "$(grep -c 'L40' "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 4
"${PYTHON}" -c 'from pathlib import Path; import torch; expected=Path("/networkhome/WMGDS/zhou_c/miniconda3/envs/test"); actual=Path(torch.__file__).resolve(); assert str(actual).startswith(str(expected)); assert torch.cuda.is_available(); assert torch.cuda.device_count()==4; assert all("L40" in torch.cuda.get_device_name(i) for i in range(4)); print(torch.__version__, torch.version.cuda, actual, flush=True)'

"${PYTHON}" scripts/audit_sofa50_s1_split_geometry_preflight.py \
    --manifest "${MANIFEST}" --s0-config "${S0_CONFIG}" \
    --s1-config "${S1_CONFIG}" --output "${PREFLIGHT}" \
    --device cuda --representative-samples 3
"${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["contract_audit"]; assert p["real_model_gradient_preflight"]["passed"]; assert p["architecture"]["storage_disjoint"]; assert all(x["pcg_converged"] for x in p["solver_audit"])' "${PREFLIGHT}"

{
    echo "experiment=S1_split_geometry_architecture_only"
    echo "initialization=from_scratch_seed7_same_S0_deterministic_scheme"
    echo "fork=shared_vertex_features_before_two_independent_complete_graph_predictors"
    echo "world_size=4 accumulation_per_rank=2 effective_global_batch=8"
    echo "optimizer_steps=20000 lambda=3e-2"
    echo "loss=mean_i_squared_l2_final_hybrid_vertices_minus_clean_only"
    echo "auxiliary_laplacian_loss=false auxiliary_direct_loss=false"
    echo "checkpoint_selector=validation_final_hybrid_unified_v2_chamfer_only"
    sha256sum "${S0_CONFIG}" "${S1_CONFIG}" "${PREFLIGHT}" \
        "${REPO}/src/mlr/learned_laplacian/model.py" \
        "${REPO}/src/mlr/learned_laplacian/multi_trainer.py"
} > "${OUTPUT}/launch_contract_${SLURM_JOB_ID}.txt"

"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node="${WORLD_SIZE}" \
    --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${S1_CONFIG}" \
    --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
    --gradient-accumulation-meshes "${ACCUMULATION_PER_RANK}"
