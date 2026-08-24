#!/bin/bash

#SBATCH --job-name=s50_sparse_lambda_bw8
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
: "${ARM:?Submit with --export=ALL,ARM=C or ARM=D}"
case "${ARM}" in
  C)
    LAMBDA="1e-3"
    OUTPUT_NAME="sofa50_v2_sparse_recovery_arm_c_lambda1e-3_20k_seed7"
    ;;
  D)
    LAMBDA="1e-4"
    OUTPUT_NAME="sofa50_v2_sparse_recovery_arm_d_lambda1e-4_20k_seed7"
    ;;
  *)
    echo "ARM must be C or D, got ${ARM}" >&2
    exit 2
    ;;
esac

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
BASELINE="${REPO}/configs/learned_laplacian/train_sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_2gpu_l40.json"
OUTPUT="${REPO}/runs/learned_laplacian/${OUTPUT_NAME}"
CONFIG="${OUTPUT}/run_config.json"
WORLD_SIZE=8
MASTER_PORT="$((22000 + SLURM_JOB_ID % 18000))"

for path in "${PYTHON}" "${MANIFEST}" "${BASELINE}"; do
  test -f "${path}"
done
test ! -e "${OUTPUT}/checkpoint_latest.pt"
test ! -e "${OUTPUT}/metrics.json"
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"

export PYTHONPATH="${REPO}/src"
"${PYTHON}" "${REPO}/scripts/build_sofa50_recovery_aware_ablation_config.py" \
  --baseline "${BASELINE}" --output "${CONFIG}" --arm "${ARM}" \
  --lambda-value "${LAMBDA}" --beta 1e-2 \
  --max-optimizer-steps 20000 --run-kind final \
  --distributed-world-size 8 --effective-global-batch 8 \
  --training-gpu-model "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
  --initialization from_scratch

"${PYTHON}" - "${CONFIG}" "${ARM}" "${LAMBDA}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
arm = sys.argv[2]
expected_lambda = float(sys.argv[3])
loss = config["training"]["recovery_aware_geometry_loss"]
metadata = config["experiment_metadata"]
assert metadata["arm"] == arm
assert metadata["initialization"] == "from_scratch"
assert metadata["distributed_world_size"] == 8
assert metadata["effective_global_batch_meshes"] == 8
assert metadata["gradient_accumulation_meshes_per_rank"] == 1
assert loss["enabled"] is True
assert loss["lambda"] == expected_lambda
assert loss["beta"] == 1e-2
assert loss["maximum_iterations"] == 2048
assert loss["tolerance"] == 1e-4
assert loss["compute_dtype"] == "float64"
assert loss["runtime_diagnostics"] is True
assert config["confidence"]["enabled"] is False
assert config["recovery"]["visibility_gate"] is False
assert config["recovery"]["confidence_weighting"] is False
assert config["recovery"]["robust_loss"] is None
assert config["recovery"]["optimizer"] is None
assert config["multi_object_training"]["max_optimizer_steps"] == 20000
assert config["multi_object_training"]["gradient_accumulation_meshes"] == 1
PY

export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8
cd "${REPO}"
{
  echo "arm=${ARM} initialization=from_scratch checkpoint=none"
  echo "world_size=8 accumulation_per_rank=1 effective_global_batch=8"
  echo "lambda=${LAMBDA} beta=1e-2 recovery_loss=enabled"
  echo "pcg_compute_dtype=float64 pcg_maximum_iterations=2048 pcg_tolerance=1e-4"
  echo "numerical_change_reason=float32_pcg_stagnated_in_preflight_for_lambda_1e-3_and_1e-4"
  echo "runtime_diagnostics=enabled objective_and_solver_unchanged=true"
} > "${OUTPUT}/launch_contract_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"

"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node="${WORLD_SIZE}" \
  --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
  scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
  --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
  --gradient-accumulation-meshes 1
