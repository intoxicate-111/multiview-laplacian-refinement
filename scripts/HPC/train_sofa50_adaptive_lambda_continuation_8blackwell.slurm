#!/bin/bash

#SBATCH --job-name=s50_adaptive_lambda_bw8
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=12:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
: "${ARM:?Submit with --export=ALL,ARM=G or ARM=H}"
case "${ARM}" in
  G) OUTPUT_NAME="sofa50_v2_g_fixed_lambda1e-2_continue5k_seed7" ;;
  H) OUTPUT_NAME="sofa50_v2_h_predicted_lambda_continue5k_seed7" ;;
  *) echo "ARM must be G or H, got ${ARM}" >&2; exit 2 ;;
esac

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
SOURCE_RUN="${REPO}/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7"
SOURCE_CONFIG="${SOURCE_RUN}/run_config.json"
SOURCE_CHECKPOINT="${SOURCE_RUN}/checkpoint_latest.pt"
ORACLE_SUMMARY="${REPO}/runs/diagnostics/sofa50_adaptive_lambda_oracle/summary.json"
OUTPUT="${REPO}/runs/learned_laplacian/${OUTPUT_NAME}"
CONFIG="${OUTPUT}/continuation_config.json"
MASTER_PORT="$((22000 + SLURM_JOB_ID % 18000))"

for path in "${PYTHON}" "${MANIFEST}" "${SOURCE_CONFIG}" "${SOURCE_CHECKPOINT}" "${ORACLE_SUMMARY}"; do
  test -f "${path}"
done
test ! -e "${OUTPUT}/checkpoint_latest.pt"
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
"${PYTHON}" - "${ORACLE_SUMMARY}" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["adaptive_training_gate"]["passed"] is True
assert summary["adaptive_training_gate"]["selection_split"] == "validation_only"
PY
"${PYTHON}" scripts/build_sofa50_adaptive_lambda_continuation_config.py \
  --arm-b-run-config "${SOURCE_CONFIG}" --output "${CONFIG}" --arm "${ARM}" \
  --continuation-steps 5000
"${PYTHON}" - "${CONFIG}" "${ARM}" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8")); arm = sys.argv[2]
r = c["training"]["recovery_aware_geometry_loss"]
h = c["model"]["recovery_lambda_head"]
assert c["multi_object_training"]["max_optimizer_steps"] == 25000
assert c["multi_object_training"]["gradient_accumulation_meshes"] == 1
assert r["beta"] == 1e-2 and r["tolerance"] == 1e-4
assert r["compute_dtype"] == "float64" and r["maximum_iterations"] == 2048
assert r["adaptive_lambda"] is (arm == "H")
assert h["enabled"] is (arm == "H")
assert h["lambda_min"] == 1e-3 and h["lambda_max"] == 1e-1
assert h["lambda_initial"] == 1e-2
assert c["experiment_metadata"]["effective_global_batch_meshes"] == 8
PY

export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8
cd "${REPO}"
sha256sum "${SOURCE_CHECKPOINT}" > "${OUTPUT}/source_checkpoint.sha256"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader \
  > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
  scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
  --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
  --gradient-accumulation-meshes 1 --resume-checkpoint "${SOURCE_CHECKPOINT}" \
  --reset-resume-tracking
