#!/bin/bash
#SBATCH --job-name=f2k_hf_smoke_bw7
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:7
#SBATCH --mem=448G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json"
CONFIG="${REPO}/configs/learned_laplacian/train_future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_hf_200k_7gpu_blackwell.json"
OUTPUT="${REPO}/runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_hf_200k_seed7_bw7_smoke_${SLURM_JOB_ID}"
INPUTS="${OUTPUT}/inputs"
WORLD_SIZE=7
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export OMP_NUM_THREADS=8
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=7
export PYTHONPATH="${REPO}/src"

"${PYTHON}" scripts/prepare_future2000_training_smoke.py \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --output-dir "${INPUTS}"
"${PYTHON}" - "${INPUTS}/config_phase1.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text())
multi = config["multi_object_training"]
multi["epochs"] = 1
multi["max_optimizer_steps"] = 1
multi["checkpoint_optimizer_steps"] = [1]
path.write_text(json.dumps(config, indent=2) + "\n")
PY
"${PYTHON}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node="${WORLD_SIZE}" \
    --master-addr=127.0.0.1 \
    --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py \
        --manifest "${INPUTS}/manifest.json" \
        --config "${INPUTS}/config_phase1.json" \
        --output-dir "${OUTPUT}/run" \
        --device cuda \
        --num-workers 0 \
        --no-persistent-workers \
        --max-optimizer-steps 1 \
        --report-every-optimizer-steps 1

"${PYTHON}" - "${OUTPUT}/run" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
config = json.loads((run / "run_config.json").read_text())
metrics = json.loads((run / "metrics.json").read_text())
experiment = config["experiment_config"]
assert config["resume_checkpoint"] is None
assert config["initialization_checkpoint"] is None
assert experiment["target_mode"] == "raw_laplacian"
assert experiment["target_definition"] == "delta_target_raw=L_current@P_proxy"
assert experiment["image_encoder"]["feature_construction"]["mode"] == "original_plus_high_frequency"
assert metrics["optimizer_steps"] == 1
assert metrics["distributed_world_size"] == 7
print(json.dumps({
    "contract_audit": True,
    "initialization": "from_scratch",
    "optimizer_steps": metrics["optimizer_steps"],
    "distributed_world_size": metrics["distributed_world_size"],
    "target_mode": metrics["target_mode"],
    "feature_mode": experiment["image_encoder"]["feature_construction"]["mode"],
}, indent=2))
PY
