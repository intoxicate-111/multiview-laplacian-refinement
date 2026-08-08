#!/bin/bash

#SBATCH --job-name=c2f2_50k
#SBATCH --partition=long
#SBATCH --array=0-2
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/c2f2_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/c2f2_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"
BASE_CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"
OUTPUT_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_50000step_3seed"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

SEEDS=(7 17 27)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

RUN_DIR="${OUTPUT_ROOT}/seed_${SEED}"
CONFIG="${RUN_DIR}/launch_config.json"

mkdir -p "${RUN_DIR}"
cd "${REPO}"

python - "${BASE_CONFIG}" "${CONFIG}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
seed = int(sys.argv[3])

cfg = json.loads(src.read_text())

cfg["seed"] = seed

# C2
cfg.setdefault("image_encoder", {})["feature_dim"] = 64
cfg.setdefault("model", {})["hidden_dim"] = 256
cfg["model"]["num_graph_layers"] = 3

# F2
cfg["image_encoder"]["first_stride"] = 1
cfg["image_encoder"]["second_stride"] = 1

# residual expert OFF
cfg["model"].pop("oracle_residual_expert", None)

# Exact validation
cfg.setdefault("query_training", {})["apply_to_validation"] = False

# Full vertex training
cfg.setdefault("training", {})["vertex_sampling"] = {
    "mode": "full"
}

# 50k optimizer steps
multi = cfg.setdefault("multi_object_training", {})
multi["epochs"] = 6000
multi["max_optimizer_steps"] = 50000
multi["checkpoint_every_epochs"] = 0
multi["checkpoint_epochs"] = []

dst.write_text(json.dumps(cfg, indent=2) + "\n")

print("seed:", seed)
print("feature_dim:", cfg["image_encoder"]["feature_dim"])
print("hidden_dim:", cfg["model"]["hidden_dim"])
print("graph_layers:", cfg["model"]["num_graph_layers"])
print("strides:", cfg["image_encoder"]["first_stride"],
                  cfg["image_encoder"]["second_stride"])
print("residual expert:",
      cfg["model"].get("oracle_residual_expert", "OFF"))
PY

echo "======================================"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Seed:       ${SEED}"
echo "Host:       $(hostname)"
echo "CUDA:       ${CUDA_VISIBLE_DEVICES:-unset}"
echo "======================================"

nvidia-smi

PYTHONPATH=src "${CONDA}" run \
    --no-capture-output \
    -n test \
    python scripts/train_multi_mesh_laplacian.py \
        --manifest "${MANIFEST}" \
        --config "${CONFIG}" \
        --output-dir "${RUN_DIR}" \
        --device cuda \
        --max-optimizer-steps 50000