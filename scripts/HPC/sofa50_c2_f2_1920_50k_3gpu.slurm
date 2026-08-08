#!/bin/bash
#SBATCH --job-name=c2f2_1920_50k
#SBATCH --partition=long
#SBATCH --array=0-2
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/c2f2_1920_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/c2f2_1920_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

# Override at submission time if your 1920 manifest lives elsewhere:
#   sbatch --export=ALL,SOFA50_1920_MANIFEST=/path/to/gt_query_manifest.json scripts/HPC/sofa50_c2_f2_1920_50k_3gpu.slurm
MANIFEST="${SOFA50_1920_MANIFEST:-/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_1920/gt_query_manifest.json}"

BASE_CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"
OUTPUT_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_1920_50000step_3seed"

SEEDS=(7 17 27)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="${OUTPUT_ROOT}/seed_${SEED}"
CONFIG="${RUN_DIR}/launch_config.json"

mkdir -p "${RUN_DIR}" "${REPO}/slurm_logs"
cd "${REPO}"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: 1920 manifest not found: ${MANIFEST}" >&2
    exit 2
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "ERROR: base config not found: ${BASE_CONFIG}" >&2
    exit 2
fi

# Reduce CUDA allocator fragmentation at 1920/F2 resolution.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python - "${BASE_CONFIG}" "${CONFIG}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
seed = int(sys.argv[3])

cfg = json.loads(src.read_text())
cfg["seed"] = seed

# C2 capacity: same as the successful 960 C2F2 experiment.
cfg.setdefault("image_encoder", {})["feature_dim"] = 64
cfg.setdefault("model", {})["hidden_dim"] = 256
cfg["model"]["num_graph_layers"] = 3

# F2 spatial feature resolution: keep both encoder strides at 1.
# With 1920 input views, the feature maps remain at 1920x1920.
cfg["image_encoder"]["first_stride"] = 1
cfg["image_encoder"]["second_stride"] = 1

# Keep the canonical non-oracle model.
cfg["model"].pop("oracle_residual_expert", None)

# Exact-query validation, matching the 960 C2F2 experiment.
cfg.setdefault("query_training", {})["apply_to_validation"] = False

# Full vertex training.
cfg.setdefault("training", {})["vertex_sampling"] = {"mode": "full"}

# Keep AMP enabled; 1920/F2 is memory-heavy on a single L40.
cfg.setdefault("training", {}).setdefault("amp", {})["enabled"] = True
cfg["training"]["amp"]["dtype"] = "float16"

# Preserve all 14 views; do not turn this into a view-count ablation.
data_loading = cfg.setdefault("data_loading", {})
data_loading["train_views_per_sample"] = None
data_loading["validation_views_per_sample"] = None

# Same optimizer-step budget as the 960 C2F2 comparison.
multi = cfg.setdefault("multi_object_training", {})
multi["epochs"] = 6000
multi["max_optimizer_steps"] = 50000
multi["gradient_accumulation_meshes"] = 4
multi["checkpoint_every_epochs"] = 0
multi["checkpoint_epochs"] = []

# Record the intended experiment resolution explicitly for audit only.
cfg["experiment_resolution"] = {
    "input_image_size": 1920,
    "feature_resolution": 1920,
    "resolution_label": "F2",
    "capacity_label": "C2",
}

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(cfg, indent=2) + "\n")

print("seed:", seed)
print("feature_dim:", cfg["image_encoder"]["feature_dim"])
print("hidden_dim:", cfg["model"]["hidden_dim"])
print("graph_layers:", cfg["model"]["num_graph_layers"])
print("strides:", cfg["image_encoder"]["first_stride"], cfg["image_encoder"]["second_stride"])
print("AMP:", cfg["training"]["amp"])
print("train_views_per_sample:", cfg["data_loading"]["train_views_per_sample"])
print("max_optimizer_steps:", cfg["multi_object_training"]["max_optimizer_steps"])
print("residual expert:", cfg["model"].get("oracle_residual_expert", "OFF"))
PY

echo "============================================================"
echo "Sofa50 C2F2 1920 / 50k"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Seed:       ${SEED}"
echo "Manifest:   ${MANIFEST}"
echo "Run dir:    ${RUN_DIR}"
echo "Host:       $(hostname)"
echo "CUDA:       ${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

nvidia-smi

PYTHONPATH="${REPO}/src" "${CONDA}" run \
    --no-capture-output \
    -n test \
    python "${REPO}/scripts/train_multi_mesh_laplacian.py" \
        --manifest "${MANIFEST}" \
        --config "${CONFIG}" \
        --output-dir "${RUN_DIR}" \
        --device cuda \
        --max-optimizer-steps 50000
