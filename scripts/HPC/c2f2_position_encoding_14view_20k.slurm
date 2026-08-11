#!/bin/bash

#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "Usage: sbatch $0 2|4"
    exit 2
fi

case "$1" in
    2|4)
        NUM_FREQUENCIES="$1"
        ;;
    *)
        echo "Invalid Fourier frequency count: $1; expected 2 or 4"
        exit 2
        ;;
esac

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
SOFA_ROOT="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement"
MANIFEST="${SOFA_ROOT}/multiview_nested_14_28_56_cpu_v3/gt_query_views_14_manifest.json"
BASE_CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"
OUTPUT_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2f2_position_encoding_k2_k4_20k_seed7"
RUN_DIR="${OUTPUT_ROOT}/views14_k${NUM_FREQUENCIES}"
CONFIG="${RUN_DIR}/launch_config.json"
SEED=7
STEPS=20000

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing manifest: ${MANIFEST}"
    exit 2
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "Missing base config: ${BASE_CONFIG}"
    exit 2
fi
if [[ -d "${RUN_DIR}" && -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty run directory: ${RUN_DIR}"
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${RUN_DIR}"
cd "${REPO}"

"${CONDA}" run --no-capture-output -n test python - \
    "${BASE_CONFIG}" "${CONFIG}" "${SEED}" "${STEPS}" \
    "${NUM_FREQUENCIES}" <<'PY'
import json
import sys
from pathlib import Path

src, dst = map(Path, sys.argv[1:3])
seed = int(sys.argv[3])
steps = int(sys.argv[4])
num_frequencies = int(sys.argv[5])

cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["seed"] = seed

image_encoder = cfg.setdefault("image_encoder", {})
image_encoder["feature_dim"] = 64
image_encoder["first_stride"] = 1
image_encoder["second_stride"] = 1

model = cfg.setdefault("model", {})
model["hidden_dim"] = 256
model["num_graph_layers"] = 3
model["geometry_mode"] = "query_fourier"
model["position_encoding"] = {
    "num_frequencies": num_frequencies,
    "include_input": True,
}
model.pop("oracle_residual_expert", None)

cfg.setdefault("query_training", {})["apply_to_validation"] = False
cfg.setdefault("training", {})["vertex_sampling"] = {"mode": "full"}

multi = cfg.setdefault("multi_object_training", {})
multi["epochs"] = 6000
multi["max_optimizer_steps"] = steps
multi["checkpoint_every_epochs"] = 0
multi["checkpoint_epochs"] = []

cfg["experiment_metadata"] = {
    **cfg.get("experiment_metadata", {}),
    "experiment": "c2f2_14view_position_encoding_ablation",
    "capacity": "C2",
    "feature_resolution": "F2",
    "views": 14,
    "input_resolution": 960,
    "position_encoding_num_frequencies": num_frequencies,
    "seed": seed,
    "max_optimizer_steps": steps,
    "k6_baseline": (
        "runs/learned_laplacian/"
        "sofa50_c2f2_views_14_28_56_20k_seed7_v4/views_14"
    ),
}

dst.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PY

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "manifest=${MANIFEST}"
echo "run_dir=${RUN_DIR}"
echo "capacity=C2"
echo "feature_resolution=F2"
echo "views=14"
echo "input_resolution=960"
echo "position_encoding_num_frequencies=${NUM_FREQUENCIES}"
echo "seed=${SEED}"
echo "steps=${STEPS}"

nvidia-smi

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/train_multi_mesh_laplacian.py \
        --manifest "${MANIFEST}" \
        --config "${CONFIG}" \
        --output-dir "${RUN_DIR}" \
        --device cuda \
        --max-optimizer-steps "${STEPS}"
