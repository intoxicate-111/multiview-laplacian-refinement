#!/bin/bash

#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    echo "Usage: sbatch $0 view 14|28|56 [RESUME_CHECKPOINT]"
    echo "   or: sbatch $0 query gt|gt_sub1|gt_adaptive [RESUME_CHECKPOINT]"
    echo "   or: sbatch $0 combo views_28_gt_adaptive [RESUME_CHECKPOINT]"
    exit 2
fi

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
SOFA_ROOT="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement"
BASE_CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"

ABLATION="$1"
ARM="$2"
SEED=7
STEPS=20000
RESUME_CHECKPOINT="${3:-}"

case "${ABLATION}:${ARM}" in
    view:14|view:28|view:56)
        MANIFEST="${SOFA_ROOT}/multiview_nested_14_28_56_cpu_v3/gt_query_views_${ARM}_manifest.json"
        OUTPUT_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2f2_views_14_28_56_20k_seed7_v4"
        RUN_DIR="${OUTPUT_ROOT}/views_${ARM}"
        EXPERIMENT="c2f2_nested_view_count_ablation"
        ;;
    query:gt|query:gt_sub1|query:gt_adaptive)
        MANIFEST="${SOFA_ROOT}/multiview_960/query_resolution_ablation_v2/${ARM}_manifest.json"
        OUTPUT_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2f2_query_resolution_gt_sub1_adaptive_20k_seed7_v2"
        RUN_DIR="${OUTPUT_ROOT}/${ARM}"
        EXPERIMENT="c2f2_query_graph_resolution_ablation"
        ;;
    combo:views_28_gt_adaptive)
        MANIFEST="${SOFA_ROOT}/view_query_combo_28_gt_adaptive_v1/manifest.json"
        OUTPUT_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2f2_view_query_combo_28_gt_adaptive_20k_seed7_v1"
        RUN_DIR="${OUTPUT_ROOT}/views_28_gt_adaptive"
        EXPERIMENT="c2f2_view_query_combination_ablation"
        ;;
    *)
        echo "Invalid ablation arm: ${ABLATION}:${ARM}"
        exit 2
        ;;
esac

CONFIG="${RUN_DIR}/launch_config.json"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing manifest: ${MANIFEST}"
    exit 2
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "Missing base config: ${BASE_CONFIG}"
    exit 2
fi
if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "Missing resume checkpoint: ${RESUME_CHECKPOINT}"
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${RUN_DIR}"
cd "${REPO}"

"${CONDA}" run --no-capture-output -n test python - \
    "${BASE_CONFIG}" "${CONFIG}" "${SEED}" "${STEPS}" \
    "${EXPERIMENT}" "${ABLATION}" "${ARM}" <<'PY'
import json
import sys
from pathlib import Path

src, dst = map(Path, sys.argv[1:3])
seed = int(sys.argv[3])
steps = int(sys.argv[4])
experiment, ablation, arm = sys.argv[5:8]

cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["seed"] = seed

image_encoder = cfg.setdefault("image_encoder", {})
image_encoder["feature_dim"] = 64
image_encoder["first_stride"] = 1
image_encoder["second_stride"] = 1

model = cfg.setdefault("model", {})
model["hidden_dim"] = 256
model["num_graph_layers"] = 3
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
    "experiment": experiment,
    "ablation": ablation,
    "arm": arm,
    "seed": seed,
    "max_optimizer_steps": steps,
}

dst.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PY

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "ablation=${ABLATION}"
echo "arm=${ARM}"
echo "manifest=${MANIFEST}"
echo "run_dir=${RUN_DIR}"
echo "seed=${SEED}"
echo "steps=${STEPS}"
echo "resume_checkpoint=${RESUME_CHECKPOINT:-none}"

nvidia-smi

TRAIN_ARGUMENTS=(
    --manifest "${MANIFEST}"
    --config "${CONFIG}"
    --output-dir "${RUN_DIR}"
    --device cuda
    --max-optimizer-steps "${STEPS}"
)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    TRAIN_ARGUMENTS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/train_multi_mesh_laplacian.py \
        "${TRAIN_ARGUMENTS[@]}"
