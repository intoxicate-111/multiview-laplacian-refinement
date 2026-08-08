#!/usr/bin/env bash
#SBATCH --job-name=sofa_c1f2_exp
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

# ---------- User/HPC settings ----------
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/multiview-laplacian-refinement}"
MANIFEST="${MANIFEST:-$HOME/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/learned_laplacian/sofa50_c1_f2_oracle_residual_expert_2000step}"
CONDA_ENV="${CONDA_ENV:-test}"
STEPS="${STEPS:-2000}"

BASE_CONFIG="$PROJECT_ROOT/configs/learned_laplacian/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"

# Add site-specific module commands here if your HPC requires them, e.g.
# module purge
# module load CUDA/12.x

cd "$PROJECT_ROOT"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST" >&2
    exit 2
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "Missing base config: $BASE_CONFIG" >&2
    exit 2
fi

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    echo "conda is not available. Load/activate the project Python environment before submission." >&2
    exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TMP_BASE="${SLURM_TMPDIR:-/tmp}/sofa50_c1_f2_base_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

# Build the common C1+F2 base from the canonical Sofa50 config.
# IMPORTANT: no expert is inserted here. The existing E0/E1 screening arm
# is the only place where the expert differs.
python - "$BASE_CONFIG" "$TMP_BASE" "$STEPS" <<'PY'
import copy
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
steps = int(sys.argv[3])

cfg = json.loads(src.read_text(encoding="utf-8"))

# C1 capacity
cfg.setdefault("image_encoder", {})["feature_dim"] = 32
cfg.setdefault("model", {})["hidden_dim"] = 128
cfg["model"]["num_graph_layers"] = 3

# F2 image feature resolution: 960 path in the existing screening definition.
cfg["image_encoder"]["first_stride"] = 1
cfg["image_encoder"]["second_stride"] = 1

# Keep the canonical target/confidence/query contract. The screening arm will
# force exact validation, full-vertex training, and 0.001h train support.
cfg["model"].pop("oracle_residual_expert", None)

multi = cfg.setdefault("multi_object_training", {})
multi["max_optimizer_steps"] = steps
multi["epochs"] = max(int(multi.get("epochs", 1)), steps)

cfg["c1_f2_expert_ablation"] = {
    "capacity": "C1",
    "image_feature_dim": 32,
    "hidden_dim": 128,
    "num_graph_layers": 3,
    "feature_resolution": "F2",
    "first_stride": 1,
    "second_stride": 1,
    "max_optimizer_steps": steps,
    "single_variable": "model.oracle_residual_expert",
}

dst.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
print(f"Wrote common C1+F2 base: {dst}")
PY

case "${SLURM_ARRAY_TASK_ID}" in
    0) ARM="E0" ;;
    1) ARM="E1" ;;
    *)
        echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

echo "============================================================"
echo "Sofa50 C1+F2 +/- oracle residual expert"
echo "Job:      ${SLURM_JOB_ID:-local}"
echo "Array:    ${SLURM_ARRAY_TASK_ID}"
echo "Arm:      $ARM"
echo "Steps:    $STEPS"
echo "Manifest: $MANIFEST"
echo "Output:   $OUTPUT_ROOT"
echo "============================================================"

# Existing project runner:
# E0 -> canonical C1+F2 baseline
# E1 -> same model + zero-initialized oracle top10 residual expert
python scripts/run_sofa50_oracle_residual_expert.py \
    --manifest "$MANIFEST" \
    --base-config "$TMP_BASE" \
    --output-root "$OUTPUT_ROOT" \
    --arm "$ARM" \
    --max-optimizer-steps "$STEPS" \
    --device cuda

echo "Finished $ARM"
