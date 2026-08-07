#!/usr/bin/env bash
set -euo pipefail

# Short capacity scaling test for Sofa50.
# Intended location in the repo: scripts/run_sofa50_capacity_2000.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/train_multi_mesh_laplacian.py" ]] && [[ -d "$SCRIPT_DIR/../src" ]]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "$SCRIPT_DIR/scripts/train_multi_mesh_laplacian.py" ]] && [[ -d "$SCRIPT_DIR/src" ]]; then
    ROOT_DIR="$SCRIPT_DIR"
else
    echo "Could not locate repository root from: $SCRIPT_DIR" >&2
    echo "Place this script in the repo root or repo/scripts/." >&2
    exit 1
fi
cd "$ROOT_DIR"

MANIFEST="${MANIFEST:-$HOME/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json}"
BASE_CONFIG="${BASE_CONFIG:-configs/learned_laplacian/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/learned_laplacian/sofa50_capacity_ablation_2000step}"
CONDA_ENV="${CONDA_ENV:-test}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-2000}"

CONFIG_DIR="$OUTPUT_ROOT/generated_configs"
mkdir -p "$CONFIG_DIR"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "Base config not found: $BASE_CONFIG" >&2
    exit 1
fi

# Build three matched configs. All arms use F2 spatial resolution (960x960
# feature maps); only image feature width and graph hidden width change.
python - "$BASE_CONFIG" "$CONFIG_DIR" "$STEPS" <<'PY'
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
steps = int(sys.argv[3])
base = json.loads(base_path.read_text(encoding="utf-8"))

arms = {
    "C0_16_64": (16, 64),
    "C1_32_128": (32, 128),
    "C2_64_256": (64, 256),
}

for arm, (image_dim, hidden_dim) in arms.items():
    cfg = copy.deepcopy(base)

    image_encoder = cfg.setdefault("image_encoder", {})
    image_encoder["feature_dim"] = image_dim
    image_encoder["first_stride"] = 1
    image_encoder["second_stride"] = 1

    model = cfg.setdefault("model", {})
    model["hidden_dim"] = hidden_dim
    # Keep graph depth fixed: this experiment changes width/capacity only.
    model["num_graph_layers"] = int(model.get("num_graph_layers", 3))

    # Match the controlled F0/F1/F2 setup as closely as possible.
    query = cfg.setdefault("query_training", {})
    query["apply_to_validation"] = False

    training = cfg.setdefault("training", {})
    training["vertex_sampling"] = {"mode": "full"}

    multi = cfg.setdefault("multi_object_training", {})
    multi["max_optimizer_steps"] = steps
    multi["epochs"] = max(int(multi.get("epochs", 1)), steps)
    multi["checkpoint_every_epochs"] = 0
    multi["checkpoint_epochs"] = []

    cfg["capacity_ablation"] = {
        "arm": arm,
        "image_feature_dim": image_dim,
        "hidden_dim": hidden_dim,
        "image_first_stride": 1,
        "image_second_stride": 1,
        "max_optimizer_steps": steps,
        "same_seed_as_base": True,
        "exact_query_validation": True,
    }

    path = out_dir / f"{arm}.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
PY

for ARM in C0_16_64 C1_32_128 C2_64_256; do
    CONFIG="$CONFIG_DIR/$ARM.json"
    OUT="$OUTPUT_ROOT/$ARM"

    if [[ -d "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then
        echo "Output directory is not empty: $OUT" >&2
        echo "Remove it or set OUTPUT_ROOT to a new directory." >&2
        exit 1
    fi

    mkdir -p "$OUT"

    echo "============================================================"
    echo "Starting $ARM: $STEPS optimizer steps"
    echo "Config: $CONFIG"
    echo "Output: $OUT"
    echo "============================================================"

    PYTHONPATH=src conda run --no-capture-output -n "$CONDA_ENV" \
        python scripts/train_multi_mesh_laplacian.py \
        --manifest "$MANIFEST" \
        --config "$CONFIG" \
        --output-dir "$OUT" \
        --device "$DEVICE" \
        --max-optimizer-steps "$STEPS" \
        2>&1 | tee "$OUT/console.log"

    echo "Finished $ARM"
done

echo "============================================================"
echo "Capacity test complete"
echo "============================================================"
echo "Results: $OUTPUT_ROOT"
echo "Compare C0/C1/C2 training_history.json and console.log."