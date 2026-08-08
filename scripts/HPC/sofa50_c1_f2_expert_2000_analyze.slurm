#!/usr/bin/env bash
#SBATCH --job-name=sofa_c1f2_an
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/multiview-laplacian-refinement}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/learned_laplacian/sofa50_c1_f2_oracle_residual_expert_2000step}"
CONDA_ENV="${CONDA_ENV:-test}"
STEPS="${STEPS:-2000}"

cd "$PROJECT_ROOT"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    echo "conda is not available. Load/activate the project Python environment first." >&2
    exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

E0="$OUTPUT_ROOT/arms/oracle_expert_e0"
E1="$OUTPUT_ROOT/arms/oracle_expert_e1"

for d in "$E0" "$E1"; do
    test -f "$d/config.json"
    test -f "$d/screening_summary.json"
    test -f "$d/best.pt"
done

# Strict contract audit before analysis: C1+F2 must be identical except expert.
python - "$E0/config.json" "$E1/config.json" "$OUTPUT_ROOT" "$STEPS" <<'PY'
import copy
import json
import sys
from pathlib import Path

e0_path = Path(sys.argv[1])
e1_path = Path(sys.argv[2])
out = Path(sys.argv[3])
steps = int(sys.argv[4])

e0 = json.loads(e0_path.read_text(encoding="utf-8"))
e1 = json.loads(e1_path.read_text(encoding="utf-8"))

def assert_c1_f2(name, cfg):
    assert cfg["image_encoder"]["feature_dim"] == 32, (name, "feature_dim")
    assert cfg["image_encoder"]["first_stride"] == 1, (name, "first_stride")
    assert cfg["image_encoder"]["second_stride"] == 1, (name, "second_stride")
    assert cfg["model"]["hidden_dim"] == 128, (name, "hidden_dim")
    assert cfg["model"]["num_graph_layers"] == 3, (name, "num_graph_layers")
    assert cfg["training"]["vertex_sampling"]["mode"] == "full", (name, "vertex_sampling")
    assert cfg["query_training"]["apply_to_validation"] is False, (name, "exact validation")
    assert cfg["multi_object_training"]["max_optimizer_steps"] == steps, (name, "steps")
    assert int(cfg["seed"]) == 7, (name, "seed")

assert_c1_f2("E0", e0)
assert_c1_f2("E1", e1)

assert "oracle_residual_expert" not in e0["model"]
expert = e1["model"]["oracle_residual_expert"]
assert expert["enabled"] is True
assert expert["hidden_dim"] == 32
assert abs(float(expert["top_fraction"]) - 0.10) < 1e-12
assert expert["gate"] == "clean_gt_normalized_laplacian_top10_per_mesh"

# Verify the only intended E0/E1 differences are expert config and arm label.
a = copy.deepcopy(e0)
b = copy.deepcopy(e1)
a["model"].pop("oracle_residual_expert", None)
b["model"].pop("oracle_residual_expert", None)
a["screening"]["arm"] = "paired"
b["screening"]["arm"] = "paired"
assert a == b, "E0/E1 configs differ outside the residual expert."

audit = {
    "passed": True,
    "steps": steps,
    "seed": 7,
    "C1": {
        "image_feature_dim": 32,
        "hidden_dim": 128,
        "num_graph_layers": 3,
    },
    "F2": {
        "first_stride": 1,
        "second_stride": 1,
    },
    "E0": "C1+F2 canonical baseline",
    "E1": "C1+F2 + oracle top10 residual expert",
    "expert": {
        "shared_feature_input_dim": 128,
        "hidden_dim": 32,
        "output_dim": 3,
        "gate": "per-mesh clean-GT normalized-Laplacian top10%",
        "final_layer_zero_initialized_by_model": True,
    },
    "only_intended_difference": "model.oracle_residual_expert",
}
analysis = out / "analysis"
analysis.mkdir(parents=True, exist_ok=True)
(analysis / "c1_f2_contract_audit.json").write_text(
    json.dumps(audit, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(audit, indent=2))
PY

# Reuse the repository's existing expert analyzer.
python scripts/run_sofa50_oracle_residual_expert.py \
    --output-root "$OUTPUT_ROOT" \
    --analyze

# The current repository analyzer was originally written for the old 1000-step,
# hidden_dim=64 experiment and contains two stale descriptive strings.
# Patch metadata only; metrics are untouched.
python - "$OUTPUT_ROOT" "$STEPS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
steps = int(sys.argv[2])
summary_path = root / "analysis" / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))

e1_cfg = json.loads(
    (root / "arms" / "oracle_expert_e1" / "config.json").read_text(encoding="utf-8")
)
hidden = int(e1_cfg["model"]["hidden_dim"])
expert_hidden = int(e1_cfg["model"]["oracle_residual_expert"]["hidden_dim"])

summary["experiment"] = (
    f"Sofa50 C1+F2 oracle top10 residual expert, {steps} optimizer steps"
)
summary["expert"]["branch"] = (
    f"shared graph feature -> Linear({hidden},{expert_hidden}) -> ReLU "
    f"-> Linear({expert_hidden},3)"
)
summary["c1_f2_contract_audit"] = "analysis/c1_f2_contract_audit.json"

summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print("\n===== FINAL PAIRED RESULT =====")
print("verdict:", summary["verdict"])
print(json.dumps(summary["comparison"], indent=2))
print(json.dumps(summary["relative_changes"], indent=2))
print("top10 improved mesh count:", summary["top10_improved_mesh_count"])
print("summary:", summary_path)
PY
