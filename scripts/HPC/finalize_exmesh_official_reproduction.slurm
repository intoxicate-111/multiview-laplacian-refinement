#!/bin/bash
#SBATCH --job-name=exmesh_gate
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
EXMESH="/networkhome/WMGDS/zhou_c/external_baselines/ExMesh"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official/bin/python"
CONFIG="${PROJECT}/configs/baselines/exmesh_official_suite.json"
OUTPUT="${PROJECT}/runs/exmesh_baselines"
DTU="${EXMESH}/workdir/DTU"

cd "${PROJECT}"
mkdir -p "${OUTPUT}"

# Always produce a failure-inclusive summary first. Missing scene status files
# remain explicit failed/not-run rows rather than disappearing.
PYTHONPATH=src "${PYTHON}" scripts/aggregate_exmesh_baselines.py \
    --config "${CONFIG}" \
    --output-root "${OUTPUT}"

# A common contract is materialized only when all released ExMesh scenes and
# their official initial meshes exist.
PYTHONPATH=src "${PYTHON}" - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("runs/exmesh_baselines/summary.json").read_text())
method = summary["methods"]["exmesh_official"]
if method["successful_scenes"] != 15:
    raise SystemExit(
        f"official reproduction incomplete: {method['successful_scenes']}/15 scenes"
    )
PY

PYTHONPATH=src "${PYTHON}" scripts/extract_exmesh_common_contract.py \
    --config "${CONFIG}" \
    --dtu-root "${DTU}" \
    --output "${OUTPUT}/common_contract.json"

PYTHONPATH=src "${PYTHON}" scripts/aggregate_exmesh_baselines.py \
    --config "${CONFIG}" \
    --output-root "${OUTPUT}"

PYTHONPATH=src "${PYTHON}" - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("runs/exmesh_baselines/summary.json").read_text())
gate = summary["official_exmesh_reproduction_gate"]
print(json.dumps(gate, indent=2))
if not gate["passed"]:
    raise SystemExit("official ExMesh reproduction did not match the paper gate")
print("Phase 1 passed. Adapter/sanity work may proceed; full benchmark remains gated.")
PY
