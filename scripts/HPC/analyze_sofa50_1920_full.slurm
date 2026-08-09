#!/bin/bash
#SBATCH --job-name=c2f2_1920_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/c2f2_1920_eval_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/c2f2_1920_eval_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

GT_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_1920/gt_query_manifest.json"
EXPANDED_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_1920/expanded_inference_manifest.json"

# Existing C0F0/C0F1/C0F2 comparison runs.
RESOLUTION_ROOT="${REPO}/runs/learned_laplacian/sofa50_image_resolution_ablation_50000step"

# Three completed C2F2 @ 1920 runs: seed_7 / seed_17 / seed_27.
C2F2_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_1920_20000step_3seed"

OUTPUT_DIR="${REPO}/runs/learned_laplacian/sofa50_cf_c2f2_1920_comparison_full"

ANALYZER="${REPO}/scripts/analyze_sofa50_cf_c2f2_full.py"
VISIBILITY_PREP="${REPO}/scripts/prepare_renderer_visibility.py"

cd "${REPO}"
mkdir -p "${REPO}/slurm_logs" "${OUTPUT_DIR}"

# 1920/F2 inference is memory-heavy.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

printf '%s\n' \
  "============================================================" \
  "Sofa50 C2F2 1920 full prediction/recovery/render evaluation" \
  "Host:              $(hostname)" \
  "SLURM job:         ${SLURM_JOB_ID}" \
  "Analyzer:          ${ANALYZER}" \
  "Visibility prep:   ${VISIBILITY_PREP}" \
  "GT manifest:       ${GT_MANIFEST}" \
  "Expanded manifest: ${EXPANDED_MANIFEST}" \
  "Resolution root:   ${RESOLUTION_ROOT}" \
  "C2F2 1920 root:    ${C2F2_ROOT}" \
  "Output:            ${OUTPUT_DIR}" \
  "============================================================"

nvidia-smi

# ============================================================
# Preflight
# ============================================================

for FILE in \
  "${ANALYZER}" \
  "${VISIBILITY_PREP}" \
  "${GT_MANIFEST}" \
  "${EXPANDED_MANIFEST}"
do
  if [[ ! -f "${FILE}" ]]; then
    echo "ERROR: required file not found: ${FILE}" >&2
    exit 2
  fi
done

for ARM in image_resolution_f0 image_resolution_f1 image_resolution_f2; do
  ARM_DIR="${RESOLUTION_ROOT}/arms/${ARM}"
  if [[ ! -d "${ARM_DIR}" ]]; then
    echo "ERROR: resolution arm not found: ${ARM_DIR}" >&2
    exit 2
  fi
done

for SEED in 7 17 27; do
  SEED_DIR="${C2F2_ROOT}/seed_${SEED}"
  if [[ ! -d "${SEED_DIR}" ]]; then
    echo "ERROR: C2F2 1920 seed directory not found: ${SEED_DIR}" >&2
    exit 2
  fi
done

echo
echo "Preflight OK."
echo "C2F2 1920 runs:"
echo "  ${C2F2_ROOT}/seed_7"
echo "  ${C2F2_ROOT}/seed_17"
echo "  ${C2F2_ROOT}/seed_27"
echo

# ============================================================
# Prepare canonical renderer visibility for expanded validation
# samples.
#
# The existing expanded .pt files only contain legacy
# `visibility`. Current canonical recovery requires
# `visibility_backface_and_occlusion`.
#
# This uses the repository's existing renderer-visibility
# implementation. It rasterizes visibility from each expanded
# sample's own vertices/faces + cameras, does not read GT depth,
# does not use GT visibility/correspondence, and does not
# regenerate the coarse/expanded meshes.
#
# Analyzer expanded evaluation uses validation split only, so
# prepare only those five samples.
# ============================================================

echo "============================================================"
echo "Preparing renderer visibility for 1920 expanded validation set"
echo "============================================================"

PYTHONPATH="${REPO}/src" "${CONDA}" run \
  --no-capture-output \
  -n test \
  python "${VISIBILITY_PREP}" \
    --manifest "${EXPANDED_MANIFEST}" \
    --backend opengl \
    --front-face-winding ccw \
    --neighborhood-radius 1 \
    --split validation \
    --attach \
    --overwrite

# Verify the required field is now attached to every validation sample.
PYTHONPATH="${REPO}/src" "${CONDA}" run \
  --no-capture-output \
  -n test \
  python - "${EXPANDED_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

from mlr.learned_laplacian.dataset import load_prepared_sample

manifest_path = Path(sys.argv[1]).expanduser().resolve()
dataset_root = manifest_path.parent
payload = json.loads(manifest_path.read_text(encoding="utf-8"))

validation = [
    record
    for record in payload["samples"]
    if record.get("split") == "validation"
]

if not validation:
    raise RuntimeError("No validation samples found in expanded manifest.")

missing = []
for record in validation:
    sample_path = Path(record["path"])
    if not sample_path.is_absolute():
        sample_path = dataset_root / sample_path

    sample = load_prepared_sample(
        sample_path,
        materialize_images=False,
        dataset_root=dataset_root,
    )

    vis = sample.get("visibility_backface_and_occlusion")
    if vis is None:
        missing.append(str(record.get("sample_id", sample_path.name)))
        continue

    vertices = sample["vertices"]
    expected = (len(sample["image_paths"]), vertices.shape[0])
    if tuple(vis.shape) != tuple(expected):
        raise RuntimeError(
            f"{record.get('sample_id')}: visibility shape "
            f"{tuple(vis.shape)} != expected {tuple(expected)}"
        )

    print(
        f"[visibility-check] {record.get('sample_id')}: "
        f"shape={tuple(vis.shape)} visible_ratio={vis.float().mean().item():.6f}",
        flush=True,
    )

if missing:
    raise RuntimeError(
        "visibility_backface_and_occlusion is still missing for: "
        + ", ".join(missing)
    )

print(
    f"Renderer visibility verified for {len(validation)} validation samples.",
    flush=True,
)
PY

# ============================================================
# Full prediction + expanded recovery + rendering analysis
# ============================================================

echo "============================================================"
echo "Starting full C/F + C2F2 1920 analysis"
echo "============================================================"

PYTHONPATH="${REPO}/src" "${CONDA}" run \
  --no-capture-output \
  -n test \
  python "${ANALYZER}" \
    --gt-manifest "${GT_MANIFEST}" \
    --expanded-manifest "${EXPANDED_MANIFEST}" \
    --resolution-root "${RESOLUTION_ROOT}" \
    --c2f2-root "${C2F2_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda

printf '%s\n' \
  "============================================================" \
  "C2F2 1920 full evaluation complete" \
  "Report:     ${OUTPUT_DIR}/REPORT.md" \
  "Recovery:   ${OUTPUT_DIR}/expanded_comparison.csv" \
  "Per mesh:   ${OUTPUT_DIR}/expanded_per_mesh.csv" \
  "Renderings: ${OUTPUT_DIR}/renderings/cross_model_main_confidence" \
  "============================================================"
