#!/bin/bash
#SBATCH --job-name=c2f2_960_1920_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/c2f2_960_1920_eval_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/c2f2_960_1920_eval_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

# ============================================================
# 960 manifests
# ============================================================

GT_MANIFEST_960="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"
EXPANDED_MANIFEST_960="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/expanded_inference_manifest.json"

# ============================================================
# 1920 manifests
# ============================================================

GT_MANIFEST_1920="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_1920/gt_query_manifest.json"
EXPANDED_MANIFEST_1920="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_1920/expanded_inference_manifest.json"

# ============================================================
# Existing resolution root
#
# Kept only because analyze_sofa50_cf_c2f2_full.py currently
# requires --resolution-root.
#
# The final paired comparison below ignores C0F0/C0F1/C0F2
# and compares only C2F2@960 against C2F2@1920.
# ============================================================

RESOLUTION_ROOT="${REPO}/runs/learned_laplacian/sofa50_image_resolution_ablation_50000step"

# ============================================================
# C2F2 three-seed roots
# ============================================================

C2F2_960_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_50000step_3seed"
C2F2_1920_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_1920_20000step_3seed"

# ============================================================
# Output
# ============================================================

OUTPUT_DIR="${REPO}/runs/learned_laplacian/sofa50_c2f2_960_vs_1920_full"
OUTPUT_960="${OUTPUT_DIR}/eval_960"
OUTPUT_1920="${OUTPUT_DIR}/eval_1920"

ANALYZER="${REPO}/scripts/analyze_sofa50_cf_c2f2_full.py"
VISIBILITY_PREP="${REPO}/scripts/prepare_renderer_visibility.py"

cd "${REPO}"

mkdir -p \
  "${REPO}/slurm_logs" \
  "${OUTPUT_DIR}" \
  "${OUTPUT_960}" \
  "${OUTPUT_1920}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

printf '%s\n' \
  "============================================================" \
  "Sofa50 C2F2 960 vs 1920 paired evaluation" \
  "Host:                $(hostname)" \
  "SLURM job:           ${SLURM_JOB_ID}" \
  "Analyzer:            ${ANALYZER}" \
  "Visibility prep:     ${VISIBILITY_PREP}" \
  "C2F2 960 root:       ${C2F2_960_ROOT}" \
  "C2F2 1920 root:      ${C2F2_1920_ROOT}" \
  "Output:              ${OUTPUT_DIR}" \
  "============================================================"

nvidia-smi

# ============================================================
# Preflight
# ============================================================

for FILE in \
  "${ANALYZER}" \
  "${VISIBILITY_PREP}" \
  "${GT_MANIFEST_960}" \
  "${EXPANDED_MANIFEST_960}" \
  "${GT_MANIFEST_1920}" \
  "${EXPANDED_MANIFEST_1920}"
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
  SEED_DIR_960="${C2F2_960_ROOT}/seed_${SEED}"
  SEED_DIR_1920="${C2F2_1920_ROOT}/seed_${SEED}"

  if [[ ! -d "${SEED_DIR_960}" ]]; then
    echo "ERROR: C2F2 960 seed directory not found: ${SEED_DIR_960}" >&2
    exit 2
  fi

  if [[ ! -d "${SEED_DIR_1920}" ]]; then
    echo "ERROR: C2F2 1920 seed directory not found: ${SEED_DIR_1920}" >&2
    exit 2
  fi
done

echo
echo "Preflight OK."
echo
echo "C2F2 960 runs:"
echo "  ${C2F2_960_ROOT}/seed_7"
echo "  ${C2F2_960_ROOT}/seed_17"
echo "  ${C2F2_960_ROOT}/seed_27"
echo
echo "C2F2 1920 runs:"
echo "  ${C2F2_1920_ROOT}/seed_7"
echo "  ${C2F2_1920_ROOT}/seed_17"
echo "  ${C2F2_1920_ROOT}/seed_27"
echo

# ============================================================
# Renderer visibility preparation helper
# ============================================================

prepare_visibility () {
  local MANIFEST="$1"
  local LABEL="$2"

  echo "============================================================"
  echo "Preparing renderer visibility: ${LABEL}"
  echo "Manifest: ${MANIFEST}"
  echo "============================================================"

  PYTHONPATH="${REPO}/src" "${CONDA}" run \
    --no-capture-output \
    -n test \
    python "${VISIBILITY_PREP}" \
    --manifest "${MANIFEST}" \
    --backend opengl \
    --front-face-winding ccw \
    --neighborhood-radius 1 \
    --split validation \
    --attach \
    --overwrite

  PYTHONPATH="${REPO}/src" "${CONDA}" run \
    --no-capture-output \
    -n test \
    python - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

from mlr.learned_laplacian.dataset import load_prepared_sample

manifest_path = Path(sys.argv[1]).expanduser().resolve()
dataset_root = manifest_path.parent

payload = json.loads(
    manifest_path.read_text(encoding="utf-8")
)

validation = [
    record
    for record in payload["samples"]
    if record.get("split") == "validation"
]

if not validation:
    raise RuntimeError(
        f"No validation samples found in {manifest_path}"
    )

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

    vis = sample.get(
        "visibility_backface_and_occlusion"
    )

    if vis is None:
        missing.append(
            str(
                record.get(
                    "sample_id",
                    sample_path.name,
                )
            )
        )
        continue

    vertices = sample["vertices"]

    expected = (
        len(sample["image_paths"]),
        vertices.shape[0],
    )

    if tuple(vis.shape) != tuple(expected):
        raise RuntimeError(
            f"{record.get('sample_id')}: "
            f"visibility shape {tuple(vis.shape)} "
            f"!= expected {tuple(expected)}"
        )

    print(
        f"[visibility-check] "
        f"{record.get('sample_id')}: "
        f"shape={tuple(vis.shape)} "
        f"visible_ratio="
        f"{vis.float().mean().item():.6f}",
        flush=True,
    )

if missing:
    raise RuntimeError(
        "visibility_backface_and_occlusion "
        "is still missing for: "
        + ", ".join(missing)
    )

print(
    f"Renderer visibility verified for "
    f"{len(validation)} validation samples.",
    flush=True,
)
PY
}

# ============================================================
# Prepare visibility independently for the two resolutions
# ============================================================

prepare_visibility \
  "${EXPANDED_MANIFEST_960}" \
  "C2F2 @ 960"

prepare_visibility \
  "${EXPANDED_MANIFEST_1920}" \
  "C2F2 @ 1920"

# ============================================================
# Evaluate C2F2 @ 960
#
# Important:
# - 960 checkpoint
# - 960 GT-query manifest
# - 960 expanded inference manifest
# ============================================================

echo "============================================================"
echo "Evaluating C2F2 @ 960"
echo "============================================================"

PYTHONPATH="${REPO}/src" "${CONDA}" run \
  --no-capture-output \
  -n test \
  python "${ANALYZER}" \
  --gt-manifest "${GT_MANIFEST_960}" \
  --expanded-manifest "${EXPANDED_MANIFEST_960}" \
  --resolution-root "${RESOLUTION_ROOT}" \
  --c2f2-root "${C2F2_960_ROOT}" \
  --output-dir "${OUTPUT_960}" \
  --device cuda

# ============================================================
# Evaluate C2F2 @ 1920
#
# Important:
# - 1920 checkpoint
# - 1920 GT-query manifest
# - 1920 expanded inference manifest
# ============================================================

echo "============================================================"
echo "Evaluating C2F2 @ 1920"
echo "============================================================"

PYTHONPATH="${REPO}/src" "${CONDA}" run \
  --no-capture-output \
  -n test \
  python "${ANALYZER}" \
  --gt-manifest "${GT_MANIFEST_1920}" \
  --expanded-manifest "${EXPANDED_MANIFEST_1920}" \
  --resolution-root "${RESOLUTION_ROOT}" \
  --c2f2-root "${C2F2_1920_ROOT}" \
  --output-dir "${OUTPUT_1920}" \
  --device cuda

# ============================================================
# Build strict paired C2F2 960 vs 1920 report
#
# We intentionally ignore the Res column produced by the old
# analyzer, because the previous report showed that this field
# could be misleading for the C2F2 rows.
#
# Resolution identity here comes from which evaluation was run:
#
#   eval_960  -> C2F2 trained/evaluated at 960
#   eval_1920 -> C2F2 trained/evaluated at 1920
#
# Only C2F2_seed7 / seed17 / seed27 are compared.
# ============================================================

echo "============================================================"
echo "Building paired C2F2 960 vs 1920 report"
echo "============================================================"

"${CONDA}" run \
  --no-capture-output \
  -n test \
  python - \
  "${OUTPUT_960}/REPORT.md" \
  "${OUTPUT_1920}/REPORT.md" \
  "${OUTPUT_DIR}" <<'PY'
import csv
import statistics
import sys
from pathlib import Path


report_960_path = Path(sys.argv[1])
report_1920_path = Path(sys.argv[2])
output_dir = Path(sys.argv[3])

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


def parse_markdown_table(text, heading):
    marker = f"## {heading}"
    start = text.find(marker)

    if start < 0:
        raise RuntimeError(
            f"Heading not found: {marker}"
        )

    section = text[start + len(marker):]

    lines = section.splitlines()

    table_lines = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## "):
            break

        if stripped.startswith("|"):
            in_table = True
            table_lines.append(stripped)
        elif in_table:
            break

    if len(table_lines) < 3:
        raise RuntimeError(
            f"No Markdown table found after {marker}"
        )

    def split_row(line):
        return [
            cell.strip()
            for cell in line.strip("|").split("|")
        ]

    headers = split_row(table_lines[0])

    rows = []

    for line in table_lines[2:]:
        values = split_row(line)

        if len(values) != len(headers):
            continue

        rows.append(
            dict(zip(headers, values))
        )

    return rows


def find_col(row, prefix):
    for key in row:
        if key.startswith(prefix):
            return key

    raise KeyError(
        f"Column beginning with {prefix!r} "
        f"not found. Available: {list(row)}"
    )


def get(row, prefix):
    return row[find_col(row, prefix)]


def as_float(value):
    return float(
        str(value)
        .replace("%", "")
        .strip()
    )


def seed_from_run(name):
    name = name.strip()

    if "_seed" not in name:
        raise ValueError(
            f"Cannot extract seed from {name!r}"
        )

    return int(
        name.rsplit("_seed", 1)[1]
    )


def select_c2f2(rows):
    out = {}

    for row in rows:
        run = get(row, "Run")

        if not run.startswith("C2F2_seed"):
            continue

        seed = seed_from_run(run)
        out[seed] = row

    required = {7, 17, 27}

    if set(out) != required:
        raise RuntimeError(
            "Expected C2F2 seeds 7/17/27, "
            f"got {sorted(out)}"
        )

    return out


text_960 = report_960_path.read_text(
    encoding="utf-8"
)

text_1920 = report_1920_path.read_text(
    encoding="utf-8"
)

pred_960 = select_c2f2(
    parse_markdown_table(
        text_960,
        "Prediction comparison",
    )
)

pred_1920 = select_c2f2(
    parse_markdown_table(
        text_1920,
        "Prediction comparison",
    )
)

rec_960 = select_c2f2(
    parse_markdown_table(
        text_960,
        "Real expanded recovery (main confidence)",
    )
)

rec_1920 = select_c2f2(
    parse_markdown_table(
        text_1920,
        "Real expanded recovery (main confidence)",
    )
)

# ------------------------------------------------------------
# Prediction paired CSV
# ------------------------------------------------------------

prediction_rows = []

for seed in [7, 17, 27]:
    a = pred_960[seed]
    b = pred_1920[seed]

    row = {
        "seed": seed,
        "epe_960": as_float(get(a, "All EPE")),
        "epe_1920": as_float(get(b, "All EPE")),
        "top10_960": as_float(get(a, "Top10")),
        "top10_1920": as_float(get(b, "Top10")),
        "top1_960": as_float(get(a, "Top1")),
        "top1_1920": as_float(get(b, "Top1")),
        "cos_960": as_float(get(a, "Cos")),
        "cos_1920": as_float(get(b, "Cos")),
        "norm_960": as_float(get(a, "Norm")),
        "norm_1920": as_float(get(b, "Norm")),
        "rgb_gap_960": as_float(get(a, "RGB gap")),
        "rgb_gap_1920": as_float(get(b, "RGB gap")),
    }

    row["epe_delta_1920_minus_960"] = (
        row["epe_1920"] - row["epe_960"]
    )

    row["epe_improvement_pct"] = (
        100.0
        * (row["epe_960"] - row["epe_1920"])
        / row["epe_960"]
    )

    row["top10_delta_1920_minus_960"] = (
        row["top10_1920"] - row["top10_960"]
    )

    row["top10_improvement_pct"] = (
        100.0
        * (row["top10_960"] - row["top10_1920"])
        / row["top10_960"]
    )

    row["top1_delta_1920_minus_960"] = (
        row["top1_1920"] - row["top1_960"]
    )

    row["top1_improvement_pct"] = (
        100.0
        * (row["top1_960"] - row["top1_1920"])
        / row["top1_960"]
    )

    row["cos_delta_1920_minus_960"] = (
        row["cos_1920"] - row["cos_960"]
    )

    row["rgb_gap_delta_1920_minus_960"] = (
        row["rgb_gap_1920"]
        - row["rgb_gap_960"]
    )

    prediction_rows.append(row)


prediction_csv = (
    output_dir
    / "c2f2_960_vs_1920_prediction.csv"
)

with prediction_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            prediction_rows[0].keys()
        ),
    )
    writer.writeheader()
    writer.writerows(prediction_rows)

# ------------------------------------------------------------
# Recovery paired CSV
# ------------------------------------------------------------

recovery_rows = []

for seed in [7, 17, 27]:
    a = rec_960[seed]
    b = rec_1920[seed]

    row = {
        "seed": seed,
        "chamfer_960": as_float(
            get(a, "Refined")
        ),
        "chamfer_1920": as_float(
            get(b, "Refined")
        ),
        "p2s_960": as_float(
            get(a, "P2S")
        ),
        "p2s_1920": as_float(
            get(b, "P2S")
        ),
        "normal_960": as_float(
            get(a, "Normal")
        ),
        "normal_1920": as_float(
            get(b, "Normal")
        ),
        "flips_960": as_float(
            get(a, "Flips")
        ),
        "flips_1920": as_float(
            get(b, "Flips")
        ),
        "better_960": get(a, "Better"),
        "better_1920": get(b, "Better"),
    }

    row["chamfer_delta_1920_minus_960"] = (
        row["chamfer_1920"]
        - row["chamfer_960"]
    )

    row["chamfer_improvement_pct"] = (
        100.0
        * (
            row["chamfer_960"]
            - row["chamfer_1920"]
        )
        / row["chamfer_960"]
    )

    row["p2s_delta_1920_minus_960"] = (
        row["p2s_1920"]
        - row["p2s_960"]
    )

    row["p2s_improvement_pct"] = (
        100.0
        * (
            row["p2s_960"]
            - row["p2s_1920"]
        )
        / row["p2s_960"]
    )

    row["normal_delta_1920_minus_960"] = (
        row["normal_1920"]
        - row["normal_960"]
    )

    row["flips_delta_1920_minus_960"] = (
        row["flips_1920"]
        - row["flips_960"]
    )

    recovery_rows.append(row)


recovery_csv = (
    output_dir
    / "c2f2_960_vs_1920_recovery.csv"
)

with recovery_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            recovery_rows[0].keys()
        ),
    )
    writer.writeheader()
    writer.writerows(recovery_rows)

# ------------------------------------------------------------
# Markdown report
# ------------------------------------------------------------

def mean(rows, key):
    return statistics.mean(
        row[key]
        for row in rows
    )


lines = []

lines.append(
    "# Sofa50 C2F2: 960 vs 1920"
)
lines.append("")
lines.append(
    "Strict paired comparison using seeds "
    "7, 17, and 27."
)
lines.append("")
lines.append(
    "Each checkpoint is evaluated using the "
    "matching image-resolution manifest."
)
lines.append("")

lines.append(
    "## Prediction comparison"
)
lines.append("")
lines.append(
    "| Seed | EPE 960 | EPE 1920 | Improvement | "
    "Top10 960 | Top10 1920 | Improvement | "
    "Cos 960 | Cos 1920 | Delta |"
)
lines.append(
    "| --- | ---: | ---: | ---: | ---: | ---: | "
    "---: | ---: | ---: | ---: |"
)

for row in prediction_rows:
    lines.append(
        "| "
        f"{row['seed']} | "
        f"{row['epe_960']:.6g} | "
        f"{row['epe_1920']:.6g} | "
        f"{row['epe_improvement_pct']:+.3f}% | "
        f"{row['top10_960']:.6g} | "
        f"{row['top10_1920']:.6g} | "
        f"{row['top10_improvement_pct']:+.3f}% | "
        f"{row['cos_960']:.6g} | "
        f"{row['cos_1920']:.6g} | "
        f"{row['cos_delta_1920_minus_960']:+.6g} |"
    )

lines.append(
    "| **Mean** | "
    f"**{mean(prediction_rows, 'epe_960'):.6g}** | "
    f"**{mean(prediction_rows, 'epe_1920'):.6g}** | "
    f"**{mean(prediction_rows, 'epe_improvement_pct'):+.3f}%** | "
    f"**{mean(prediction_rows, 'top10_960'):.6g}** | "
    f"**{mean(prediction_rows, 'top10_1920'):.6g}** | "
    f"**{mean(prediction_rows, 'top10_improvement_pct'):+.3f}%** | "
    f"**{mean(prediction_rows, 'cos_960'):.6g}** | "
    f"**{mean(prediction_rows, 'cos_1920'):.6g}** | "
    f"**{mean(prediction_rows, 'cos_delta_1920_minus_960'):+.6g}** |"
)

lines.append("")
lines.append(
    "## Additional prediction metrics"
)
lines.append("")
lines.append(
    "| Seed | Top1 960 | Top1 1920 | Improvement | "
    "Norm 960 | Norm 1920 | RGB gap 960 | RGB gap 1920 |"
)
lines.append(
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)

for row in prediction_rows:
    lines.append(
        "| "
        f"{row['seed']} | "
        f"{row['top1_960']:.6g} | "
        f"{row['top1_1920']:.6g} | "
        f"{row['top1_improvement_pct']:+.3f}% | "
        f"{row['norm_960']:.6g} | "
        f"{row['norm_1920']:.6g} | "
        f"{row['rgb_gap_960']:.6g} | "
        f"{row['rgb_gap_1920']:.6g} |"
    )

lines.append("")
lines.append(
    "## Real expanded recovery"
)
lines.append("")
lines.append(
    "| Seed | Chamfer 960 | Chamfer 1920 | Improvement | "
    "P2S 960 | P2S 1920 | Improvement | "
    "Normal 960 | Normal 1920 | Flips 960 | Flips 1920 |"
)
lines.append(
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | "
    "---: | ---: | ---: | ---: |"
)

for row in recovery_rows:
    lines.append(
        "| "
        f"{row['seed']} | "
        f"{row['chamfer_960']:.8g} | "
        f"{row['chamfer_1920']:.8g} | "
        f"{row['chamfer_improvement_pct']:+.3f}% | "
        f"{row['p2s_960']:.8g} | "
        f"{row['p2s_1920']:.8g} | "
        f"{row['p2s_improvement_pct']:+.3f}% | "
        f"{row['normal_960']:.6g} | "
        f"{row['normal_1920']:.6g} | "
        f"{row['flips_960']:.0f} | "
        f"{row['flips_1920']:.0f} |"
    )

lines.append(
    "| **Mean** | "
    f"**{mean(recovery_rows, 'chamfer_960'):.8g}** | "
    f"**{mean(recovery_rows, 'chamfer_1920'):.8g}** | "
    f"**{mean(recovery_rows, 'chamfer_improvement_pct'):+.3f}%** | "
    f"**{mean(recovery_rows, 'p2s_960'):.8g}** | "
    f"**{mean(recovery_rows, 'p2s_1920'):.8g}** | "
    f"**{mean(recovery_rows, 'p2s_improvement_pct'):+.3f}%** | "
    f"**{mean(recovery_rows, 'normal_960'):.6g}** | "
    f"**{mean(recovery_rows, 'normal_1920'):.6g}** | "
    f"**{mean(recovery_rows, 'flips_960'):.1f}** | "
    f"**{mean(recovery_rows, 'flips_1920'):.1f}** |"
)

lines.append("")
lines.append(
    "## Evaluation contract"
)
lines.append("")
lines.append(
    "- Architecture is C2F2 in both arms."
)
lines.append(
    "- Seeds are paired: 7, 17, 27."
)
lines.append(
    "- 960 checkpoints use the 960 manifests."
)
lines.append(
    "- 1920 checkpoints use the 1920 manifests."
)
lines.append(
    "- Recovery implementation is unchanged."
)
lines.append(
    "- C0F0/C0F1/C0F2 rows emitted by the legacy analyzer "
    "are excluded from this paired comparison."
)
lines.append(
    "- The legacy analyzer's C2F2 `Res` field is not used "
    "to determine resolution."
)

report_path = (
    output_dir
    / "REPORT.md"
)

report_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print(
    f"Wrote paired prediction CSV: {prediction_csv}"
)
print(
    f"Wrote paired recovery CSV: {recovery_csv}"
)
print(
    f"Wrote paired report: {report_path}"
)
PY

printf '%s\n' \
  "============================================================" \
  "C2F2 960 vs 1920 evaluation complete" \
  "960 raw report:    ${OUTPUT_960}/REPORT.md" \
  "1920 raw report:   ${OUTPUT_1920}/REPORT.md" \
  "Paired report:     ${OUTPUT_DIR}/REPORT.md" \
  "Prediction CSV:    ${OUTPUT_DIR}/c2f2_960_vs_1920_prediction.csv" \
  "Recovery CSV:      ${OUTPUT_DIR}/c2f2_960_vs_1920_recovery.csv" \
  "============================================================"