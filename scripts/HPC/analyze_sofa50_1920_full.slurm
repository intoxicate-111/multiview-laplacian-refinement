#!/bin/bash
#SBATCH --job-name=cf_1920_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/cf_1920_eval_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/cf_1920_eval_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

GT_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_1920/gt_query_manifest.json"
EXPANDED_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_1920/expanded_inference_manifest.json"

RUNS_ROOT="${REPO}/runs/learned_laplacian"

# Optional override:
#   sbatch --export=ALL,RESOLUTION_ROOT=/absolute/path/to/your/1920/run/root analyze_sofa50_1920_full.slurm
#
# Otherwise find the unique 1920 run root containing all three resolution arms.
if [[ -z "${RESOLUTION_ROOT:-}" ]]; then
  mapfile -t CANDIDATES < <(
    find "${RUNS_ROOT}" -maxdepth 3 -type d -iname '*1920*' -print 2>/dev/null | while read -r d; do
      if [[ -d "${d}/arms/image_resolution_f0" && \
            -d "${d}/arms/image_resolution_f1" && \
            -d "${d}/arms/image_resolution_f2" ]]; then
        printf '%s\n' "${d}"
      fi
    done | sort -u
  )

  if [[ ${#CANDIDATES[@]} -ne 1 ]]; then
    echo "ERROR: expected exactly one 1920 resolution root with f0/f1/f2 arms."
    echo "Found ${#CANDIDATES[@]} candidate(s):"
    printf '  %s\n' "${CANDIDATES[@]:-}"
    echo
    echo "Submit with an explicit root, e.g.:"
    echo "  sbatch --export=ALL,RESOLUTION_ROOT=/path/to/1920/root $0"
    exit 1
  fi

  RESOLUTION_ROOT="${CANDIDATES[0]}"
fi

# Keep the existing C2F2 comparison root unless explicitly overridden.
C2F2_ROOT="${C2F2_ROOT:-${REPO}/runs/learned_laplacian/sofa50_c2_f2_50000step_3seed}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/runs/learned_laplacian/sofa50_cf_c2f2_1920_comparison_full}"
ANALYZER="${REPO}/scripts/analyze_sofa50_cf_c2f2_full.py"

cd "${REPO}"
mkdir -p "${REPO}/slurm_logs"

printf '%s\n' \
  "============================================================" \
  "Sofa50 1920 C/F + C2F2 full prediction/recovery/render evaluation" \
  "Host:              $(hostname)" \
  "SLURM job:         ${SLURM_JOB_ID}" \
  "Analyzer:          ${ANALYZER}" \
  "GT manifest:       ${GT_MANIFEST}" \
  "Expanded manifest: ${EXPANDED_MANIFEST}" \
  "1920 root:         ${RESOLUTION_ROOT}" \
  "C2F2 root:         ${C2F2_ROOT}" \
  "Output:            ${OUTPUT_DIR}" \
  "============================================================"

nvidia-smi

# Preflight: fail immediately on path mistakes.
test -f "${ANALYZER}"
test -f "${GT_MANIFEST}"
test -f "${EXPANDED_MANIFEST}"
test -d "${RESOLUTION_ROOT}/arms/image_resolution_f0"
test -d "${RESOLUTION_ROOT}/arms/image_resolution_f1"
test -d "${RESOLUTION_ROOT}/arms/image_resolution_f2"
test -d "${C2F2_ROOT}/seed_7"
test -d "${C2F2_ROOT}/seed_17"
test -d "${C2F2_ROOT}/seed_27"

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
  "1920 full evaluation complete" \
  "Report:     ${OUTPUT_DIR}/REPORT.md" \
  "Recovery:   ${OUTPUT_DIR}/expanded_comparison.csv" \
  "Per mesh:   ${OUTPUT_DIR}/expanded_per_mesh.csv" \
  "Renderings: ${OUTPUT_DIR}/renderings/cross_model_main_confidence" \
  "============================================================"
