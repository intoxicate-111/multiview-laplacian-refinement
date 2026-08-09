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

# Existing F0/F1/F2 resolution-ablation runs required by the analyzer.
RESOLUTION_ROOT="${REPO}/runs/learned_laplacian/sofa50_image_resolution_ablation_50000step"

# Three completed C2F2 @ 1920 runs: seed_7 / seed_17 / seed_27.
C2F2_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_1920_20000step_3seed"

OUTPUT_DIR="${REPO}/runs/learned_laplacian/sofa50_cf_c2f2_1920_comparison_full"
ANALYZER="${REPO}/scripts/analyze_sofa50_cf_c2f2_full.py"

cd "${REPO}"
mkdir -p "${REPO}/slurm_logs" "${OUTPUT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

printf '%s\n' \
  "============================================================" \
  "Sofa50 C2F2 1920 full prediction/recovery/render evaluation" \
  "Host:              $(hostname)" \
  "SLURM job:         ${SLURM_JOB_ID}" \
  "Analyzer:          ${ANALYZER}" \
  "GT manifest:       ${GT_MANIFEST}" \
  "Expanded manifest: ${EXPANDED_MANIFEST}" \
  "Resolution root:   ${RESOLUTION_ROOT}" \
  "C2F2 1920 root:    ${C2F2_ROOT}" \
  "Output:            ${OUTPUT_DIR}" \
  "============================================================"

nvidia-smi

# Preflight checks.
if [[ ! -f "${ANALYZER}" ]]; then
    echo "ERROR: analyzer not found: ${ANALYZER}" >&2
    exit 2
fi

if [[ ! -f "${GT_MANIFEST}" ]]; then
    echo "ERROR: GT manifest not found: ${GT_MANIFEST}" >&2
    exit 2
fi

if [[ ! -f "${EXPANDED_MANIFEST}" ]]; then
    echo "ERROR: expanded manifest not found: ${EXPANDED_MANIFEST}" >&2
    exit 2
fi

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
