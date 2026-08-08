#!/bin/bash
#SBATCH --job-name=cf_full_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/cf_full_eval_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/cf_full_eval_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

GT_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"
EXPANDED_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/expanded_inference_manifest.json"
RESOLUTION_ROOT="${REPO}/runs/learned_laplacian/sofa50_image_resolution_ablation_50000step"
C2F2_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_50000step_3seed"
OUTPUT_DIR="${REPO}/runs/learned_laplacian/sofa50_cf_c2f2_comparison_full"
ANALYZER="${REPO}/scripts/analyze_sofa50_cf_c2f2_full.py"

cd "${REPO}"
mkdir -p "${REPO}/slurm_logs"

printf '%s\n' \
  "============================================================" \
  "Sofa50 C/F + C2F2 full prediction/recovery/render evaluation" \
  "Host:              $(hostname)" \
  "SLURM job:         ${SLURM_JOB_ID}" \
  "Analyzer:          ${ANALYZER}" \
  "GT manifest:       ${GT_MANIFEST}" \
  "Expanded manifest: ${EXPANDED_MANIFEST}" \
  "Resolution root:   ${RESOLUTION_ROOT}" \
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
  "Full evaluation complete" \
  "Report:    ${OUTPUT_DIR}/REPORT.md" \
  "Recovery:  ${OUTPUT_DIR}/expanded_comparison.csv" \
  "Per mesh:  ${OUTPUT_DIR}/expanded_per_mesh.csv" \
  "Renderings:${OUTPUT_DIR}/renderings/cross_model_main_confidence" \
  "============================================================"
