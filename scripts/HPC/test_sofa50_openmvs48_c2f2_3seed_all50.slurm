#!/bin/bash
#SBATCH --job-name=openmvs48_c2f2_all50
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/openmvs48_c2f2_all50_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/openmvs48_c2f2_all50_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
TEST_SCRIPT="${REPO}/scripts/test_sofa50_openmvs48_c2f2_3seed.py"

# Defaults: canonical 960 C2F2 50k three-seed model + original 14-view RGB.
# Override any of these at sbatch submission time with --export=ALL,NAME=value.
C2F2_ROOT="${C2F2_ROOT:-${REPO}/runs/learned_laplacian/sofa50_c2_f2_50000step_3seed}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json}"
COARSE_MODELS_ROOT="${COARSE_MODELS_ROOT:-/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/openmvs_texture_test_v6_48view/reconstruction/models}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/runs/learned_laplacian/sofa50_openmvs48_c2f2_3seed_all50}"
SPLIT="${SPLIT:-all}"
VISIBILITY_SIZE="${VISIBILITY_SIZE:-480}"

cd "${REPO}"
mkdir -p "${REPO}/slurm_logs" "${OUTPUT_DIR}"

printf '%s\n' \
  "============================================================" \
  "Sofa50 ALL-50 OpenMVS-48 coarse -> original-14-view C2F2 3-seed prediction" \
  "Host:               $(hostname)" \
  "SLURM job:          ${SLURM_JOB_ID}" \
  "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" \
  "Test script:        ${TEST_SCRIPT}" \
  "C2F2 root:          ${C2F2_ROOT}" \
  "Source manifest:    ${SOURCE_MANIFEST}" \
  "Coarse models:      ${COARSE_MODELS_ROOT}" \
  "Split:              ${SPLIT}" \
  "Visibility size:    ${VISIBILITY_SIZE}" \
  "Output:             ${OUTPUT_DIR}" \
  "============================================================"

# Fail early on path mistakes instead of consuming a GPU allocation.
test -f "${TEST_SCRIPT}"
test -f "${SOURCE_MANIFEST}"
test -d "${COARSE_MODELS_ROOT}"
test -d "${C2F2_ROOT}/seed_7"
test -d "${C2F2_ROOT}/seed_17"
test -d "${C2F2_ROOT}/seed_27"

nvidia-smi

# C2F2/F2 can be memory-heavy; match the existing HPC training convention.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PYTHONPATH="${REPO}/src" "${CONDA}" run \
  --no-capture-output \
  -n test \
  python "${TEST_SCRIPT}" \
    --repo-root "${REPO}" \
    --c2f2-root "${C2F2_ROOT}" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --coarse-models-root "${COARSE_MODELS_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --device cuda \
    --visibility-backend cpu \
    --visibility-size "${VISIBILITY_SIZE}" \
    --require-all

printf '%s\n' \
  "============================================================" \
  "OpenMVS48 C2F2 ALL-50 3-seed prediction complete" \
  "Summary:   ${OUTPUT_DIR}/summary.json" \
  "Per mesh:  ${OUTPUT_DIR}/per_mesh_metrics.csv" \
  "Aggregate: ${OUTPUT_DIR}/aggregate_metrics.csv" \
  "Recovered: ${OUTPUT_DIR}/recovered" \
  "============================================================"
