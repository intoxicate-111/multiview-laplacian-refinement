#!/bin/bash
#SBATCH --job-name=openmvs48_c2f2_gl
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/openmvs_coarse14_c2f2_gl480_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/openmvs_coarse14_c2f2_gl480_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

SCRIPT="${REPO}/scripts/test_sofa50_openmvs48_c2f2_3seed.py"

C2F2_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_50000step_3seed"
SOURCE_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"

COARSE_MODELS_ROOT="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/openmvs_texture_test_v6_48view/reconstruction/models"

OUTPUT_DIR="${REPO}/runs/learned_laplacian/sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480"

# Keep the same raster size as the CPU run so backend is the only intended change.
VISIBILITY_SIZE="${VISIBILITY_SIZE:-480}"

cd "${REPO}"
mkdir -p "${REPO}/slurm_logs" "${OUTPUT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

printf '%s\n' \
  "============================================================" \
  "OpenMVS coarse -> original 14-view C2F2 inference" \
  "Visibility backend: OpenGL" \
  "Host:               $(hostname)" \
  "SLURM job:          ${SLURM_JOB_ID}" \
  "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" \
  "C2F2 root:          ${C2F2_ROOT}" \
  "Source manifest:    ${SOURCE_MANIFEST}" \
  "Coarse models:      ${COARSE_MODELS_ROOT}" \
  "Visibility size:    ${VISIBILITY_SIZE}" \
  "Output:             ${OUTPUT_DIR}" \
  "============================================================"

test -f "${SCRIPT}"
test -f "${SOURCE_MANIFEST}"
test -d "${COARSE_MODELS_ROOT}"
test -d "${C2F2_ROOT}/seed_7"
test -d "${C2F2_ROOT}/seed_17"
test -d "${C2F2_ROOT}/seed_27"

MESH_COUNT="$(find "${COARSE_MODELS_ROOT}" -mindepth 2 -maxdepth 2 -name coarse.obj | wc -l)"
echo "Found coarse meshes: ${MESH_COUNT}"

nvidia-smi

echo "Starting OpenGL visibility preparation + CUDA C2F2 inference..."

PYTHONPATH="${REPO}/src" "${CONDA}" run \
  --no-capture-output \
  -n test \
  python "${SCRIPT}" \
    --repo-root "${REPO}" \
    --c2f2-root "${C2F2_ROOT}" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --coarse-models-root "${COARSE_MODELS_ROOT}" \
    --mesh-name coarse.obj \
    --output-dir "${OUTPUT_DIR}" \
    --split all \
    --device cuda \
    --visibility-backend opengl \
    --visibility-size "${VISIBILITY_SIZE}"

printf '%s\n' \
  "============================================================" \
  "Finished" \
  "Summary:   ${OUTPUT_DIR}/summary.json" \
  "Per mesh:  ${OUTPUT_DIR}/per_mesh_metrics.csv" \
  "Aggregate: ${OUTPUT_DIR}/aggregate_metrics.csv" \
  "Pred:      ${OUTPUT_DIR}/predictions" \
  "Recovered: ${OUTPUT_DIR}/recovered" \
  "============================================================"
