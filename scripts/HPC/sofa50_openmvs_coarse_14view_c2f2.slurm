#!/bin/bash
#SBATCH --job-name=openmvs48_c2f2
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/openmvs48_c2f2_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/openmvs48_c2f2_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

C2F2_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_50000step_3seed"

SOURCE_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"

COARSE_MODELS_ROOT="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/openmvs_texture_test_v6_48view/reconstruction/models"

OUTPUT_DIR="${REPO}/runs/learned_laplacian/sofa50_openmvs48_c2f2_3seed_48mesh"

SCRIPT="${REPO}/scripts/test_sofa50_openmvs48_c2f2_3seed.py"

cd "${REPO}"

mkdir -p "${REPO}/slurm_logs" "${OUTPUT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

nvidia-smi

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
    --visibility-backend cpu \
    --visibility-size 480

echo "============================================================"
echo "Finished"
echo "Summary:   ${OUTPUT_DIR}/summary.json"
echo "Per mesh:  ${OUTPUT_DIR}/per_mesh_metrics.csv"
echo "Aggregate: ${OUTPUT_DIR}/aggregate_metrics.csv"
echo "Pred:      ${OUTPUT_DIR}/predictions"
echo "Recovered: ${OUTPUT_DIR}/recovered"
echo "============================================================"