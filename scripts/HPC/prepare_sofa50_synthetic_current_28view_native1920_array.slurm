#!/bin/bash

#SBATCH --job-name=prep_s50_1920
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%10
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
SOURCE="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
OUTPUT="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1"
SCRIPT="${REPO}/scripts/prepare_sofa50_synthetic_current_28view_1920.py"

test -f "${SOURCE}"
test -f "${SCRIPT}"
if [[ -f "${OUTPUT}/manifest.json" ]]; then
    echo "Refusing to overwrite completed dataset: ${OUTPUT}" >&2
    exit 2
fi
mkdir -p "${REPO}/slurm_logs" "${OUTPUT}/shards"
cd "${REPO}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${SCRIPT}" \
        --source-manifest "${SOURCE}" \
        --output-root "${OUTPUT}" \
        --shard-index "${SLURM_ARRAY_TASK_ID}" \
        --shard-count 10 \
        --resume
