#!/bin/bash

#SBATCH --job-name=merge_s50_1920
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
SOURCE="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
OUTPUT="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1"
SCRIPT="${REPO}/scripts/prepare_sofa50_synthetic_current_28view_1920.py"

for shard in 0 1 2 3 4 5 6 7 8 9; do
    test -f "${OUTPUT}/shards/shard_${shard}.json"
done
if [[ -f "${OUTPUT}/manifest.json" ]]; then
    echo "Refusing to overwrite completed dataset: ${OUTPUT}" >&2
    exit 2
fi
cd "${REPO}"
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${SCRIPT}" \
        --source-manifest "${SOURCE}" \
        --output-root "${OUTPUT}" \
        --shard-count 10 \
        --merge-shards
