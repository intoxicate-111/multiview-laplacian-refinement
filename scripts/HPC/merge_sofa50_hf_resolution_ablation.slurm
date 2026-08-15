#!/bin/bash

#SBATCH --job-name=merge_s50_hf1920
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST_960="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
MANIFEST_1920="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_hf_resolution_1920_20k_seed7/analysis"
SCRIPT="${REPO}/scripts/evaluate_sofa50_hf_resolution_ablation.py"

for shard in 0 1 2 3; do
    test -f "${OUTPUT}/shards/shard_${shard}.json"
    test -f "${OUTPUT}/shards/arrays_shard_${shard}.npz"
done
if [[ -f "${OUTPUT}/REPORT.md" ]]; then
    echo "Refusing to overwrite completed analysis: ${OUTPUT}" >&2
    exit 2
fi
cd "${REPO}"
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${SCRIPT}" \
        --manifest-960 "${MANIFEST_960}" \
        --manifest-1920 "${MANIFEST_1920}" \
        --output-dir "${OUTPUT}" \
        --shard-count 4 \
        --merge-shards
