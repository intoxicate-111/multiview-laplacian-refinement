#!/bin/bash

#SBATCH --job-name=merge_cur28_h2
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
TRAIN_ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7"
A_RUN="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7/A_no_jitter"
B_RUN="${TRAIN_ROOT}/B_direct_raw_laplacian"
C_RUN="${TRAIN_ROOT}/C_normalized_output_raw_loss"
OUTPUT="${TRAIN_ROOT}/analysis"
SCRIPT="${REPO}/scripts/evaluate_sofa50_synthetic_current_28view_h2_ablation.py"

for shard in 0 1 2; do
    test -f "${OUTPUT}/shards/shard_${shard}.json"
    test -f "${OUTPUT}/shards/small_h_arrays_shard_${shard}.npz"
    test -f "${OUTPUT}/shards/small_h_per_vertex_shard_${shard}.csv"
done
if [[ -f "${OUTPUT}/REPORT.md" ]]; then
    echo "Refusing to overwrite completed analysis: ${OUTPUT}" >&2
    exit 2
fi

cd "${REPO}"
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${SCRIPT}" \
        --manifest "${MANIFEST}" \
        --arm-a-run "${A_RUN}" \
        --arm-b-run "${B_RUN}" \
        --arm-c-run "${C_RUN}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --shard-count 3 \
        --merge-shards
