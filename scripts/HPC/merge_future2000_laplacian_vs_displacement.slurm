#!/bin/bash
#SBATCH --job-name=f2k_merge_eval
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="${MANIFEST:-/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json}"
LAPLACIAN_RUN="${LAPLACIAN_RUN:-${REPO}/runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k_seed7}"
OUTPUT="${OUTPUT:-${LAPLACIAN_RUN}/analysis}"

for shard in 0 1 2; do
    test -f "${OUTPUT}/shards/per_sample_shard_$(printf '%03d' "${shard}").csv"
    test -f "${OUTPUT}/shards/metadata_shard_$(printf '%03d' "${shard}").json"
done
if [[ -f "${OUTPUT}/aggregate.json" ]]; then
    echo "Refusing to overwrite completed aggregate: ${OUTPUT}/aggregate.json" >&2
    exit 2
fi

cd "${REPO}"
PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/merge_future2000_learned_evaluation.py \
        --manifest "${MANIFEST}" \
        --output-dir "${OUTPUT}" \
        --shard-count 3
