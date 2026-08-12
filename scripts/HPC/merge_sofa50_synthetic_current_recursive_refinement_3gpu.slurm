#!/bin/bash

#SBATCH --job-name=merge_rec_b3
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
H2_ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7"
B_RUN="${H2_ROOT}/B_direct_raw_laplacian"
BASELINE="${H2_ROOT}/analysis"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_b_recursive_refinement_3round_seed7"
SCRIPT="${REPO}/scripts/evaluate_sofa50_synthetic_current_recursive_refinement.py"

for shard in 0 1 2; do
    test -f "${OUTPUT}/shards/shard_${shard}.json"
done
if [[ -f "${OUTPUT}/REPORT.md" ]]; then
    echo "Refusing to overwrite completed report: ${OUTPUT}/REPORT.md" >&2
    exit 2
fi

cd "${REPO}"
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${SCRIPT}" \
        --manifest "${MANIFEST}" \
        --run-dir "${B_RUN}" \
        --baseline-analysis-dir "${BASELINE}" \
        --output-dir "${OUTPUT}" \
        --rounds 3 \
        --shard-count 3 \
        --merge-shards
