#!/bin/bash

#SBATCH --job-name=merge_B_s2data
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
B_RUN="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/B_direct_raw_laplacian"
EXPERIMENT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_b_stage2_adaptation_20k_seed7"
OUTPUT="${EXPERIMENT}/dataset"
SCRIPT="${REPO}/scripts/prepare_sofa50_synthetic_current_stage2_adaptation.py"

for shard in 0 1 2; do test -f "${OUTPUT}/shards/dataset_shard_${shard}.json"; done
if [[ -f "${OUTPUT}/contract_audit.json" ]]; then
    echo "Refusing to overwrite completed stage-2 dataset audit" >&2
    exit 2
fi
cd "${REPO}"
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${SCRIPT}" \
        --manifest "${MANIFEST}" \
        --b-run-dir "${B_RUN}" \
        --output-dir "${OUTPUT}" \
        --shard-count 3 \
        --merge-shards
