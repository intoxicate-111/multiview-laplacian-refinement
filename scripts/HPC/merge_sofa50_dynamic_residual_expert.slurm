#!/bin/bash

#SBATCH --job-name=merge_dynexp
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_dynamic_residual_expert_20k_seed7/analysis"

cd "${REPO}"
PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/evaluate_sofa50_dynamic_residual_expert.py \
    --manifest "${MANIFEST}" \
    --output-dir "${OUTPUT}" \
    --shard-count 4 \
    --merge-shards
