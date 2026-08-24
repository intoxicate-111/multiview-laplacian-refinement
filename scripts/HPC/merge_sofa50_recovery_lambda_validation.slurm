#!/bin/bash

#SBATCH --job-name=s50_lambda_merge
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
OUTPUT="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/lambda_validation"
export PYTHONPATH="${REPO}/src"
cd "${REPO}"
"${PYTHON}" scripts/select_sofa50_regularized_sparse_lambda.py \
    --output-dir "${OUTPUT}" \
    --shard-count 2 \
    --merge-only
