#!/bin/bash
#SBATCH --job-name=f2k_flip_merge
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
ROOT=${PROJECT}/runs/benchmarks/future2000_same_initial_full1000_20260822
OUTPUT=${ROOT}/diagnostics/flip_harmfulness

cd "${PROJECT}"
export PYTHONPATH=${PROJECT}/src
"${PYTHON}" scripts/analyze_future2000_flip_harmfulness.py \
    --ours-results "${ROOT}/results/ours" \
    --output-dir "${OUTPUT}" \
    --shard-count 8 \
    --expected-samples 1000 \
    --merge
