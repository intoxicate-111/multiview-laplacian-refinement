#!/bin/bash
#SBATCH --job-name=f2k_flip_diag
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-00:59:00
#SBATCH --array=0-7%8
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
ROOT=${PROJECT}/runs/benchmarks/future2000_same_initial_full1000_20260822
OUTPUT=${ROOT}/diagnostics/flip_harmfulness

cd "${PROJECT}"
export PYTHONPATH=${PROJECT}/src
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
"${PYTHON}" scripts/analyze_future2000_flip_harmfulness.py \
    --manifest "${MANIFEST}" \
    --ours-results "${ROOT}/results/ours" \
    --output-dir "${OUTPUT}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 8 \
    --expected-samples 1000
