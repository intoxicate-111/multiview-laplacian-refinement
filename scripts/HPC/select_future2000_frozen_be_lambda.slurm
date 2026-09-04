#!/bin/bash
#SBATCH --job-name=f2k_be_lock
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
OUTPUT=${PROJECT}/runs/benchmarks/future2000_frozen_be_vs_baselines_20260904_v1
cd "${PROJECT}"
export PYTHONPATH=${PROJECT}/src:${PROJECT}/scripts
"${PYTHON}" scripts/select_future2000_frozen_be_lambda.py \
    --manifest "${MANIFEST}" \
    --output-dir "${OUTPUT}" \
    --shard-count 8 \
    --expected-samples 1000
test -s "${OUTPUT}/validation/lambda_lock.json"
