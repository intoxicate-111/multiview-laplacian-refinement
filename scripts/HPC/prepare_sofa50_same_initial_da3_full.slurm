#!/bin/bash
#SBATCH --job-name=s50_same_da3_full
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=0-04:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
MANIFEST=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820/benchmark_manifest.json
CONFIG=${PROJECT}/configs/baselines/sofa50_same_initial_benchmark.json
GATE=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820/SANITY_GATE.json
DA3_ROOT=/networkhome/WMGDS/zhou_c/external_baselines/depth-anything-3
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/future_da3/bin/python
OUTPUT=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820/full_exmesh_da3_priors

test -f "${MANIFEST}"
test -f "${CONFIG}"
test -f "${GATE}"
"${PYTHON}" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["full_benchmark_submission_allowed"] else 2)' "${GATE}"
mkdir -p "${PROJECT}/slurm_logs" "${OUTPUT}"
cd "${PROJECT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME=/networkhome/WMGDS/zhou_c/.cache/huggingface
PYTHONPATH=src "${PYTHON}" scripts/prepare_future2000_exmesh_da3_priors.py \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --da3-root "${DA3_ROOT}" \
    --output-dir "${OUTPUT}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 5 \
    --expected-test-samples 25 \
    --expected-test-objects 5 \
    --device cuda
