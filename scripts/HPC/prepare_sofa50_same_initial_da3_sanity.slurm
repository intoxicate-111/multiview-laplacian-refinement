#!/bin/bash
#SBATCH --job-name=s50_same_da3
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=0-04:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
MANIFEST=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820/benchmark_manifest.json
CONFIG=${PROJECT}/configs/baselines/sofa50_same_initial_benchmark.json
DA3_ROOT=/networkhome/WMGDS/zhou_c/external_baselines/depth-anything-3
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/future_da3/bin/python
OUTPUT=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820/exmesh_da3_priors
OBJECT_ID=5ac05fe8-b550-4786-8e8c-c5a43c30d112

test -f "${MANIFEST}"
test -f "${CONFIG}"
test -x "${PYTHON}"
mkdir -p "${PROJECT}/slurm_logs" "${OUTPUT}"
cd "${PROJECT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME=/networkhome/WMGDS/zhou_c/.cache/huggingface
PYTHONPATH=src "${PYTHON}" scripts/prepare_future2000_exmesh_da3_priors.py \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --da3-root "${DA3_ROOT}" \
    --output-dir "${OUTPUT}" \
    --shard-index 0 \
    --shard-count 1 \
    --expected-test-samples 25 \
    --expected-test-objects 5 \
    --object-id "${OBJECT_ID}" \
    --device cuda
