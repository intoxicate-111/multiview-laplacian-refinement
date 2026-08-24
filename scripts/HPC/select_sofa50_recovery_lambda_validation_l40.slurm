#!/bin/bash

#SBATCH --job-name=s50_lambda_val
#SBATCH --partition=xlong
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
RUN="${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_seed7"
OUTPUT="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/lambda_validation"

export PYTHONPATH="${REPO}/src"
export OMP_NUM_THREADS=8
mkdir -p "${OUTPUT}/shards" "${REPO}/slurm_logs"
cd "${REPO}"
"${PYTHON}" scripts/select_sofa50_regularized_sparse_lambda.py \
    --manifest "${MANIFEST}" \
    --run-dir "${RUN}" \
    --output-dir "${OUTPUT}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 2 \
    --device cuda
