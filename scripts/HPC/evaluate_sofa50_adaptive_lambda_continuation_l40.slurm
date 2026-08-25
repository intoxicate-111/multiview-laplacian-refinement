#!/bin/bash

#SBATCH --job-name=s50_adaptive_lambda_eval
#SBATCH --partition=xlong
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
RUNS="${REPO}/runs/learned_laplacian"
OUTPUT="${REPO}/runs/diagnostics/sofa50_adaptive_lambda_continuation/final"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=8
mkdir -p "${OUTPUT}/shards" "${REPO}/slurm_logs"
cd "${REPO}"
"${PYTHON}" scripts/evaluate_sofa50_adaptive_lambda_continuation.py \
  --manifest "${MANIFEST}" --runs-root "${RUNS}" --output-dir "${OUTPUT}" \
  --arm-index "${SLURM_ARRAY_TASK_ID}" --device cuda
