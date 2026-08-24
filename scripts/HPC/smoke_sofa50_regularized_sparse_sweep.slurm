#!/bin/bash

#SBATCH --job-name=s50_reg_sp_smoke
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
SOURCE=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_seed7/evaluation_v1_vs_v2_strong_smooth/raw
OUTPUT=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_seed7/diagnostics/regularized_sparse_sweep_smoke/v2_strong_smoothing
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH=${REPO}/src
export OMP_NUM_THREADS=6
export PYTHONHASHSEED=7

"${PYTHON}" scripts/diagnose_sofa50_regularized_sparse_sweep.py \
    --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
    --prediction-source-dir "${SOURCE}" \
    --prediction-arm-name v2_strong_smoothing \
    --dataset-arm v2_strong_smoothing \
    --output-dir "${OUTPUT}" \
    --shard-index 0 --shard-count 50 --max-samples 1
