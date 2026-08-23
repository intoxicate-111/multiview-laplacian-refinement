#!/bin/bash

#SBATCH --job-name=s50_v1v2_unified
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json
ROOT=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_seed7/evaluation_v1_vs_v2_strong_smooth
cd "${REPO}"
export PYTHONPATH=${REPO}/src
export OMP_NUM_THREADS=8
"${PYTHON}" scripts/reevaluate_sofa50_multitopology_unified_geometry.py \
    --manifest "${MANIFEST}" --source-dir "${ROOT}/raw" \
    --output-dir "${ROOT}/unified" --mode standard \
    --old-arm-name v1_legacy_smoothing \
    --new-arm-name v2_strong_smoothing \
    --shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count 2
