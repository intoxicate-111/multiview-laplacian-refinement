#!/bin/bash

#SBATCH --job-name=s50_mt_unified_v2
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
if [[ "$#" -ne 4 ]]; then
    echo "Usage: sbatch $0 MANIFEST SOURCE_DIR OUTPUT_DIR MODE" >&2
    exit 2
fi
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
cd "${REPO}"
export PYTHONPATH="${REPO}/src"
export OMP_NUM_THREADS=8
"${PYTHON}" scripts/reevaluate_sofa50_multitopology_unified_geometry.py \
    --manifest "$1" --source-dir "$2" --output-dir "$3" --mode "$4" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count 8
