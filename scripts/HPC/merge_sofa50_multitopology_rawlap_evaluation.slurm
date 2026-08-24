#!/bin/bash

#SBATCH --job-name=s50_mt_eval_merge
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# The short partition has a one-hour hard limit.  This job only merges eight
# completed evaluation shards and writes aggregate tables/reports.
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "Usage: sbatch $0 MANIFEST OUTPUT_DIR" >&2
    exit 2
fi

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
cd "${REPO}"
export PYTHONPATH="${REPO}/src"
"${PYTHON}" scripts/evaluate_sofa50_multitopology_rawlap.py \
    --manifest "$1" \
    --output-dir "$2" \
    --shard-count 8 \
    --merge-only
