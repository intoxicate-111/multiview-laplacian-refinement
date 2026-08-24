#!/bin/bash

#SBATCH --job-name=s50_mt_openmvs_merge
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
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
ROOT="${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v1_hf_20k_seed7"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
"${PYTHON}" scripts/evaluate_sofa50_openmvs48_multitopology_zero_shot.py \
    --manifest "${ROOT}/evaluation/openmvs48_prepared/manifest.json" \
    --output-dir "${ROOT}/evaluation/openmvs48_zero_shot" \
    --shard-count 8 \
    --merge-only
