#!/bin/bash

#SBATCH --job-name=s50_openmvs_oracle_merge
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
RUN=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v1_hf_20k_seed7
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
OUTPUT=${OUTPUT:-${RUN}/diagnostics/openmvs48_projected_gt_oracle_v2}
SHARD_COUNT=${SHARD_COUNT:-8}
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
"${PYTHON}" scripts/diagnose_sofa50_openmvs_projected_gt_oracle.py \
    --manifest "${RUN}/evaluation/openmvs48_prepared/manifest.json" \
    --recovery-source "${RUN}/evaluation/openmvs48_zero_shot" \
    --unified-source "${RUN}/evaluation_unified_v2/openmvs48_zero_shot" \
    --output-dir "${OUTPUT}" \
    --shard-count "${SHARD_COUNT}" \
    --merge-only
