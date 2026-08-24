#!/bin/bash

#SBATCH --job-name=s50_openmvs_oracle
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
RUN=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v1_hf_20k_seed7
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=${RUN}/evaluation/openmvs48_prepared/manifest.json
RECOVERY_SOURCE=${RUN}/evaluation/openmvs48_zero_shot
UNIFIED_SOURCE=${RUN}/evaluation_unified_v2/openmvs48_zero_shot
OUTPUT=${OUTPUT:-${RUN}/diagnostics/openmvs48_projected_gt_oracle_v2}
SHARD_COUNT=${SHARD_COUNT:-8}
PROCESSES_PER_TASK=${PROCESSES_PER_TASK:-4}

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export OMP_NUM_THREADS=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if (( SHARD_COUNT != PROCESSES_PER_TASK * 2 )); then
    echo "Expected SHARD_COUNT=2*PROCESSES_PER_TASK for the two-task array." >&2
    exit 2
fi
pids=()
for ((offset=0; offset<PROCESSES_PER_TASK; offset++)); do
    shard_index=$((SLURM_ARRAY_TASK_ID * PROCESSES_PER_TASK + offset))
    "${PYTHON}" scripts/diagnose_sofa50_openmvs_projected_gt_oracle.py \
        --manifest "${MANIFEST}" \
        --new-run "${RUN}" \
        --recovery-source "${RECOVERY_SOURCE}" \
        --unified-source "${UNIFIED_SOURCE}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --projection-chunk-size 8192 \
        --shard-index "${shard_index}" \
        --shard-count "${SHARD_COUNT}" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "${pid}"
done
