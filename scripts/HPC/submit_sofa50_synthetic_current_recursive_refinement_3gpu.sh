#!/usr/bin/env bash

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_b_recursive_refinement_3round_seed7"
WORKER_SCRIPT="${REPO}/scripts/HPC/evaluate_sofa50_synthetic_current_recursive_refinement_3gpu.slurm"
MERGE_SCRIPT="${REPO}/scripts/HPC/merge_sofa50_synthetic_current_recursive_refinement_3gpu.slurm"

test -f "${WORKER_SCRIPT}"
test -f "${MERGE_SCRIPT}"
if [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty output: ${OUTPUT}" >&2
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${OUTPUT}/shards"
worker_job=$(sbatch --parsable "${WORKER_SCRIPT}")
merge_job=$(sbatch --parsable --dependency="afterok:${worker_job}" "${MERGE_SCRIPT}")

echo "worker_array=${worker_job}"
echo "merge_job=${merge_job}"
echo "output=${OUTPUT}"
