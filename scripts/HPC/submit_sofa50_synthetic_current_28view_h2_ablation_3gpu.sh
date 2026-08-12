#!/usr/bin/env bash

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7"
OUTPUT="${ROOT}/analysis"
WORKER_SCRIPT="${REPO}/scripts/HPC/evaluate_sofa50_synthetic_current_28view_h2_ablation_3gpu.slurm"
MERGE_SCRIPT="${REPO}/scripts/HPC/merge_sofa50_synthetic_current_28view_h2_ablation_3gpu.slurm"

test -f "${WORKER_SCRIPT}"
test -f "${MERGE_SCRIPT}"
if [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty analysis: ${OUTPUT}" >&2
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${OUTPUT}/shards"
worker_job=$(sbatch --parsable "${WORKER_SCRIPT}")
merge_job=$(sbatch --parsable --dependency="afterok:${worker_job}" "${MERGE_SCRIPT}")

echo "worker_array=${worker_job}"
echo "merge_job=${merge_job}"
echo "output=${OUTPUT}"
