#!/usr/bin/env bash

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
EXPERIMENT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_b_stage2_adaptation_20k_seed7"
PREP="${REPO}/scripts/HPC/prepare_sofa50_stage2_dataset_3gpu.slurm"
MERGE_DATA="${REPO}/scripts/HPC/merge_sofa50_stage2_dataset.slurm"
TRAIN="${REPO}/scripts/HPC/train_evaluate_sofa50_stage2_adaptation_3gpu.slurm"
MERGE_RESULT="${REPO}/scripts/HPC/merge_sofa50_stage2_adaptation.slurm"

for path in "${PREP}" "${MERGE_DATA}" "${TRAIN}" "${MERGE_RESULT}"; do test -f "${path}"; done
if [[ -d "${EXPERIMENT}" && -n "$(find "${EXPERIMENT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty experiment directory: ${EXPERIMENT}" >&2
    exit 2
fi
mkdir -p "${REPO}/slurm_logs" "${EXPERIMENT}/dataset/shards"
prep_job=$(sbatch --parsable "${PREP}")
data_merge_job=$(sbatch --parsable --dependency="afterok:${prep_job}" "${MERGE_DATA}")
train_job=$(sbatch --parsable --dependency="afterok:${data_merge_job}" "${TRAIN}")
result_merge_job=$(sbatch --parsable --dependency="afterok:${train_job}" "${MERGE_RESULT}")

echo "dataset_worker_array=${prep_job}"
echo "dataset_merge_job=${data_merge_job}"
echo "training_evaluation_array=${train_job}"
echo "result_merge_job=${result_merge_job}"
echo "output=${EXPERIMENT}"
