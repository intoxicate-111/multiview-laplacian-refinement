#!/bin/bash

#SBATCH --job-name=s50_mt_openmvs_eval
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
ROOT="${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v1_hf_20k_seed7"
MANIFEST="${ROOT}/evaluation/openmvs48_prepared/manifest.json"
OLD_RUN="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_image_feature_ablation_20k_seed7/C_original_plus_high_frequency_2gpu"
NEW_RUN="${ROOT}"
OUTPUT="${ROOT}/evaluation/openmvs48_zero_shot"

for path in "${PYTHON}" "${MANIFEST}" "${OLD_RUN}/checkpoint_latest.pt" "${NEW_RUN}/checkpoint_latest.pt"; do test -e "${path}"; done
cd "${REPO}"
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
"${PYTHON}" scripts/evaluate_sofa50_openmvs48_multitopology_zero_shot.py \
    --manifest "${MANIFEST}" \
    --old-run "${OLD_RUN}" \
    --new-run "${NEW_RUN}" \
    --output-dir "${OUTPUT}" \
    --device cuda \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 8
