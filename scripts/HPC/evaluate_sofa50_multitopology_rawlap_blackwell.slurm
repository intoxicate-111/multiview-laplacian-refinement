#!/bin/bash

#SBATCH --job-name=s50_mt_eval
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

if [[ "$#" -ne 2 ]]; then
    echo "Usage: sbatch $0 MANIFEST OUTPUT_DIR" >&2
    exit 2
fi

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="$1"
OUTPUT="$2"
OLD_RUN="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_image_feature_ablation_20k_seed7/C_original_plus_high_frequency_2gpu"
NEW_RUN="${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v1_hf_20k_seed7"

for path in "${PYTHON}" "${MANIFEST}" "${OLD_RUN}/checkpoint_latest.pt" "${NEW_RUN}/checkpoint_latest.pt"; do
    test -e "${path}"
done
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH="${REPO}/src"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=7

"${PYTHON}" scripts/evaluate_sofa50_multitopology_rawlap.py \
    --manifest "${MANIFEST}" \
    --old-run "${OLD_RUN}" \
    --new-run "${NEW_RUN}" \
    --output-dir "${OUTPUT}" \
    --device cuda \
    --dataset-contract "${DATASET_CONTRACT:-clean_reference}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 8
