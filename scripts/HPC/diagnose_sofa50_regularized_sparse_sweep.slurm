#!/bin/bash

#SBATCH --job-name=s50_reg_sparse
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=0-15%16
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
ROOT=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_seed7/diagnostics/regularized_sparse_sweep
LOCAL_SHARD=$((SLURM_ARRAY_TASK_ID % 8))

if (( SLURM_ARRAY_TASK_ID < 8 )); then
    DATASET_ARM=v1_legacy_smoothing
    MANIFEST=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1/manifest.json
    SOURCE=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v1_hf_20k_seed7/evaluation/in_domain
    PREDICTION_ARM=new_multitopology_rawlap
else
    DATASET_ARM=v2_strong_smoothing
    MANIFEST=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json
    SOURCE=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_seed7/evaluation_v1_vs_v2_strong_smooth/raw
    PREDICTION_ARM=v2_strong_smoothing
fi
OUTPUT=${ROOT}/${DATASET_ARM}

for path in "${PYTHON}" "${MANIFEST}" "${SOURCE}/recovery_per_sample.csv"; do
    test -f "${path}"
done
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH=${REPO}/src
export OMP_NUM_THREADS=6
export PYTHONHASHSEED=7

"${PYTHON}" scripts/diagnose_sofa50_regularized_sparse_sweep.py \
    --manifest "${MANIFEST}" \
    --prediction-source-dir "${SOURCE}" \
    --prediction-arm-name "${PREDICTION_ARM}" \
    --dataset-arm "${DATASET_ARM}" \
    --output-dir "${OUTPUT}" \
    --shard-index "${LOCAL_SHARD}" --shard-count 8
