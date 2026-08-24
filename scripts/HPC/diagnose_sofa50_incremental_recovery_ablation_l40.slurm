#!/bin/bash

#SBATCH --job-name=s50_recovery_ablate
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
ROOT=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_seed7/diagnostics/incremental_exact_target_recovery_ablation
REFERENCE_ROOT=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_seed7/diagnostics/exact_target_oracle_matched_v1_v2
LOCAL_SHARD=$((SLURM_ARRAY_TASK_ID % 4))

if (( SLURM_ARRAY_TASK_ID < 4 )); then
    DATASET_ARM=v1_legacy_smoothing
    MANIFEST=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1/manifest.json
    RUN=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v1_hf_20k_seed7
else
    DATASET_ARM=v2_strong_smoothing
    MANIFEST=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json
    RUN=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_seed7
fi
OUTPUT=${ROOT}/${DATASET_ARM}
REFERENCE=${REFERENCE_ROOT}/${DATASET_ARM}/oracle_recovery

for path in "${PYTHON}" "${MANIFEST}" "${RUN}/checkpoint_latest.pt" "${REFERENCE}"; do
    test -e "${path}"
done
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH=${REPO}/src
export OMP_NUM_THREADS=8
export PYTHONHASHSEED=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON}" scripts/diagnose_sofa50_incremental_recovery_ablation.py \
    --manifest "${MANIFEST}" \
    --run-dir "${RUN}" \
    --dataset-arm "${DATASET_ARM}" \
    --output-dir "${OUTPUT}" \
    --reference-oracle-dir "${REFERENCE}" \
    --device cuda \
    --shard-index "${LOCAL_SHARD}" \
    --shard-count 4
