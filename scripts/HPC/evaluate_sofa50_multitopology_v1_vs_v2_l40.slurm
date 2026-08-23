#!/bin/bash

#SBATCH --job-name=s50_v1v2_eval
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json
V1_RUN=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v1_hf_20k_seed7
V2_RUN=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_seed7
OUTPUT=${V2_RUN}/evaluation_v1_vs_v2_strong_smooth/raw

for path in "${PYTHON}" "${MANIFEST}" \
    "${V1_RUN}/checkpoint_latest.pt" "${V1_RUN}/metrics.json" \
    "${V2_RUN}/checkpoint_latest.pt" "${V2_RUN}/metrics.json"; do
    test -f "${path}"
done

mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH=${REPO}/src
export OMP_NUM_THREADS=8
export PYTHONHASHSEED=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON}" scripts/evaluate_sofa50_multitopology_rawlap.py \
    --manifest "${MANIFEST}" \
    --old-run "${V1_RUN}" \
    --new-run "${V2_RUN}" \
    --old-arm-name v1_legacy_smoothing \
    --new-arm-name v2_strong_smoothing \
    --output-dir "${OUTPUT}" \
    --device cuda \
    --dataset-contract clean_reference \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 2
