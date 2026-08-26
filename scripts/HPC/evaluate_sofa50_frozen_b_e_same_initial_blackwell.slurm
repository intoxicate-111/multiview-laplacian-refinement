#!/bin/bash

#SBATCH --job-name=s50_fbe_same25
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python
MANIFEST=${REPO}/runs/benchmarks/sofa50_same_initial_20260820/benchmark_manifest.json
RUN=${REPO}/runs/learned_laplacian/sofa50_v2_continuous_pretrained_b_e_hybrid_lambda3e-2_lr1e-4_20k_seed7
CHECKPOINT=${RUN}/checkpoints/checkpoint_step_000000.pt
OUTPUT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/frozen_b_e_same_initial_v00_v04
ARGS=()

if [[ -n "${INPUT_IMAGE_SIZE:-}" ]]; then
    ARGS+=(--input-image-size "${INPUT_IMAGE_SIZE}")
    OUTPUT=${OUTPUT}_${INPUT_IMAGE_SIZE}px
fi

if [[ -n "${SAMPLE_ID:-}" ]]; then
    ARGS+=(--sample-id "${SAMPLE_ID}")
    OUTPUT=${OUTPUT}/smoke
fi

test -f "${MANIFEST}"
test -f "${CHECKPOINT}"
mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
cd "${REPO}"
export PYTHONPATH=${REPO}/src:${REPO}/scripts
export PYTHONNOUSERSITE=1
unset PYTHONHOME
PYTHONHASHSEED=7 "${PYTHON}" scripts/evaluate_sofa50_frozen_b_e_same_initial.py \
    --manifest "${MANIFEST}" \
    --run "${RUN}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT}" \
    --expected-samples 25 \
    --view-chunk-size 4 \
    --device cuda \
    "${ARGS[@]}"
