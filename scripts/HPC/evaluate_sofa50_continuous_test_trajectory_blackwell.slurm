#!/bin/bash

#SBATCH --job-name=s50_ct_test_curve
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json
RUN=${REPO}/runs/learned_laplacian/sofa50_v2_continuous_pretrained_b_e_hybrid_lambda3e-2_lr1e-4_20k_seed7
OUTPUT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/continuous_b_e_hybrid/checkpoint_diagnostics
STEPS=(000000 000100 000200 000500 001000 002000 003000 005000 007500 010000)

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
mkdir -p "${OUTPUT}"
for STEP in "${STEPS[@]}"; do
    CHECKPOINT=${RUN}/checkpoints/checkpoint_step_${STEP}.pt
    test -f "${CHECKPOINT}"
    "${PYTHON}" scripts/evaluate_sofa50_continuous_checkpoint_validation.py \
        --manifest "${MANIFEST}" \
        --run "${RUN}" \
        --checkpoint "${CHECKPOINT}" \
        --label "step${STEP}" \
        --split test \
        --output "${OUTPUT}/step${STEP}_test.json" \
        --device cuda
done
"${PYTHON}" scripts/aggregate_sofa50_continuous_test_trajectory.py \
    --input-dir "${OUTPUT}" \
    --output-dir "${OUTPUT}"
