#!/bin/bash

#SBATCH --job-name=s50_mech_evolution
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4%2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
RUN="${REPO}/runs/learned_laplacian/sofa50_v2_end_to_end_hybrid_single_loss_lambda3e-2_20k_seed7"
OUTPUT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/frozen_vs_joint_mechanism_analysis_v1"

case "${SLURM_ARRAY_TASK_ID}" in
    0) LABEL="step005000"; CHECKPOINT="${RUN}/checkpoints/checkpoint_step_005000.pt" ;;
    1) LABEL="step010000"; CHECKPOINT="${RUN}/checkpoints/checkpoint_step_010000.pt" ;;
    2) LABEL="step015000"; CHECKPOINT="${RUN}/checkpoints/checkpoint_step_015000.pt" ;;
    3) LABEL="step020000"; CHECKPOINT="${RUN}/checkpoints/checkpoint_step_020000.pt" ;;
    4) LABEL="best"; CHECKPOINT="${RUN}/checkpoint_best.pt" ;;
esac
test -f "${CHECKPOINT}"
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=8 PYTHONHASHSEED=7
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)'
"${PYTHON}" scripts/analyze_sofa50_joint_mechanism_evolution.py \
    --manifest "${MANIFEST}" --run "${RUN}" --checkpoint "${CHECKPOINT}" \
    --label "${LABEL}" --output-dir "${OUTPUT}" --device cuda \
    --indices 0,5,10,15,20,25,30,35,40,45 --chebyshev-order 128
