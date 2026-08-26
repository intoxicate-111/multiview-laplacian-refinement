#!/usr/bin/env bash
#SBATCH --job-name=s50_s1_traj
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-02,gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:L40:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-10%4
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
RUN=${REPO}/runs/learned_laplacian/sofa50_v2_s1_split_geometry_hybrid_lambda3e-2_20k_seed7
OUTPUT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/s1_split_geometry_hybrid/trajectory
LABELS=(step000000 step001000 step002500 step005000 step007500 step010000 step012500 step015000 step017500 step020000 best)
FILES=(checkpoints/checkpoint_step_000000.pt checkpoints/checkpoint_step_001000.pt checkpoints/checkpoint_step_002500.pt checkpoints/checkpoint_step_005000.pt checkpoints/checkpoint_step_007500.pt checkpoints/checkpoint_step_010000.pt checkpoints/checkpoint_step_012500.pt checkpoints/checkpoint_step_015000.pt checkpoints/checkpoint_step_017500.pt checkpoints/checkpoint_step_020000.pt checkpoint_best.pt)
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
CHECKPOINT=${RUN}/${FILES[$SLURM_ARRAY_TASK_ID]}
test -f "${CHECKPOINT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
"${PYTHON}" scripts/evaluate_sofa50_s1_checkpoint_trajectory.py \
    --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
    --run "${RUN}" --checkpoint "${CHECKPOINT}" --label "${LABEL}" \
    --output-dir "${OUTPUT}" --device cuda
