#!/usr/bin/env bash
#SBATCH --job-name=s50_loss_evol
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-02,gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:L40:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4%1
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
RUN=${REPO}/runs/learned_laplacian/sofa50_v2_end_to_end_hybrid_single_loss_lambda3e-2_20k_seed7
OUTPUT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/loss_mechanism_audit_v1/evolution
LABELS=(step005000 step010000 step015000 step020000 best)
FILES=(checkpoints/checkpoint_step_005000.pt checkpoints/checkpoint_step_010000.pt checkpoints/checkpoint_step_015000.pt checkpoints/checkpoint_step_020000.pt checkpoint_best.pt)
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
"${PYTHON}" scripts/analyze_sofa50_loss_mechanism_evolution.py \
    --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
    --run "${RUN}" \
    --checkpoint "${RUN}/${FILE}" \
    --label "${LABEL}" \
    --output-dir "${OUTPUT}" \
    --device cuda \
    --indices 0,5,10,15,20,25,30,35,40,45 \
    --chebyshev-order 128
