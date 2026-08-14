#!/bin/bash
#SBATCH --job-name=f2k_da3_prior
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
MANIFEST="${MANIFEST:-/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json}"
CONFIG="${REPO}/configs/baselines/future2000_external_baselines.json"
DA3_ROOT="/networkhome/WMGDS/zhou_c/external_baselines/depth-anything-3"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/future_da3/bin/python"
OUTPUT="${REPO}/runs/baselines/future2000_gt_adaptive_2000mesh_external_baselines/exmesh_da3_priors"

test -f "${MANIFEST}"
test -f "${CONFIG}"
test -x "${PYTHON}"
mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
cd "${REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME="/networkhome/WMGDS/zhou_c/.cache/huggingface"
nvidia-smi

PYTHONPATH=src "${PYTHON}" scripts/prepare_future2000_exmesh_da3_priors.py \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --da3-root "${DA3_ROOT}" \
    --output-dir "${OUTPUT}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 3 \
    --device cuda
