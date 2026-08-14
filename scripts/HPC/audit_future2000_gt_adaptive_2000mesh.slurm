#!/bin/bash
#SBATCH --job-name=f2k_contract
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
CONDA=/networkhome/WMGDS/zhou_c/miniconda3/bin/conda
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
CONFIG="${REPO}/configs/learned_laplacian/train_future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k.json"
ANALYSIS="${REPO}/runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k_seed7/analysis"

mkdir -p "${REPO}/slurm_logs" "${ANALYSIS}"
cd "${REPO}"
PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
  python scripts/audit_2000mesh_expanded_current_contract.py \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --output-dir "${ANALYSIS}"
