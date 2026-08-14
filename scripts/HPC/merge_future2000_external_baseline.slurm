#!/bin/bash
#SBATCH --job-name=f2k_ext_merge
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

METHOD="${1:?usage: sbatch merge_future2000_external_baseline.slurm METHOD}"
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
MANIFEST="${MANIFEST:-/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json}"
OUTPUT="${REPO}/runs/baselines/future2000_gt_adaptive_2000mesh_external_baselines"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"

cd "${REPO}"
PYTHONPATH=src "${PYTHON}" scripts/merge_future2000_external_baseline.py \
    --manifest "${MANIFEST}" \
    --output-dir "${OUTPUT}" \
    --method "${METHOD}" \
    --shard-count 3
