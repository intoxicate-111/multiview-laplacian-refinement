#!/bin/bash
#SBATCH --job-name=f2k_adapt_merge28
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
CONDA=/networkhome/WMGDS/zhou_c/miniconda3/bin/conda
OUTPUT=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2
cd "${REPO}"
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
  python scripts/prepare_future2000_synthetic_current_28view.py \
    --output-root "${OUTPUT}" --shard-count 3 --merge-only
du -sh "${OUTPUT}"
