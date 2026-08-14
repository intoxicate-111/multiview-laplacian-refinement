#!/bin/bash
#SBATCH --job-name=f2k_adapt_prep28
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
CONDA=/networkhome/WMGDS/zhou_c/miniconda3/bin/conda
SOURCE=/networkhome/WMGDS/zhou_c/future2000_compact/multiview_960/gt_query_manifest.json
OUTPUT=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2
UPSTREAM=/networkhome/WMGDS/zhou_c/data_prepare
NVDIFFRAST=/networkhome/WMGDS/zhou_c/dmesh2/external/nvdiffrast

mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
cd "${REPO}"
nvidia-smi
PYTHONPATH="${REPO}/src:${UPSTREAM}/src:${NVDIFFRAST}" "${CONDA}" run --no-capture-output -n test \
  python scripts/prepare_future2000_synthetic_current_28view.py \
    --source-manifest "${SOURCE}" \
    --output-root "${OUTPUT}" \
    --downstream-root "${REPO}" \
    --upstream-root "${UPSTREAM}" \
    --shard-count 3 \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --adaptive-reference sub2 \
    --adaptive-area-scale 1.0 \
    --adaptive-max-iters 12 \
    --max-vertices 1000000 \
    --visibility-backend cuda \
    --resume
