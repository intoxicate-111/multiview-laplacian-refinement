#!/bin/bash
#SBATCH --job-name=f2k1000_ours_bw
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
RUN=${PROJECT}/runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_hf_200k_seed7_bw7
ROOT=${PROJECT}/runs/benchmarks/future2000_same_initial_full1000_20260822
INDEX=${SLURM_ARRAY_TASK_ID}

mkdir -p "${PROJECT}/slurm_logs" "${ROOT}/results"
cd "${PROJECT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=${PROJECT}/src
export PYTHONHASHSEED=7
"${PYTHON}" scripts/run_future2000_same_initial_ours.py \
    --manifest "${MANIFEST}" \
    --run-dir "${RUN}" \
    --output-dir "${ROOT}/results" \
    --shard-index "${INDEX}" \
    --shard-count 8 \
    --expected-test-samples 1000 \
    --view-chunk-size 4 \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01 \
    --device cuda
