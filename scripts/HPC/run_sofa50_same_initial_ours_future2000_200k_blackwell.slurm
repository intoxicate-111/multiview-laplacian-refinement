#!/bin/bash
#SBATCH --job-name=s50_f2k200k_ours
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=0-01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python
MANIFEST=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820/benchmark_manifest.json
RUN=${PROJECT}/runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_hf_200k_seed7_bw7
ROOT=${PROJECT}/runs/benchmarks/sofa50_same_initial_future2000_200k_20260822
OUTPUT=${ROOT}/full/ours

test -f "${MANIFEST}"
test -f "${RUN}/checkpoint_latest.pt"
mkdir -p "${PROJECT}/slurm_logs" "${OUTPUT}"
cd "${PROJECT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH=${PROJECT}/src
PYTHONHASHSEED=7 "${PYTHON}" scripts/run_sofa50_same_initial_ours.py \
    --manifest "${MANIFEST}" \
    --run-dir "${RUN}" \
    --output-dir "${OUTPUT}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 25 \
    --view-chunk-size 4 \
    --device cuda
