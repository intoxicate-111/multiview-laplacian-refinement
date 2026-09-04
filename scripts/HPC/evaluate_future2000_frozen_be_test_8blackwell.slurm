#!/bin/bash
#SBATCH --job-name=f2k_be_test
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --array=0-7%4
# Eight deterministic shards; at most four one-GPU tasks run concurrently.
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
B_RUN=${PROJECT}/runs/learned_laplacian/future2000_current28view_arm_b_mixed_loss_200k_seed7_bw4_v1
E_RUN=${PROJECT}/runs/learned_laplacian/future2000_current28view_arm_e_200k_seed7_gb8_v1
OUTPUT=${PROJECT}/runs/benchmarks/future2000_frozen_be_vs_baselines_20260904_v1
B_SHA=fa934cd44c4009dd392c415fe2c5f731c8cf1b78cda6a31fab199d4c15510b82
E_SHA=5a6aaa32bec6edcdd2c30face02c4ae8bc139fef18d4d05b3394c987057cb50f

mkdir -p "${PROJECT}/slurm_logs" "${OUTPUT}"
cd "${PROJECT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=${PROJECT}/src:${PROJECT}/scripts
export PYTHONHASHSEED=7
"${PYTHON}" scripts/evaluate_future2000_frozen_be.py \
    --manifest "${MANIFEST}" \
    --arm-b-run "${B_RUN}" \
    --arm-e-run "${E_RUN}" \
    --arm-b-checkpoint-sha256 "${B_SHA}" \
    --arm-e-checkpoint-sha256 "${E_SHA}" \
    --output-dir "${OUTPUT}" \
    --split test \
    --lambda-lock "${OUTPUT}/validation/lambda_lock.json" \
    --expected-samples 1000 \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 8 \
    --view-chunk-size 4 \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01 \
    --device cuda
