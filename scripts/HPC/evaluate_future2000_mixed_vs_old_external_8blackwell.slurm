#!/bin/bash
#SBATCH --job-name=f2k_mix_test
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
# Keep four GPUs available for the concurrently submitted Future2000 Arm-E
# specialist training job on the same eight-GPU node.
#SBATCH --array=0-7%4
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
RUN=${PROJECT}/runs/learned_laplacian/future2000_current28view_arm_b_mixed_loss_200k_seed7_bw4_v1
ROOT=${PROJECT}/runs/benchmarks/future2000_mixed_vs_old_external_20260831_v2
BEST_SHA=fa934cd44c4009dd392c415fe2c5f731c8cf1b78cda6a31fab199d4c15510b82
INDEX=${SLURM_ARRAY_TASK_ID}

mkdir -p "${PROJECT}/slurm_logs" "${ROOT}/results"
cd "${PROJECT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=${PROJECT}/src
export PYTHONHASHSEED=7
"${PYTHON}" scripts/run_future2000_same_initial_ours.py \
    --manifest "${MANIFEST}" \
    --run-dir "${RUN}" \
    --checkpoint-name checkpoint_best.pt \
    --expected-checkpoint-sha256 "${BEST_SHA}" \
    --output-dir "${ROOT}/results" \
    --shard-index "${INDEX}" \
    --shard-count 8 \
    --expected-test-samples 1000 \
    --view-chunk-size 4 \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01 \
    --device cuda
