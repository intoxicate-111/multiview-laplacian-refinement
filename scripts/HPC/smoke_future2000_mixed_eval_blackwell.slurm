#!/bin/bash
#SBATCH --job-name=f2k_mix_smoke
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
RUN=${PROJECT}/runs/learned_laplacian/future2000_current28view_arm_b_mixed_loss_200k_seed7_bw4_v1
ROOT=${PROJECT}/runs/benchmarks/future2000_mixed_vs_old_external_20260831_smoke_v2
BEST_SHA=fa934cd44c4009dd392c415fe2c5f731c8cf1b78cda6a31fab199d4c15510b82

mkdir -p "${PROJECT}/slurm_logs" "${ROOT}/results"
cd "${PROJECT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=${PROJECT}/src
export PYTHONHASHSEED=7
SAMPLE_ID=$("${PYTHON}" -c '
import sys
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
dataset = PreparedMeshDataset.from_manifest(sys.argv[1], "test")
print(dataset.sample_ids[0])
' "${MANIFEST}")
"${PYTHON}" scripts/run_future2000_same_initial_ours.py \
    --manifest "${MANIFEST}" \
    --run-dir "${RUN}" \
    --checkpoint-name checkpoint_best.pt \
    --expected-checkpoint-sha256 "${BEST_SHA}" \
    --output-dir "${ROOT}/results" \
    --sample-id "${SAMPLE_ID}" \
    --shard-index 0 \
    --shard-count 1 \
    --expected-test-samples 1000 \
    --view-chunk-size 4 \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01 \
    --device cuda
test -s "${ROOT}/results/ours/samples/${SAMPLE_ID}/status.json"
test -s "${ROOT}/results/ours/samples/${SAMPLE_ID}/recovery/solver_audit.json"
"${PYTHON}" -c '
import json, sys
status = json.load(open(sys.argv[1]))
row = status["row"]
assert status["status"] == "completed"
assert row["formal_all_vertex_recovery"] is True
assert row["recovery_solver_all_converged"] is True
assert row["checkpoint_sha256"] == sys.argv[2]
print("future2000_mixed_eval_smoke=passed", row["sample_id"], row["refined_chamfer"])
' "${ROOT}/results/ours/samples/${SAMPLE_ID}/status.json" "${BEST_SHA}"
