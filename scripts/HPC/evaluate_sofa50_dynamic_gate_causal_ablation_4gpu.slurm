#!/bin/bash

#SBATCH --job-name=gate_causal
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:L40:1
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_dynamic_residual_expert_20k_seed7"
RUN="${ROOT}/joint_from_scratch_4gpu"
SOURCE="${ROOT}/analysis"
OUTPUT="${ROOT}/gate_causal_ablation"

test -f "${RUN}/checkpoint_best.pt"
test -f "${RUN}/metrics.json"
test -f "${SOURCE}/dynamic_expert_summary.json"
test -d "${SOURCE}/per_vertex"
mkdir -p "${OUTPUT}"
cd "${REPO}"
PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/evaluate_sofa50_dynamic_gate_causal_ablation.py \
    --manifest "${MANIFEST}" \
    --expert-run "${RUN}" \
    --source-analysis "${SOURCE}" \
    --output-dir "${OUTPUT}" \
    --device cuda \
    --shuffle-seeds 7,17,27,37,47 \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 4
