#!/usr/bin/env bash

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_dynamic_residual_expert_20k_seed7"
OUTPUT="${ROOT}/gate_causal_ablation"
EVAL_SCRIPT="${REPO}/scripts/HPC/evaluate_sofa50_dynamic_gate_causal_ablation_4gpu.slurm"
MERGE_SCRIPT="${REPO}/scripts/HPC/merge_sofa50_dynamic_gate_causal_ablation.slurm"

test -f "${EVAL_SCRIPT}"
test -f "${MERGE_SCRIPT}"
if [[ -e "${OUTPUT}" ]]; then
    echo "Refusing to overwrite existing gate causal-ablation output: ${OUTPUT}" >&2
    exit 2
fi

evaluation_job=$(sbatch --parsable "${EVAL_SCRIPT}")
merge_job=$(sbatch --parsable --dependency="afterok:${evaluation_job}" "${MERGE_SCRIPT}")

echo "evaluation_array=${evaluation_job}"
echo "merge_job=${merge_job}"
echo "output=${OUTPUT}"
