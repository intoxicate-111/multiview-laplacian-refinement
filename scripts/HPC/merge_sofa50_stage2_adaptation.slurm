#!/bin/bash

#SBATCH --job-name=merge_B_s2
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
EXPERIMENT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_b_stage2_adaptation_20k_seed7"
SCRIPT="${REPO}/scripts/evaluate_sofa50_synthetic_current_stage2_adaptation.py"

for arm in continue_original continue_B_result continue_mix_50_50; do
    test -f "${EXPERIMENT}/evaluation/${arm}/arm_summary.json"
    test -f "${EXPERIMENT}/training/${arm}/metrics.json"
done
if [[ -f "${EXPERIMENT}/analysis/REPORT.md" ]]; then
    echo "Refusing to overwrite completed final report" >&2
    exit 2
fi
cd "${REPO}"
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${SCRIPT}" \
        --experiment-dir "${EXPERIMENT}" \
        --continuation-steps 20000 \
        --merge-arms
