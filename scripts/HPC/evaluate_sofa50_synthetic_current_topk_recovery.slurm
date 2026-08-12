#!/bin/bash

#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=eval_cur_topk
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
ORACLE="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_50k_downstream_evaluation_seed7/oracle_recovery_comparison"
OUTPUT="${ORACLE}/topk_prediction_error_recovery_comparison"

for path in "${ORACLE}/summary.json"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required artifact: ${path}"
        exit 2
    fi
done
if [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty output: ${OUTPUT}"
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
cd "${REPO}"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "oracle_summary=${ORACLE}/summary.json"
echo "output=${OUTPUT}"
nvidia-smi

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/evaluate_sofa50_synthetic_current_topk_recovery.py \
        --oracle-summary "${ORACLE}/summary.json" \
        --output-dir "${OUTPUT}" \
        --device cuda
