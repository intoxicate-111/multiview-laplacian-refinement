#!/bin/bash

#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=c2f2_current_b
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current/manifest.json"
ORACLE="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current/recovery_oracle_validation/summary.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_synthetic_current_c2f2_14view_20k.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_c2f2_14view_20k_seed7"
STEPS=20000

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing Experiment B manifest: ${MANIFEST}"
    exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing Experiment B config: ${CONFIG}"
    exit 2
fi
if [[ ! -f "${ORACLE}" ]] || ! grep -q '"all_chamfer_improved": true' "${ORACLE}"; then
    echo "Experiment B recovery oracle is missing or failed: ${ORACLE}"
    exit 2
fi
if [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty Experiment B run: ${OUTPUT}"
    exit 2
fi

mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
cd "${REPO}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "experiment=A_frozen_existing_vs_B_new_current_query"
echo "arm=B"
echo "capacity=C2"
echo "feature_resolution=F2"
echo "views=14"
echo "input_resolution=960"
echo "manifest=${MANIFEST}"
echo "recovery_oracle=${ORACLE}"
echo "config=${CONFIG}"
echo "output=${OUTPUT}"
echo "steps=${STEPS}"

nvidia-smi

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/train_multi_mesh_laplacian.py \
        --manifest "${MANIFEST}" \
        --config "${CONFIG}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --max-optimizer-steps "${STEPS}"
