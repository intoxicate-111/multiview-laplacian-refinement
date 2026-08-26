#!/bin/bash

#SBATCH --job-name=s50_cot_select
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
PILOT_ROOT="${REPO}/runs/learned_laplacian/sofa50_v2_cotangent_lambda_pilot"
OUTPUT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/uniform_vs_cotangent_single_loss/lambda_pilot"
SELECTED="${REPO}/configs/learned_laplacian/train_sofa50_v2_cotangent_hybrid_selected_8blackwell.json"

mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="$(dirname "${PYTHON}"):${PATH}"
"${PYTHON}" scripts/select_sofa50_cotangent_lambda_pilot.py \
    --pilot-root "${PILOT_ROOT}" \
    --uniform-config configs/learned_laplacian/train_sofa50_v2_end_to_end_hybrid_single_loss_8blackwell.json \
    --selected-config "${SELECTED}" --output-dir "${OUTPUT}"
"${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["contract_audit"] and not p["test_or_ood_used"] and len(p["pilot_curve"]) == 7' "${OUTPUT}/selection.json"
