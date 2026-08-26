#!/bin/bash

#SBATCH --job-name=s50_u_c_report
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent"
MAIN_REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
ROOT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/uniform_vs_cotangent_single_loss"
U_RUN="${MAIN_REPO}/runs/learned_laplacian/sofa50_v2_end_to_end_hybrid_single_loss_lambda3e-2_20k_seed7"
C_RUN="${REPO}/runs/learned_laplacian/sofa50_v2_cotangent_hybrid_selected_20k_seed7"

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=8 PYTHONHASHSEED=7
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="$(dirname "${PYTHON}"):${PATH}"
"${PYTHON}" scripts/report_sofa50_uniform_vs_cotangent_hybrid.py \
    --evaluation-dir "${ROOT}/evaluation" \
    --operator-audit-dir "${ROOT}/operator_audit" \
    --lambda-pilot-dir "${ROOT}/lambda_pilot" \
    --uniform-run "${U_RUN}" --cotangent-run "${C_RUN}" \
    --output-dir "${ROOT}/final"
