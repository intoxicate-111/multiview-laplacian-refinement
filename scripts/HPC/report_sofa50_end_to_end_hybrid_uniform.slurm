#!/bin/bash

#SBATCH --job-name=s50_uniform_report
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
COT_REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
RUN="${REPO}/runs/learned_laplacian/sofa50_v2_end_to_end_hybrid_single_loss_lambda3e-2_20k_seed7"
ROOT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/end_to_end_hybrid_single_loss"

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${COT_REPO}/scripts" OMP_NUM_THREADS=4 PYTHONHASHSEED=7
"${PYTHON}" scripts/report_sofa50_end_to_end_hybrid_single_loss.py \
    --evaluation-dir "${ROOT}/evaluation" --run "${RUN}" --output-dir "${ROOT}/final"
