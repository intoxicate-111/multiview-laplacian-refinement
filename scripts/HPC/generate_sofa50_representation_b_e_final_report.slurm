#!/bin/bash
#SBATCH --job-name=s50_be_final_report
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
ROOT="${REPO}/runs/diagnostics/sofa50_representation_b_vs_e_frozen_v1"
mkdir -p "${ROOT}/final"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
cd "${REPO}"
"${PYTHON}" scripts/generate_sofa50_representation_b_e_report.py \
  --matched-dir "${ROOT}/matched" --ood-dir "${ROOT}/ood" --visual-dir "${ROOT}/visuals" \
  --output "${ROOT}/final/FINAL_REPORT.md"
