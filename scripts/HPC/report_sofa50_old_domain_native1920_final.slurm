#!/bin/bash

#SBATCH --job-name=s50_old1920_report
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
REPORT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1"
RUN_B="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_arm_b_20k_seed7_v1"
RUN_E="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_arm_e_20k_seed7_v1"
RUN_CONTINUOUS="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_continuous_b_e_lr1e-4_20k_seed7_v1"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=4
cd "${REPO}"
"${PYTHON}" scripts/report_sofa50_old_domain_native1920_final.py \
  --report-root "${REPORT}" --arm-b-run "${RUN_B}" --arm-e-run "${RUN_E}" \
  --continuous-run "${RUN_CONTINUOUS}" --output "${REPORT}/FINAL_REPORT.md"
