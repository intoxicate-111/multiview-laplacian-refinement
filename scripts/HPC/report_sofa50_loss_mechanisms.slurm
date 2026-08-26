#!/usr/bin/env bash
#SBATCH --job-name=s50_loss_report
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
OUTPUT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/loss_mechanism_audit_v1
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"
"${PYTHON}" scripts/report_sofa50_loss_mechanisms.py \
    --audit-dir "${OUTPUT}" \
    --mechanism-report "${REPO}/reports/sofa50_multitopology_rawlap500_v2/frozen_vs_joint_mechanism_analysis_v1"
