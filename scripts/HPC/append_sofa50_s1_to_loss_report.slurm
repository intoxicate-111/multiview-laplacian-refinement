#!/usr/bin/env bash
#SBATCH --job-name=s50_s1_loss_append
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"
"${PYTHON}" scripts/append_sofa50_s1_to_loss_report.py \
    --loss-root "${REPO}/reports/sofa50_multitopology_rawlap500_v2/loss_mechanism_audit_v1" \
    --s1-root "${REPO}/reports/sofa50_multitopology_rawlap500_v2/s1_split_geometry_hybrid"
