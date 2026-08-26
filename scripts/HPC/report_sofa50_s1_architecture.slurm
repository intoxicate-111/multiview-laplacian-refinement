#!/usr/bin/env bash
#SBATCH --job-name=s50_s1_report
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
S1_ROOT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/s1_split_geometry_hybrid
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"
"${PYTHON}" scripts/report_sofa50_s1_architecture.py \
    --s1-root "${S1_ROOT}" \
    --run "${REPO}/runs/learned_laplacian/sofa50_v2_s1_split_geometry_hybrid_lambda3e-2_20k_seed7" \
    --mechanism-root "${REPO}/reports/sofa50_multitopology_rawlap500_v2/frozen_vs_joint_mechanism_analysis_v1" \
    --loss-root "${REPO}/reports/sofa50_multitopology_rawlap500_v2/loss_mechanism_audit_v1"
