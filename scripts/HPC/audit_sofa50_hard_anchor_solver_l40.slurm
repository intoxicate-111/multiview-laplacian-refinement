#!/bin/bash

#SBATCH --job-name=s50_hard_anchor_solver_audit
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
SOURCE_RUN="${REPO}/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7"
OUTPUT="${REPO}/runs/diagnostics/sofa50_hard_anchor_lambda0/solver_audit"
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO}"
"${PYTHON}" scripts/audit_sofa50_hard_anchor_solver.py \
  --manifest "${MANIFEST}" --arm-b-run "${SOURCE_RUN}" \
  --output-dir "${OUTPUT}" --device cuda
