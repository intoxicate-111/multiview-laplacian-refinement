#!/bin/bash

#SBATCH --job-name=s50_diffsolve_audit
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
OUTPUT="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/differentiable_solve_audit.json"

export PYTHONPATH="${REPO}/src"
mkdir -p "$(dirname "${OUTPUT}")" "${REPO}/slurm_logs"
cd "${REPO}"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
"${PYTHON}" scripts/audit_sofa50_differentiable_sparse_recovery.py \
    --manifest "${MANIFEST}" \
    --output "${OUTPUT}" \
    --split validation \
    --sample-index 0 \
    --lambda-value 1e-2 \
    --maximum-iterations 256 \
    --tolerance 1e-4 \
    --device cuda
