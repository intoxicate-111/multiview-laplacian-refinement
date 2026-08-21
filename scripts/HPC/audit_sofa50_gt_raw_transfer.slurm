#!/bin/bash

#SBATCH --job-name=s50_gt_raw_audit
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
GT_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3/gt_query_views_28_manifest.json"
CURRENT_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_gt_query_direct_raw_zero_shot_20k_seed7/contract_audit.json"

mkdir -p "${REPO}/slurm_logs" "$(dirname "${OUTPUT}")"
cd "${REPO}"
PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/audit_sofa50_gt_raw_transfer_contract.py \
        --gt-manifest "${GT_MANIFEST}" \
        --current-manifest "${CURRENT_MANIFEST}" \
        --output "${OUTPUT}"
