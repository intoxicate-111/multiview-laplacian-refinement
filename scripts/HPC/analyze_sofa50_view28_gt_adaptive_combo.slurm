#!/bin/bash

#SBATCH --job-name=analyze_v28_adapt
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
SOFA_ROOT="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

cd "${REPO}"
"${CONDA}" run --no-capture-output -n test \
    python scripts/analyze_sofa50_view28_gt_adaptive_combo.py \
        --views28-metrics "${REPO}/runs/learned_laplacian/sofa50_c2f2_views_14_28_56_20k_seed7_v4/views_28/metrics.json" \
        --adaptive-metrics "${REPO}/runs/learned_laplacian/sofa50_c2f2_query_resolution_gt_sub1_adaptive_20k_seed7_v2/gt_adaptive/metrics.json" \
        --combo-metrics "${REPO}/runs/learned_laplacian/sofa50_c2f2_view_query_combo_28_gt_adaptive_20k_seed7_v1/views_28_gt_adaptive/metrics.json" \
        --combo-data-summary "${SOFA_ROOT}/view_query_combo_28_gt_adaptive_v1/summary.json" \
        --output-dir "${REPO}/runs/learned_laplacian/sofa50_c2f2_view_query_combo_28_gt_adaptive_20k_seed7_v1/analysis"
