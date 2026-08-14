#!/bin/bash
#SBATCH --job-name=f2k_final_report
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
MANIFEST="/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json"
LAP="${REPO}/runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k_seed7"
DISP="${REPO}/runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_displacement_20k_seed7"
LEARNED="${LAP}/analysis"
EXTERNAL="${REPO}/runs/baselines/future2000_gt_adaptive_2000mesh_external_baselines"
QUALITATIVE="${EXTERNAL}/qualitative"
REPORT="${EXTERNAL}/report"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"

cd "${REPO}"
PYTHONPATH=src "${PYTHON}" scripts/render_future2000_unified_qualitative.py \
    --manifest "${MANIFEST}" \
    --learned-analysis "${LEARNED}" \
    --external-output "${EXTERNAL}" \
    --output-dir "${QUALITATIVE}" \
    --count 4 \
    --image-size 320

PYTHONPATH=src "${PYTHON}" scripts/generate_future2000_unified_report.py \
    --laplacian-run "${LAP}" \
    --displacement-run "${DISP}" \
    --learned-analysis "${LEARNED}" \
    --external-output "${EXTERNAL}" \
    --qualitative-dir "${QUALITATIVE}" \
    --output-dir "${REPORT}" \
    --job-ids 15758 15759 15760 15761 15762 15763 15764 15765 15766 15767 \
        15768 15769 15770 15771 15772 15773 15774 15775 15776 15778
