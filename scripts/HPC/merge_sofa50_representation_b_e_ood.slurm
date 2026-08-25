#!/bin/bash
#SBATCH --job-name=s50_be_ood_merge
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
OUTPUT="${REPO}/runs/diagnostics/sofa50_representation_b_vs_e_frozen_v1/ood"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
cd "${REPO}"
"${PYTHON}" scripts/evaluate_frozen_sofa50_b_e_ood.py --output-dir "${OUTPUT}" --merge-only \
  --domain "legacy_v1|/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1/manifest.json|1" \
  --domain "unseen_recipes_v1|/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLapUnseen25_v1/manifest.json|1" \
  --domain "future2000_x5|/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json|8"
