#!/bin/bash
#SBATCH --job-name=s50_be_spectral_vis
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
ROOT="${REPO}/runs/diagnostics/sofa50_representation_b_vs_e_frozen_v1"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
AD="${REPO}/runs/diagnostics/sofa50_recovery_aware_lambda_extension/final"
AE="${REPO}/runs/diagnostics/sofa50_direct_vertex_arm_e_extension/final"
test ! -e "${ROOT}/visuals"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${REPO}"
"${PYTHON}" scripts/render_sofa50_representation_b_e_spectral.py \
  --manifest "${MANIFEST}" --ad-report-dir "${AD}" --ae-report-dir "${AE}" \
  --matched-dir "${ROOT}/matched" --output-dir "${ROOT}/visuals" \
  --chebyshev-order 128 --image-size 384 --views 0 7
