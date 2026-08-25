#!/bin/bash

#SBATCH --job-name=s50_hard_anchor_lambda0_eval
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
RUNS="${REPO}/runs/learned_laplacian"
OUTPUT="${REPO}/runs/diagnostics/sofa50_recovery_aware_fixed_lambda_with_hard_anchor/final"
mkdir -p "${OUTPUT}/shards" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=8
cd "${REPO}"
"${PYTHON}" scripts/evaluate_sofa50_hard_anchor_lambda0.py \
  --manifest "${MANIFEST}" --runs-root "${RUNS}" \
  --output-dir "${OUTPUT}" --device cuda
