#!/bin/bash

#SBATCH --job-name=s50_fixed_lambda_eval
#SBATCH --partition=xlong
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
RUNS="${REPO}/runs/learned_laplacian"
LAMBDA_SELECTION="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/lambda_validation/lambda_selection.json"
BETA_SELECTION="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/beta_validation/beta_selection.json"
OUTPUT="${REPO}/runs/diagnostics/sofa50_recovery_aware_fixed_lambda_extended/final"
ARM_INDEX="$((SLURM_ARRAY_TASK_ID + 3))"

export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=8
mkdir -p "${OUTPUT}/shards" "${REPO}/slurm_logs"
cd "${REPO}"
"${PYTHON}" scripts/evaluate_sofa50_recovery_aware_ablation.py \
  --manifest "${MANIFEST}" --runs-root "${RUNS}" \
  --lambda-selection "${LAMBDA_SELECTION}" --beta-selection "${BETA_SELECTION}" \
  --output-dir "${OUTPUT}" --arm-index "${ARM_INDEX}" --device cuda \
  --include-fixed-lambda-sweep
