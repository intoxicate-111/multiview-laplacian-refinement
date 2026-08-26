#!/bin/bash

#SBATCH --job-name=s50_old1920_lamsweep
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json"
SPECIALISTS="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/validation_selection/specialists"
OUTPUT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/validation_selection/frozen_lambda"
PREDICTIONS="${SPECIALISTS}/validation_specialist_predictions.npz"
SUMMARY="${SPECIALISTS}/validation_specialist_summary.json"
for path in "${MANIFEST}" "${PREDICTIONS}" "${SUMMARY}"; do test -f "${path}"; done
mkdir -p "${OUTPUT}/shards" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" PYTHONHASHSEED=7 OMP_NUM_THREADS=8
cd "${REPO}"
"${PYTHON}" scripts/sweep_sofa50_old_domain_frozen_lambda.py \
  --mode shard --manifest "${MANIFEST}" --predictions "${PREDICTIONS}" \
  --specialist-summary "${SUMMARY}" --output-dir "${OUTPUT}" \
  --shard-count 5 --shard-index "${SLURM_ARRAY_TASK_ID}" --device cuda
