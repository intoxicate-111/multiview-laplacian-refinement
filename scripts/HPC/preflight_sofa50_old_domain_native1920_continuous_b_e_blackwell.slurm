#!/bin/bash

#SBATCH --job-name=s50_old1920_ctpre
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json"
BASELINE="${REPO}/configs/learned_laplacian/train_sofa50_old_domain_native1920_arm_b_20k_4blackwell.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_old_domain_native1920_continuous_b_e_4blackwell.json"
RUN_B="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_arm_b_20k_seed7_v1"
RUN_E="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_arm_e_20k_seed7_v1"
SPECIALISTS="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/validation_selection/specialists"
LAMBDA_DIR="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/validation_selection/frozen_lambda"
PREFLIGHT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/continuous/preflight"

for path in \
  "${PYTHON}" "${MANIFEST}" "${BASELINE}" \
  "${RUN_B}/checkpoint_best.pt" "${RUN_E}/checkpoint_best.pt" \
  "${SPECIALISTS}/validation_specialist_summary.json" \
  "${SPECIALISTS}/validation_specialist_predictions.npz" \
  "${LAMBDA_DIR}/lambda_selection.json" \
  "${LAMBDA_DIR}/lambda_sweep_per_sample.csv"; do
  test -f "${path}"
done
mkdir -p "${PREFLIGHT}" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" PYTHONHASHSEED=7 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO}"

"${PYTHON}" scripts/build_sofa50_old_domain_native1920_continuous_config.py \
  --baseline "${BASELINE}" \
  --lambda-selection "${LAMBDA_DIR}/lambda_selection.json" \
  --arm-b-checkpoint "${RUN_B}/checkpoint_best.pt" \
  --arm-e-checkpoint "${RUN_E}/checkpoint_best.pt" \
  --output "${CONFIG}"

"${PYTHON}" scripts/audit_sofa50_old_domain_native1920_continuous_step0.py \
  --manifest "${MANIFEST}" --config "${CONFIG}" \
  --specialist-summary "${SPECIALISTS}/validation_specialist_summary.json" \
  --specialist-predictions "${SPECIALISTS}/validation_specialist_predictions.npz" \
  --lambda-selection "${LAMBDA_DIR}/lambda_selection.json" \
  --lambda-per-sample "${LAMBDA_DIR}/lambda_sweep_per_sample.csv" \
  --output "${PREFLIGHT}/step0_validation.json" \
  --per-sample-output "${PREFLIGHT}/step0_validation_per_sample.csv" \
  --device cuda
