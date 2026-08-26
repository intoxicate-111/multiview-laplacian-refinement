#!/bin/bash

#SBATCH --job-name=s50_old1920_preflight
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json"
CONFIG_B="${REPO}/configs/learned_laplacian/train_sofa50_old_domain_native1920_arm_b_20k_4blackwell.json"
CONFIG_E="${REPO}/configs/learned_laplacian/train_sofa50_old_domain_native1920_arm_e_20k_4blackwell.json"
DATA_AUDIT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/preflight/contract_audit.json"
OUTPUT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/preflight/runtime_preflight.json"

for path in "${PYTHON}" "${MANIFEST}" "${CONFIG_B}" "${CONFIG_E}" "${DATA_AUDIT}"; do
  test -f "${path}"
done
mkdir -p "${REPO}/slurm_logs" "$(dirname "${OUTPUT}")"
export PYTHONPATH="${REPO}/src"
export PYTHONHASHSEED=7 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO}"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
"${PYTHON}" scripts/preflight_sofa50_old_domain_native1920_arms.py \
  --manifest "${MANIFEST}" \
  --arm-b-config "${CONFIG_B}" \
  --arm-e-config "${CONFIG_E}" \
  --data-audit "${DATA_AUDIT}" \
  --output "${OUTPUT}"
