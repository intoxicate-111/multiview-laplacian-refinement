#!/bin/bash

#SBATCH --job-name=s50_old1920_ctval
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
RUN="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_continuous_b_e_lr1e-4_20k_seed7_v1"
PREFLIGHT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/continuous/preflight/step0_validation.json"
OUTPUT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/continuous/validation/selected_checkpoint.json"
for path in "${MANIFEST}" "${RUN}/run_config.json" "${RUN}/checkpoint_best.pt" "${RUN}/metrics.json" "${PREFLIGHT}"; do test -f "${path}"; done
"${PYTHON}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["contract_audit"] is True' "${PREFLIGHT}"
mkdir -p "$(dirname "${OUTPUT}")" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" PYTHONHASHSEED=7 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO}"
"${PYTHON}" scripts/evaluate_sofa50_continuous_checkpoint_validation.py \
  --manifest "${MANIFEST}" --run "${RUN}" --checkpoint "${RUN}/checkpoint_best.pt" \
  --label old_domain_continuous_B_E --split validation --output "${OUTPUT}" --device cuda
