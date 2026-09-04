#!/bin/bash

#SBATCH --job-name=s50_old1920_valspec_l40
#SBATCH --partition=long
#SBATCH --nodelist=gpu-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json"
RUN_B="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_arm_b_20k_seed7_v1"
RUN_E="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_arm_e_20k_seed7_v1"
OUTPUT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/validation_selection/specialists"

for path in "${PYTHON}" "${MANIFEST}" "${RUN_B}/checkpoint_best.pt" "${RUN_E}/checkpoint_best.pt"; do
  test -f "${path}"
done
test ! -e "${OUTPUT}/validation_specialist_summary.json"
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" PYTHONHASHSEED=7 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO}"

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; assert "L40" in torch.cuda.get_device_name(0); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)'
sha256sum "${RUN_B}/checkpoint_best.pt" "${RUN_E}/checkpoint_best.pt"

"${PYTHON}" scripts/evaluate_sofa50_old_domain_specialists.py \
  --manifest "${MANIFEST}" --arm-b-run "${RUN_B}" --arm-e-run "${RUN_E}" \
  --split validation --output-dir "${OUTPUT}" --device cuda
