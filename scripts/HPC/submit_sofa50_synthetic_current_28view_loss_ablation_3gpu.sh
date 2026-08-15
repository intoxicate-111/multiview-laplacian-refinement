#!/usr/bin/env bash

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_synthetic_current_28view_direct_raw_mse_20k_3gpu.json"
MSE_RUN="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_loss_ablation_20k_seed7/MSE_raw_3gpu"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_loss_ablation_20k_seed7/analysis"
TRAIN_SCRIPT="${REPO}/scripts/HPC/train_multi_mesh_ddp.slurm"
EVAL_SCRIPT="${REPO}/scripts/HPC/evaluate_sofa50_synthetic_current_28view_loss_ablation_4gpu.slurm"
MERGE_SCRIPT="${REPO}/scripts/HPC/merge_sofa50_synthetic_current_28view_loss_ablation.slurm"

for path in "${MANIFEST}" "${CONFIG}" "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}" "${MERGE_SCRIPT}"; do
    test -f "${path}"
done
if [[ -e "${MSE_RUN}/checkpoint_latest.pt" || -e "${OUTPUT}/REPORT.md" ]]; then
    echo "Refusing to overwrite an existing completed run." >&2
    exit 2
fi

training_job=$(sbatch --parsable \
    --partition=xlong \
    --nodelist=gpu-03 \
    --gres=gpu:L40:3 \
    --job-name=sofa50_mse_3g \
    --export=ALL,GPUS_PER_NODE=3,NCCL_SOCKET_IFNAME=lo,GLOO_SOCKET_IFNAME=lo \
    "${TRAIN_SCRIPT}" "${MANIFEST}" "${CONFIG}" "${MSE_RUN}")
evaluation_job=$(sbatch --parsable --dependency="afterok:${training_job}" "${EVAL_SCRIPT}")
merge_job=$(sbatch --parsable --dependency="afterok:${evaluation_job}" "${MERGE_SCRIPT}")

echo "training_job=${training_job}"
echo "evaluation_array=${evaluation_job}"
echo "merge_job=${merge_job}"
echo "output=${OUTPUT}"
