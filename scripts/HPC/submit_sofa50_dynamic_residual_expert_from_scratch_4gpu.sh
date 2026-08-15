#!/usr/bin/env bash

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_synthetic_current_28view_dynamic_residual_expert_from_scratch_20k_4gpu.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_dynamic_residual_expert_20k_seed7/joint_from_scratch_4gpu"
TRAIN_SCRIPT="${REPO}/scripts/HPC/train_multi_mesh_ddp.slurm"

for path in "${MANIFEST}" "${CONFIG}" "${TRAIN_SCRIPT}"; do
    test -f "${path}"
done
if [[ -e "${OUTPUT}/checkpoint_latest.pt" || -e "${OUTPUT}/metrics.json" ]]; then
    echo "Refusing to overwrite an existing run: ${OUTPUT}" >&2
    exit 2
fi

training_job=$(sbatch --parsable \
    --partition=xlong \
    --gres=gpu:L40:4 \
    --job-name=sofa50_dynzero_4g \
    --export="ALL,GPUS_PER_NODE=4,NCCL_SOCKET_IFNAME=lo,GLOO_SOCKET_IFNAME=lo" \
    "${TRAIN_SCRIPT}" "${MANIFEST}" "${CONFIG}" "${OUTPUT}")

echo "training_job=${training_job}"
echo "initialization=random_seed_7_no_checkpoint"
echo "output=${OUTPUT}"
