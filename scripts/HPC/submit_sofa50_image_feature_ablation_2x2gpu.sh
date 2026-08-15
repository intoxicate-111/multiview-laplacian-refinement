#!/usr/bin/env bash

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
TRAIN_SCRIPT="${REPO}/scripts/HPC/train_multi_mesh_ddp.slurm"
ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_image_feature_ablation_20k_seed7"
GAUSSIAN_CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_synthetic_current_28view_direct_raw_gaussian_feature_20k_2gpu.json"
HIGH_CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_synthetic_current_28view_direct_raw_original_plus_high_frequency_20k_2gpu.json"
GAUSSIAN_RUN="${ROOT}/B_gaussian_feature_2gpu"
HIGH_RUN="${ROOT}/C_original_plus_high_frequency_2gpu"

for path in \
    "${MANIFEST}" \
    "${TRAIN_SCRIPT}" \
    "${GAUSSIAN_CONFIG}" \
    "${HIGH_CONFIG}"
do
    test -f "${path}"
done
for run_dir in "${GAUSSIAN_RUN}" "${HIGH_RUN}"; do
    if [[ -e "${run_dir}/checkpoint_latest.pt" || -e "${run_dir}/metrics.json" ]]; then
        echo "Refusing to overwrite an existing run: ${run_dir}" >&2
        exit 2
    fi
done

common=(
    --parsable
    --partition=xlong
    --nodelist=gpu-03
    --gres=gpu:L40:2
    --cpus-per-task=12
    --mem=64G
    --export=ALL,GPUS_PER_NODE=2,NCCL_SOCKET_IFNAME=lo,GLOO_SOCKET_IFNAME=lo
)
gaussian_job=$(sbatch "${common[@]}" \
    --job-name=s50_gauss2g \
    "${TRAIN_SCRIPT}" "${MANIFEST}" "${GAUSSIAN_CONFIG}" "${GAUSSIAN_RUN}")
high_job=$(sbatch "${common[@]}" \
    --job-name=s50_high2g \
    "${TRAIN_SCRIPT}" "${MANIFEST}" "${HIGH_CONFIG}" "${HIGH_RUN}")

echo "gaussian_training_job=${gaussian_job}"
echo "high_frequency_training_job=${high_job}"
echo "output_root=${ROOT}"
