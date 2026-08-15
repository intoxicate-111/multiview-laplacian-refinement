#!/usr/bin/env bash

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_synthetic_current_28view_direct_raw_original_plus_high_frequency_1920_20k_4gpu.json"
ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_hf_resolution_1920_20k_seed7"
RUN="${ROOT}/HF_native1920_4gpu"
PREP="${REPO}/scripts/HPC/prepare_sofa50_synthetic_current_28view_native1920_array.slurm"
MERGE="${REPO}/scripts/HPC/merge_sofa50_synthetic_current_28view_native1920.slurm"
SMOKE="${REPO}/scripts/HPC/smoke_sofa50_hf1920_4gpu.slurm"
TRAIN="${REPO}/scripts/HPC/train_multi_mesh_ddp.slurm"

for path in "${CONFIG}" "${PREP}" "${MERGE}" "${SMOKE}" "${TRAIN}"; do
    test -f "${path}"
done
if [[ -e "${RUN}/checkpoint_latest.pt" || -e "${RUN}/metrics.json" ]]; then
    echo "Refusing to overwrite existing training output: ${RUN}" >&2
    exit 2
fi

if [[ -f "${MANIFEST}" ]]; then
    echo "Reusing completed audited 1920 dataset: ${MANIFEST}"
    prep_job=""
    merge_job=""
    smoke_dependency=""
else
    prep_job=$(sbatch --parsable "${PREP}")
    merge_job=$(sbatch --parsable --dependency="afterok:${prep_job}" "${MERGE}")
    smoke_dependency="--dependency=afterok:${merge_job}"
fi
smoke_job=$(sbatch --parsable ${smoke_dependency:-} "${SMOKE}")
training_job=$(sbatch --parsable \
    --dependency="afterok:${smoke_job}" \
    --partition=xlong \
    --nodelist=gpu-03 \
    --gres=gpu:L40:4 \
    --cpus-per-task=32 \
    --mem=128G \
    --job-name=s50_hf1920_4g \
    --export=ALL,GPUS_PER_NODE=4,NCCL_SOCKET_IFNAME=lo,GLOO_SOCKET_IFNAME=lo,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${TRAIN}" "${MANIFEST}" "${CONFIG}" "${RUN}")

echo "preparation_array_job=${prep_job:-reused}"
echo "dataset_merge_job=${merge_job:-reused}"
echo "four_gpu_smoke_job=${smoke_job}"
echo "training_job=${training_job}"
echo "training_output=${RUN}"
