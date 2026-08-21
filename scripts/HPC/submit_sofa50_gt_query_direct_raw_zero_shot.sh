#!/usr/bin/env bash

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3/gt_query_views_28_manifest.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_gt_query_28view_direct_raw_hf_20k_2gpu.json"
ROOT="${REPO}/runs/learned_laplacian/sofa50_gt_query_direct_raw_zero_shot_20k_seed7"
RUN="${ROOT}/B_gt_query_direct_raw_hf_2gpu"
AUDIT="${REPO}/scripts/HPC/audit_sofa50_gt_raw_transfer.slurm"
SMOKE="${REPO}/scripts/HPC/smoke_sofa50_gt_query_direct_raw_hf_2gpu.slurm"
TRAIN="${REPO}/scripts/HPC/train_multi_mesh_ddp.slurm"

for path in "${MANIFEST}" "${CONFIG}" "${AUDIT}" "${SMOKE}" "${TRAIN}"; do
    test -f "${path}"
done
if [[ -e "${RUN}/checkpoint_latest.pt" || -e "${RUN}/metrics.json" ]]; then
    echo "Refusing to overwrite existing formal run: ${RUN}" >&2
    exit 2
fi

audit_job=$(sbatch --parsable "${AUDIT}")
smoke_job=$(sbatch --parsable --dependency="afterok:${audit_job}" "${SMOKE}")
training_job=$(sbatch --parsable \
    --dependency="afterok:${smoke_job}" \
    --partition=xlong \
    --nodelist=gpu-03 \
    --gres=gpu:L40:2 \
    --cpus-per-task=12 \
    --mem=64G \
    --time=2-00:00:00 \
    --job-name=s50_gt_raw_20k \
    --export=ALL,GPUS_PER_NODE=2,NCCL_SOCKET_IFNAME=lo,GLOO_SOCKET_IFNAME=lo,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${TRAIN}" "${MANIFEST}" "${CONFIG}" "${RUN}")

echo "contract_audit_job=${audit_job}"
echo "two_gpu_smoke_job=${smoke_job}"
echo "formal_training_job=${training_job}"
echo "formal_training_output=${RUN}"
