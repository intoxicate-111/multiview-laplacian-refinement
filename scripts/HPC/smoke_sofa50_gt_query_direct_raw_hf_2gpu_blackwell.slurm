#!/bin/bash

#SBATCH --job-name=s50_gt_raw_smoke_bw
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3/gt_query_views_28_manifest.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_gt_query_28view_direct_raw_hf_20k_2gpu.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_gt_query_direct_raw_zero_shot_20k_seed7/smoke_blackwell_2gpu_${SLURM_JOB_ID}"
SMOKE_CONFIG="${OUTPUT}/smoke_config.json"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

for path in "${PYTHON}" "${MANIFEST}" "${CONFIG}"; do
    test -e "${path}"
done
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
"${PYTHON}" -c 'import json,sys; src,dst=sys.argv[1:3]; cfg=json.load(open(src)); multi=cfg["multi_object_training"]; multi["checkpoint_optimizer_steps"]=[]; multi["checkpoint_epochs"]=[]; multi["checkpoint_every_epochs"]=0; json.dump(cfg,open(dst,"w"),indent=2)' "${CONFIG}" "${SMOKE_CONFIG}"
export OMP_NUM_THREADS=8
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO}/src"
"${PYTHON}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node=2 \
    --master-addr=127.0.0.1 \
    --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py \
        --manifest "${MANIFEST}" \
        --config "${SMOKE_CONFIG}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --max-optimizer-steps 1 \
        --report-every-optimizer-steps 1
