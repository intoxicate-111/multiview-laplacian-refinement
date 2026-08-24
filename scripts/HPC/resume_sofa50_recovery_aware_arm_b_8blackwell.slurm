#!/bin/bash

#SBATCH --job-name=s50_sparse_B_bw8
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
BASELINE="${REPO}/configs/learned_laplacian/train_sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_2gpu_l40.json"
LAMBDA_SELECTION="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/lambda_validation/lambda_selection.json"
BETA_SELECTION="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/beta_validation/beta_selection.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7"
RESUME="${OUTPUT}/checkpoint_migration_bw8.pt"
CONFIG="${OUTPUT}/resume_config_bw8.json"
WORLD_SIZE=8
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

for path in \
    "${PYTHON}" "${MANIFEST}" "${BASELINE}" \
    "${LAMBDA_SELECTION}" "${BETA_SELECTION}" "${RESUME}"; do
    test -f "${path}"
done
read -r LAMBDA BETA < <("${PYTHON}" -c \
    'import json,sys; l=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); assert l["contract_audit"]["passed"] and b["contract_audit"]["passed"]; assert l["selected_lambda"] == b["lambda"]; print(l["selected_lambda"], b["selected_beta"])' \
    "${LAMBDA_SELECTION}" "${BETA_SELECTION}")
RESUME_STEP=$("${PYTHON}" -c \
    'import sys,torch; p=torch.load(sys.argv[1],map_location="cpu",weights_only=False); s=p.get("training_state",{}); assert int(s.get("groups_completed_in_epoch",-1)) == 0; print(int(p["optimizer_steps"]))' \
    "${RESUME}")

export PYTHONPATH="${REPO}/src"
"${PYTHON}" "${REPO}/scripts/build_sofa50_recovery_aware_ablation_config.py" \
    --baseline "${BASELINE}" --output "${CONFIG}" --arm B \
    --lambda-value "${LAMBDA}" --beta "${BETA}" \
    --max-optimizer-steps 20000 --run-kind final \
    --distributed-world-size 8 --effective-global-batch 8 \
    --training-gpu-model "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
    --initialization resume

export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
{
    echo "arm=B initialization=resume checkpoint=${RESUME}"
    echo "resume_optimizer_step=${RESUME_STEP} old_world_size=2 new_world_size=8 epoch_boundary=true"
    echo "world_size=8 accumulation_per_rank=1 effective_global_batch=8"
    echo "execution_difference=DDP_sharding_and_sample_grouping_change_after_step_${RESUME_STEP}"
    echo "lambda=${LAMBDA} beta=${BETA} recovery_loss=enabled"
    sha256sum "${RESUME}"
} > "${OUTPUT}/launch_contract_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node="${WORLD_SIZE}" \
    --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
    --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
    --gradient-accumulation-meshes 1 --resume-checkpoint "${RESUME}"
