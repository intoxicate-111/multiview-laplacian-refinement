#!/bin/bash

#SBATCH --job-name=s50_e2e_hybrid_bw4
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
BASE_CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_v2_end_to_end_hybrid_single_loss_8blackwell.json"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_v2_end_to_end_hybrid_single_loss_lambda3e-2_20k_seed7"
RESUME="${OUTPUT}/checkpoint_pause_bw8_step11150.pt"
CONFIG="${OUTPUT}/resume_config_bw4_step11150.json"
WORLD_SIZE=4
ACCUMULATION_PER_RANK=2
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

for path in "${PYTHON}" "${MANIFEST}" "${BASE_CONFIG}" "${RESUME}"; do
    test -f "${path}"
done
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"

"${PYTHON}" -c 'import json,sys; source,output,resume=sys.argv[1:]; c=json.load(open(source)); m=c["experiment_metadata"]; m.update({"distributed_world_size":4,"effective_global_batch_meshes":8,"gradient_accumulation_meshes_per_rank":2,"initialization":"resume_step11150_after_user_requested_bw8_to_bw4","strict_execution_contract_preserved":False,"execution_contract_change":"world_size_8_to_4_and_accumulation_per_rank_1_to_2_at_epoch_boundary_step11150;effective_global_batch_remains_8","resume_checkpoint":resume}); c["multi_object_training"]["gradient_accumulation_meshes"]=2; json.dump(c,open(output,"w"),indent=2,sort_keys=True)' "${BASE_CONFIG}" "${CONFIG}" "${RESUME}"

RESUME_STEP=$("${PYTHON}" -c 'import sys,torch; p=torch.load(sys.argv[1],map_location="cpu",weights_only=False); s=p.get("training_state",{}); assert int(p["optimizer_steps"])==11150; assert int(s.get("groups_completed_in_epoch",-1))==0; assert int(s.get("distributed_world_size",-1))==8; print(int(p["optimizer_steps"]))' "${RESUME}")
"${PYTHON}" -c 'import json,sys; c=json.load(open(sys.argv[1])); m=c["experiment_metadata"]; assert m["distributed_world_size"]==4; assert m["effective_global_batch_meshes"]==8; assert m["gradient_accumulation_meshes_per_rank"]==2; assert c["multi_object_training"]["gradient_accumulation_meshes"]==2; assert c["multi_object_training"]["max_optimizer_steps"]==20000; assert c["training"]["hybrid_single_geometry_loss"]["enabled"]; assert c["recovery"]["lambda"]==0.03' "${CONFIG}"

export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
test "$(wc -l < "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 4
test "$(grep -c 'RTX PRO 6000 Blackwell' "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 4

{
    echo "initialization=resume checkpoint=${RESUME}"
    echo "resume_optimizer_step=${RESUME_STEP} old_world_size=8 new_world_size=4 epoch_boundary=true"
    echo "world_size=4 accumulation_per_rank=2 effective_global_batch=8"
    echo "user_requested_gpu_release=true strict_execution_contract_preserved=false"
    echo "unchanged=model_dataset_loss_lambda_optimizer_lr_scheduler_seed_max_optimizer_steps_global_batch_validation_cadence"
    sha256sum "${BASE_CONFIG}" "${CONFIG}" "${RESUME}"
} > "${OUTPUT}/launch_contract_${SLURM_JOB_ID}.txt"

"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node="${WORLD_SIZE}" \
    --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
    --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
    --gradient-accumulation-meshes "${ACCUMULATION_PER_RANK}" --resume-checkpoint "${RESUME}"
