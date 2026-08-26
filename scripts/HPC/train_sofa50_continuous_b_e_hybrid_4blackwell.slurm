#!/bin/bash

#SBATCH --job-name=s50_ct_be_bw4
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_v2_continuous_b_e_hybrid_4blackwell.json"
PREFLIGHT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/continuous_b_e_hybrid/preflight"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_v2_continuous_pretrained_b_e_hybrid_lambda3e-2_lr1e-4_20k_seed7"
B_CHECKPOINT="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7/checkpoint_best.pt"
E_CHECKPOINT="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_direct_vertex_arm_e_20k_seed7/checkpoint_best.pt"
WORLD_SIZE=4
ACCUMULATION_PER_RANK=2
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

for path in "${PYTHON}" "${MANIFEST}" "${CONFIG}" "${B_CHECKPOINT}" "${E_CHECKPOINT}" "${PREFLIGHT}/step0_validation.json" "${PREFLIGHT}/step0_test.json" "${PREFLIGHT}/nondeterminism_envelope_validation_5runs.json"; do
    test -f "${path}"
done
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=7 OMP_NUM_THREADS=8
export PYTHONNOUSERSITE=1
unset PYTHONHOME

"${PYTHON}" -c 'import json,sys; validation,test,noise=[json.load(open(x)) for x in sys.argv[1:]]; assert validation["contract_audit"] and test["contract_audit"] and noise["contract_audit"]; assert validation["gradient_audit"]["passed"]; assert validation["parameter_count"]==test["parameter_count"]==1652230; assert test["geometry"]["improved_worsened"]==[49,1]; assert len(noise["realizations"])>=5' "${PREFLIGHT}/step0_validation.json" "${PREFLIGHT}/step0_test.json" "${PREFLIGHT}/nondeterminism_envelope_validation_5runs.json"
test "$(sha256sum "${B_CHECKPOINT}" | cut -d' ' -f1)" = "a483e2212f568e771873594cf1e37d13d62cbd2e1e72244baded7dd15573970c"
test "$(sha256sum "${E_CHECKPOINT}" | cut -d' ' -f1)" = "6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1"

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader > "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv"
test "$(wc -l < "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 4
test "$(grep -c 'RTX PRO 6000 Blackwell' "${OUTPUT}/gpu_inventory_${SLURM_JOB_ID}.csv")" -eq 4

{
    echo "initialization=exact_complete_pretrained_Arm_B_plus_complete_pretrained_Arm_E"
    echo "shared_backbone=false parameter_count=1652230"
    echo "optimizer=fresh_Adam continuation_lr=1e-4"
    echo "world_size=4 accumulation_per_rank=2 effective_global_batch=8"
    echo "loss=mean_i_squared_l2_final_hybrid_vertices_minus_clean only"
    echo "operator=uniform_random_walk lambda=3e-2 pcg=float64,tol1e-8,max2048"
    echo "checkpoint_selector=validation_final_hybrid_unified_v2_chamfer_only"
    sha256sum "${CONFIG}" "${B_CHECKPOINT}" "${E_CHECKPOINT}" \
        "${PREFLIGHT}/step0_validation.json" "${PREFLIGHT}/step0_test.json" \
        "${PREFLIGHT}/nondeterminism_envelope_validation_5runs.json"
} > "${OUTPUT}/launch_contract_${SLURM_JOB_ID}.txt"

"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc-per-node="${WORLD_SIZE}" \
    --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py --manifest "${MANIFEST}" --config "${CONFIG}" \
    --output-dir "${OUTPUT}" --device cuda --num-workers 0 --no-persistent-workers \
    --gradient-accumulation-meshes "${ACCUMULATION_PER_RANK}"
