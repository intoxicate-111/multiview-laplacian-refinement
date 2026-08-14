#!/bin/bash
#SBATCH --job-name=f2k_adapt_c2f2
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
CONDA=/networkhome/WMGDS/zhou_c/miniconda3/bin/conda
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
CONFIG="${REPO}/configs/learned_laplacian/train_future2000x5_current_28view_c2f2_direct_raw_20k.json"
OUTPUT="${REPO}/runs/learned_laplacian/future2000x5_gt_adaptive_current_28view_c2f2_direct_raw_20k_seed7"
GPUS_PER_NODE="${GPUS_PER_NODE:-3}"

for path in "${MANIFEST}" "${CONFIG}"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}"; exit 2; }
done
mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
export OMP_NUM_THREADS=8 NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cd "${REPO}"
echo "world_size=${GPUS_PER_NODE} manifest=${MANIFEST} output=${OUTPUT}"
nvidia-smi
srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  "${CONDA}" run --no-capture-output -n test torchrun \
    --nnodes=1 --nproc-per-node="${GPUS_PER_NODE}" --node-rank=0 \
    --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
    scripts/train_multi_mesh_laplacian.py \
      --manifest "${MANIFEST}" --config "${CONFIG}" --output-dir "${OUTPUT}" --device cuda
