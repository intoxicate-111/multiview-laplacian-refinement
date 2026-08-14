#!/bin/bash
#SBATCH --job-name=f2k_lap_200k
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-03
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    echo "Usage: sbatch $0 CONFIG OUTPUT_DIR [RESUME_CHECKPOINT]" >&2
    exit 2
fi

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
CONDA=/networkhome/WMGDS/zhou_c/miniconda3/bin/conda
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
RGB_SOURCE=/networkhome/WMGDS/zhou_c/future2000_compact/multiview_960/rendered
CONFIG="$1"
OUTPUT="$2"
RESUME_CHECKPOINT="${3:-}"
GPUS_PER_NODE=4
WORKERS_PER_RANK="${WORKERS_PER_RANK:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
GRADIENT_ACCUMULATION_MESHES="${GRADIENT_ACCUMULATION_MESHES:-1}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-200000}"
REPORT_EVERY_OPTIMIZER_STEPS="${REPORT_EVERY_OPTIMIZER_STEPS:-2000}"

for path in "${MANIFEST}" "${CONFIG}"; do
    [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
[[ -d "${RGB_SOURCE}" ]] || { echo "Missing RGB source: ${RGB_SOURCE}" >&2; exit 2; }
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    [[ -f "${RESUME_CHECKPOINT}" ]] || {
        echo "Missing resume checkpoint: ${RESUME_CHECKPOINT}" >&2
        exit 2
    }
fi
for value in "${WORKERS_PER_RANK}" "${PREFETCH_FACTOR}" "${GRADIENT_ACCUMULATION_MESHES}" "${MAX_OPTIMIZER_STEPS}" "${REPORT_EVERY_OPTIMIZER_STEPS}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "Worker, prefetch and accumulation settings must be positive." >&2
        exit 2
    }
done

NODE_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}/mlr-${SLURM_JOB_ID}}"
RGB_LOCAL="${NODE_TMP_ROOT}/future2000_compact/multiview_960/rendered"
mkdir -p "${RGB_LOCAL}" "${OUTPUT}" "${REPO}/slurm_logs"

echo "Staging compressed RGB files to node-local storage: ${RGB_LOCAL}"
rsync -a --info=stats2 "${RGB_SOURCE}/" "${RGB_LOCAL}/"
SOURCE_COUNT="$(find "${RGB_SOURCE}" -type f | wc -l)"
LOCAL_COUNT="$(find "${RGB_LOCAL}" -type f | wc -l)"
[[ "${SOURCE_COUNT}" == "${LOCAL_COUNT}" ]] || {
    echo "RGB staging count mismatch: source=${SOURCE_COUNT} local=${LOCAL_COUNT}" >&2
    exit 1
}

export MLR_IMAGE_PATH_REMAP_FROM="${RGB_SOURCE}"
export MLR_IMAGE_PATH_REMAP_TO="${RGB_LOCAL}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
ulimit -n "$(ulimit -Hn)"

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
echo "world_size=${GPUS_PER_NODE} accumulation=${GRADIENT_ACCUMULATION_MESHES} global_batch=$((GPUS_PER_NODE * GRADIENT_ACCUMULATION_MESHES))"
echo "workers_per_rank=${WORKERS_PER_RANK} prefetch=${PREFETCH_FACTOR}"
echo "persistent_workers=false multiprocessing_sharing_strategy=file_system"
echo "max_optimizer_steps=${MAX_OPTIMIZER_STEPS} report_every_optimizer_steps=${REPORT_EVERY_OPTIMIZER_STEPS}"
echo "open_file_limit_soft=$(ulimit -Sn) open_file_limit_hard=$(ulimit -Hn)"
echo "manifest=${MANIFEST} config=${CONFIG} output=${OUTPUT} resume=${RESUME_CHECKPOINT:-none}"
echo "rgb_files=${LOCAL_COUNT} omp_threads=${OMP_NUM_THREADS} node_tmp=${NODE_TMP_ROOT}"
nvidia-smi

arguments=(
    --manifest "${MANIFEST}"
    --config "${CONFIG}"
    --output-dir "${OUTPUT}"
    --device cuda
    --num-workers "${WORKERS_PER_RANK}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --no-persistent-workers
    --gradient-accumulation-meshes "${GRADIENT_ACCUMULATION_MESHES}"
    --max-optimizer-steps "${MAX_OPTIMIZER_STEPS}"
    --report-every-optimizer-steps "${REPORT_EVERY_OPTIMIZER_STEPS}"
)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    arguments+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

cd "${REPO}"
srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    "${CONDA}" run --no-capture-output -n test torchrun \
        --nnodes=1 \
        --nproc-per-node="${GPUS_PER_NODE}" \
        --node-rank=0 \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        scripts/train_multi_mesh_laplacian.py \
        "${arguments[@]}"
