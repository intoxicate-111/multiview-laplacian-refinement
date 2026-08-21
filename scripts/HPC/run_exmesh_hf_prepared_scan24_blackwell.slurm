#!/bin/bash
#SBATCH --job-name=exmesh_hf_prep_s24
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=0-08:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python
EXMESH=/networkhome/WMGDS/zhou_c/external_baselines/ExMesh
DTU=${EXMESH}/workdir/DTU
EXMESH_PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_blackwell/bin/python
RUN=${PROJECT}/runs/learned_laplacian/sofa50_synthetic_current_28view_hf_resolution_1920_20k_seed7/HF_native1920_4gpu
CHECKPOINT=${RUN}/checkpoint_latest.pt
CONFIG=${RUN}/config.json
OUTPUT=${PROJECT}/runs/exmesh_prepared_protocol_20260820/ours/scan24

# Submit with:
# sbatch --export=ALL,PREPARED_MESH=/absolute/path/to/the/audited/scan24/prepared_mesh.obj \
#   scripts/HPC/run_exmesh_hf_prepared_scan24_blackwell.slurm
: "${PREPARED_MESH:?Set PREPARED_MESH to the exact pipeline-prepared DTU scan24 mesh}"

for path in \
    "${PYTHON}" \
    "${PROJECT}/runs/exmesh_baselines/common_contract.json" \
    "${CHECKPOINT}" \
    "${CONFIG}" \
    "${PREPARED_MESH}" \
    "${PROJECT}/scripts/run_exmesh_hf_zero_shot.py"
do
    test -e "${path}"
done
if [[ -e "${OUTPUT}" ]]; then
    echo "Refusing to overwrite versioned prepared-mesh result: ${OUTPUT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT}/logs" "${PROJECT}/slurm_logs"
cd "${PROJECT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=7
export PYTHONPATH="${PROJECT}/src"

nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader,nounits > "${OUTPUT}/logs/gpu_inventory_${SLURM_JOB_ID}.csv"
/usr/bin/time -v "${PYTHON}" scripts/run_exmesh_hf_zero_shot.py \
    --contract runs/exmesh_baselines/common_contract.json \
    --scene-id 24 \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --input-mesh "${PREPARED_MESH}" \
    --input-role prepared_current \
    --output-dir "${OUTPUT}" \
    --exmesh-root "${EXMESH}" \
    --dtu-root "${DTU}" \
    --exmesh-python "${EXMESH_PYTHON}" \
    --device cuda \
    > "${OUTPUT}/logs/pipeline_${SLURM_JOB_ID}.out" \
    2> "${OUTPUT}/logs/pipeline_${SLURM_JOB_ID}.err"
