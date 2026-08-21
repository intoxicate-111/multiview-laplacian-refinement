#!/bin/bash
#SBATCH --job-name=exmesh_hf_s24
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=0-08:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
CONDA=/networkhome/WMGDS/zhou_c/miniconda3/bin/conda
EXMESH=/networkhome/WMGDS/zhou_c/external_baselines/ExMesh
DTU=${EXMESH}/workdir/DTU
EXMESH_PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official/bin/python
RUN=${PROJECT}/runs/learned_laplacian/sofa50_synthetic_current_28view_hf_resolution_1920_20k_seed7/HF_native1920_4gpu
CHECKPOINT=${RUN}/checkpoint_latest.pt
CONFIG=${RUN}/config.json
OUTPUT=${PROJECT}/runs/exmesh_baselines/ours/scan24

for path in \
    "${PROJECT}/runs/exmesh_baselines/common_contract.json" \
    "${CHECKPOINT}" \
    "${CONFIG}" \
    "${DTU}/scan24/mesh.ply" \
    "${PROJECT}/scripts/run_exmesh_hf_zero_shot.py"
do
    test -f "${path}"
done
if [[ -f "${OUTPUT}/status.json" ]]; then
    echo "Refusing to overwrite completed HF zero-shot result: ${OUTPUT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT}/logs" "${PROJECT}/slurm_logs"
cd "${PROJECT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=7

/usr/bin/time -v "${CONDA}" run --no-capture-output -n test \
    python scripts/run_exmesh_hf_zero_shot.py \
        --contract runs/exmesh_baselines/common_contract.json \
        --scene-id 24 \
        --checkpoint "${CHECKPOINT}" \
        --config "${CONFIG}" \
        --output-dir "${OUTPUT}" \
        --exmesh-root "${EXMESH}" \
        --dtu-root "${DTU}" \
        --exmesh-python "${EXMESH_PYTHON}" \
        --device cuda \
    > "${OUTPUT}/logs/pipeline_${SLURM_JOB_ID}.out" \
    2> "${OUTPUT}/logs/pipeline_${SLURM_JOB_ID}.err"

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/aggregate_exmesh_baselines.py \
        --config configs/baselines/exmesh_official_suite.json \
        --output-root runs/exmesh_baselines \
    > "${OUTPUT}/logs/aggregate_${SLURM_JOB_ID}.out" \
    2> "${OUTPUT}/logs/aggregate_${SLURM_JOB_ID}.err"
