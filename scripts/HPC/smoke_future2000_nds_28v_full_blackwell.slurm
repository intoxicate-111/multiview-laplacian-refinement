#!/bin/bash
#SBATCH --job-name=f2k_nds28v_smoke
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
SOURCE=/networkhome/WMGDS/zhou_c/external_baselines/neural-deferred-shading
ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/f2k8_nds_blackwell_20260822
PROJECT_PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
CONFIG=${PROJECT}/configs/baselines/future2000_same_initial_full1000_blackwell.json
ROOT=${PROJECT}/runs/benchmarks/future2000_same_initial_full1000_20260822
SAMPLE=016bc42e-c044-3a36-8bbf-5db79d89585b__v00
STAGE=${SLURM_TMPDIR:-/tmp}/f2k-nds28v-smoke-${SLURM_JOB_ID}
ROOT_METHOD=${STAGE}/neural-deferred-shading

mkdir -p "${STAGE}" "${PROJECT}/slurm_logs"
cp -R --no-preserve=ownership,mode,timestamps "${SOURCE}" "${ROOT_METHOD}"
patch --batch --forward -d "${ROOT_METHOD}" -p1 \
    < "${PROJECT}/scripts/HPC/patches/nds_allow_nonmanifold_edges.patch"

export CUDA_HOME=/usr/local/cuda-13.1
export PATH=${ENV}/bin:${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${ENV}/lib:${ENV}/lib64:${LIBRARY_PATH:-}
export CPATH=${ENV}/include:${CPATH:-}
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR=${SLURM_TMPDIR:-/tmp}/torch-extensions-nds28v-smoke-${SLURM_JOB_ID}
export PYTHONPATH=${PROJECT}/src:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONHASHSEED=7
mkdir -p "${TORCH_EXTENSIONS_DIR}"

cd "${PROJECT}"
"${PROJECT_PYTHON}" scripts/evaluate_future2000_external_baseline.py \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --method nds_28v_full \
    --external-root "${ROOT_METHOD}" \
    --external-python "${ENV}/bin/python" \
    --output-dir "${ROOT}/smoke_results" \
    --shard-index 0 \
    --shard-count 1 \
    --sample-id "${SAMPLE}" \
    --expected-test-samples 1000 \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01 \
    --retry-failed \
    --fail-on-sample-error \
    --keep-work
