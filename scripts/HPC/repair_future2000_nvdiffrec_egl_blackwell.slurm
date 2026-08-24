#!/bin/bash
#SBATCH --job-name=f2k_nvd_egl_fix
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/f2k8_nvdiffrec_blackwell_20260822
CONDA=/networkhome/WMGDS/zhou_c/miniconda3/bin/conda
CUDA_HOME=/usr/local/cuda-13.1

"${CONDA}" install -y -p "${ENV}" -c conda-forge 'libegl-devel=1.7.0'

export CUDA_HOME
export PATH=${ENV}/bin:${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${ENV}/lib:${ENV}/lib64:${LIBRARY_PATH:-}
export CPATH=${ENV}/include:${CPATH:-}
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=${SLURM_CPUS_PER_TASK}
export TORCH_EXTENSIONS_DIR=${SLURM_TMPDIR:-/tmp}/torch-extensions-nvdiffrec-egl-${SLURM_JOB_ID}
mkdir -p "${TORCH_EXTENSIONS_DIR}"

"${ENV}/bin/python" - <<'PY'
import nvdiffrast.torch as dr
import torch

assert torch.cuda.get_device_capability() == (12, 0)
context = dr.RasterizeGLContext()
print("nvdiffrast RasterizeGLContext preflight: OK", context)
PY
