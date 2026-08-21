#!/bin/bash
#SBATCH --job-name=exmesh_env
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
EXMESH="/networkhome/WMGDS/zhou_c/external_baselines/ExMesh"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
ENV_DIR="/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official"
OUTPUT="${PROJECT}/runs/exmesh_baselines/exmesh_official"
EXPECTED_COMMIT="09950d283fc5372a09079e30c88d998f1c40b2d0"

mkdir -p "${PROJECT}/slurm_logs" "${OUTPUT}"
test "$(git -C "${EXMESH}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${EXMESH}" status --short --untracked-files=no)"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    "${CONDA}" create -y -p "${ENV_DIR}" python=3.11 pip
fi
PYTHON="${ENV_DIR}/bin/python"
"${CONDA}" install -y -p "${ENV_DIR}" -c nvidia \
    cuda-version=12.1 cuda-nvcc=12.1 cuda-cudart=12.1 \
    cuda-cudart-dev=12.1 cuda-cccl=12.1 \
    libcublas-dev=12.1.3.1 libcusparse-dev=12.1.0.106 \
    libcusolver-dev=11.4.5.107
export CUDA_HOME="${ENV_DIR}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

"${PYTHON}" -m pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
"${PYTHON}" -m pip install --no-cache-dir -r "${EXMESH}/requirements.txt"
"${PYTHON}" -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"
"${PYTHON}" -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@3143b3baf8ef8b1023ed76f225af59e2e8a71e06"
"${PYTHON}" -m pip install --no-cache-dir --no-build-isolation \
    "${EXMESH}/PGSR/submodules/diff-plane-rasterization"
"${PYTHON}" -m pip install --no-cache-dir --no-build-isolation \
    "${EXMESH}/PGSR/submodules/simple-knn"

"${PYTHON}" -m pip freeze > "${OUTPUT}/pip_freeze.txt"
"${CONDA}" list -p "${ENV_DIR}" --explicit > "${OUTPUT}/conda_explicit.txt"
cd "${EXMESH}"
"${PYTHON}" - <<'PY' > "${OUTPUT}/environment.json"
import json
import platform
import subprocess

import cv2
import numpy
import open3d
import torch

payload = {
    "status": "ready",
    "repository": "https://github.com/Fan-Treasure/ExMesh.git",
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "opencv": cv2.__version__,
    "open3d": open3d.__version__,
    "numpy": numpy.__version__,
    "execution_adaptation": "PyTorch is pinned to 2.5.1+cu121 because the official README does not pin a version.",
    "scientific_algorithm_change": False,
}
print(json.dumps(payload, indent=2))
PY

"${PYTHON}" -c 'import nvdiffrast.torch, pytorch3d, diff_plane_rasterization, simple_knn; print("official ExMesh imports: ok")'
