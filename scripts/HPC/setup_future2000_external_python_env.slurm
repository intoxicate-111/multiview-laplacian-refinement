#!/bin/bash
#SBATCH --job-name=f2k_ext_setup
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

METHOD="${1:?usage: sbatch setup_future2000_external_python_env.slurm nds|nerf2mesh|exmesh|da3}"
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
EXTERNAL="/networkhome/WMGDS/zhou_c/external_baselines"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
ENV_NAME="future_${METHOD}"
ENV_DIR="/networkhome/WMGDS/zhou_c/miniconda3/envs/${ENV_NAME}"
CONFIG="${REPO}/configs/baselines/future2000_external_baselines.json"

case "${METHOD}" in
    nds|nerf2mesh|exmesh|da3) ;;
    *) echo "Unsupported method: ${METHOD}" >&2; exit 2 ;;
esac

mkdir -p "${REPO}/slurm_logs"
cd "${REPO}"
echo "method=${METHOD} env=${ENV_NAME} node=$(hostname)"
nvidia-smi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    "${CONDA}" create -y -n "${ENV_NAME}" --clone test
fi
PYTHON="${ENV_DIR}/bin/python"
PIP=("${PYTHON}" -m pip install --no-cache-dir)

# Preserve the cluster-tested torch 2.5.1+cu124 runtime.  Newer CUDA wheels
# requested by some upstream requirement files are incompatible with driver 550.
"${PYTHON}" - <<'PY'
import torch
assert torch.__version__.startswith("2.5.1"), torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

if [[ "${METHOD}" == "nds" ]]; then
    "${CONDA}" install -y -n "${ENV_NAME}" -c nvidia cuda-nvcc=12.4
    export CUDA_HOME="${ENV_DIR}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    "${PIP[@]}" \
        imageio==2.37.3 imageio-ffmpeg==0.6.0 gpytoolbox==0.3.7 \
        opencv-python==4.11.0.86 'numpy<2' matplotlib pillow scikit-image \
        tqdm trimesh
    "${PIP[@]}" "git+https://github.com/mworchel/meshzoo.git@c2767359737f4a76a4f65da4bd4a694d61303f7b"
    "${PIP[@]}" --no-build-isolation \
        "git+https://github.com/NVlabs/nvdiffrast.git@20c4135fda78ae29d7658bea28b6c1d5b5e103e5"
    ROOT="${EXTERNAL}/neural-deferred-shading"
    "${PYTHON}" -c 'import gpytoolbox, nvdiffrast.torch, torch; print(torch.cuda.is_available())'
elif [[ "${METHOD}" == "nerf2mesh" ]]; then
    "${CONDA}" install -y -n "${ENV_NAME}" -c nvidia cuda-nvcc=12.4
    export CUDA_HOME="${ENV_DIR}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    "${PIP[@]}" \
        rich tqdm ninja scipy lpips pandas trimesh PyMCubes torch-ema \
        packaging matplotlib tensorboardX opencv-python imageio \
        imageio-ffmpeg pymeshlab xatlas scikit-learn torchmetrics dearpygui
    "${PIP[@]}" --no-build-isolation \
        "git+https://github.com/NVlabs/tiny-cuda-nn.git@749dd70c5afc5a9dadb85e5652ed65d55e0ba187#subdirectory=bindings/torch"
    "${PIP[@]}" --no-build-isolation \
        "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"
    "${PIP[@]}" --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@3143b3baf8ef8b1023ed76f225af59e2e8a71e06"
    "${PIP[@]}" --no-build-isolation torch-scatter
    ROOT="${EXTERNAL}/nerf2mesh"
    (cd "${ROOT}" && "${PYTHON}" -c 'import tinycudann, nvdiffrast.torch, pytorch3d, raymarching; print("imports_ok")')
elif [[ "${METHOD}" == "exmesh" ]]; then
    "${CONDA}" install -y -n "${ENV_NAME}" -c nvidia cuda-nvcc=12.4
    export CUDA_HOME="${ENV_DIR}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    "${PIP[@]}" open3d plyfile opencv-python lpips trimesh xatlas \
        imageio scikit-image tensorboard
    "${PIP[@]}" --no-build-isolation \
        "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"
    ROOT="${EXTERNAL}/ExMesh"
    (cd "${ROOT}" && "${PYTHON}" -c 'import nvdiffrast.torch, open3d, lpips; from scene import Scene; print("imports_ok")')
else
    "${PIP[@]}" 'numpy<2' einops huggingface_hub imageio opencv-python \
        xformers==0.0.29.post1 open3d fastapi uvicorn requests typer pillow \
        omegaconf evo e3nn moviepy==1.0.3 plyfile pillow_heif safetensors pycolmap \
        hatchling hatch-vcs
    ROOT="${EXTERNAL}/depth-anything-3"
    "${PIP[@]}" --no-deps -e "${ROOT}"
    "${PYTHON}" -c 'from depth_anything_3.api import DepthAnything3; print("imports_ok")'
fi

"${PYTHON}" - <<PY
import json, pathlib, subprocess, torch
root = pathlib.Path("${ROOT}")
payload = {
    "status": "completed",
    "method": "${METHOD}",
    "environment": "${ENV_DIR}",
    "python": subprocess.check_output(["${PYTHON}", "--version"], text=True, stderr=subprocess.STDOUT).strip(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "external_root": str(root),
    "external_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "environment_config": json.loads(pathlib.Path("${CONFIG}").read_text())["environment_pins"],
}
path = pathlib.Path("${REPO}/runs/baselines/future2000_gt_adaptive_2000mesh_external_baselines/setup")
path.mkdir(parents=True, exist_ok=True)
(path / "${METHOD}.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
