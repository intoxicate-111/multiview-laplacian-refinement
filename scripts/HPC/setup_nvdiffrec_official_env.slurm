#!/bin/bash
#SBATCH --job-name=setup_nvdiffrec
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
CONDA=/networkhome/WMGDS/zhou_c/miniconda3/bin/conda
SOURCE_ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds
ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/nvdiffrec_official
NVDIFFREC=/networkhome/WMGDS/zhou_c/external_baselines/nvdiffrec
TCNN=/networkhome/WMGDS/zhou_c/external_baselines/tiny-cuda-nn
OUTPUT=${PROJECT}/runs/exmesh_baselines/nvdiffrec/environment_setup
EXPECTED_NVDIFFREC_COMMIT=abf3a34b1eb6e782abffefc2462c7e9bcd89f9bb
EXPECTED_TCNN_COMMIT=749dd70c5afc5a9dadb85e5652ed65d55e0ba187

mkdir -p "${OUTPUT}"
test "$(git -C "${NVDIFFREC}" rev-parse HEAD)" = "${EXPECTED_NVDIFFREC_COMMIT}"
test "$(git -C "${TCNN}" rev-parse HEAD)" = "${EXPECTED_TCNN_COMMIT}"
test -z "$(git -C "${NVDIFFREC}" status --short --untracked-files=no)"
test -z "$(git -C "${TCNN}" status --short --untracked-files=no)"

if [[ ! -x "${ENV}/bin/python" ]]; then
    "${CONDA}" create -y -p "${ENV}" --clone "${SOURCE_ENV}"
fi

export CUDA_HOME="${ENV}"
export PATH="${ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV}/lib:${ENV}/lib64:${LD_LIBRARY_PATH:-}"
NVRTC_ROOT="${ENV}/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
test -f "${NVRTC_ROOT}/include/nvrtc.h"
test -f "${NVRTC_ROOT}/lib/libnvrtc.so.12"
ln -sfn "${NVRTC_ROOT}/lib/libnvrtc.so.12" "${ENV}/lib/libnvrtc.so"
export CPATH="${NVRTC_ROOT}/include:${CPATH:-}"
export LIBRARY_PATH="${NVRTC_ROOT}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${NVRTC_ROOT}/lib:${LD_LIBRARY_PATH}"
export TCNN_CUDA_ARCHITECTURES=89
export CMAKE_BUILD_PARALLEL_LEVEL=16
export MAX_JOBS=16

"${ENV}/bin/python" -m pip install "setuptools<81" xatlas==0.0.11
# tiny-cuda-nn's setup.py imports pkg_resources while pip's isolated build
# environment currently resolves a newer setuptools release that no longer
# ships it.  Build against the explicitly pinned environment above.
"${ENV}/bin/python" -m pip install --no-build-isolation "${TCNN}/bindings/torch"

cd "${NVDIFFREC}"
PYTHONPATH="${NVDIFFREC}" "${ENV}/bin/python" - <<'PY'
import json
import pathlib
import platform
import subprocess
import torch
import nvdiffrast.torch
import tinycudann
import xatlas

output = pathlib.Path(
    "/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/exmesh_baselines/nvdiffrec/environment_setup"
)
payload = {
    "success": True,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "nvdiffrec_commit": subprocess.check_output(
        ["git", "-C", "/networkhome/WMGDS/zhou_c/external_baselines/nvdiffrec", "rev-parse", "HEAD"], text=True
    ).strip(),
    "tiny_cuda_nn_commit": subprocess.check_output(
        ["git", "-C", "/networkhome/WMGDS/zhou_c/external_baselines/tiny-cuda-nn", "rev-parse", "HEAD"], text=True
    ).strip(),
    "tcnn_cuda_architectures": "89",
}
(output / "status.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
"${ENV}/bin/python" -m pip freeze > "${OUTPUT}/pip_freeze.txt"
