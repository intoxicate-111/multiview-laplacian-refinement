#!/bin/bash

set -euo pipefail

METHOD="${1:-list}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTERNAL_ROOT="${F2K_EXTERNAL_ROOT:-${REPO}/.external}"
CONDA="${F2K_CONDA:-${REPO}/../miniconda3/bin/conda}"
BASE_ENV="${F2K_BASE_ENV:-test}"
CUDA_VERSION="${F2K_CUDA_VERSION:-12.4}"

find_env() {
    local name="$1"
    "${CONDA}" env list | awk -v name="${name}" '$1 == name {print $NF; exit}'
}

clone_env() {
    local name="$1" env_dir
    env_dir="$(find_env "${name}")"
    if [[ -z "${env_dir}" || ! -x "${env_dir}/bin/python" ]]; then
        "${CONDA}" create -y -n "${name}" --clone "${BASE_ENV}"
        env_dir="$(find_env "${name}")"
    fi
    [[ -n "${env_dir}" && -x "${env_dir}/bin/python" ]] || {
        echo "Unable to resolve conda environment: ${name}" >&2
        return 1
    }
    printf '%s\n' "${env_dir}"
}

checkout() {
    local url="$1" commit="$2" path="$3"
    if [[ ! -d "${path}/.git" ]]; then
        git clone "${url}" "${path}"
    fi
    git -C "${path}" fetch origin "${commit}"
    git -C "${path}" checkout --detach "${commit}"
}

setup_nds() {
    local env_dir python
    env_dir="$(clone_env future_nds)"; python="${env_dir}/bin/python"
    "${CONDA}" install -y -n future_nds -c nvidia "cuda-nvcc=${CUDA_VERSION}" "cuda-cudart-dev=${CUDA_VERSION}"
    "${python}" -m pip install --no-cache-dir \
        imageio==2.37.3 imageio-ffmpeg==0.6.0 gpytoolbox==0.3.7 \
        opencv-python==4.11.0.86 'numpy<2' matplotlib pillow scikit-image tqdm trimesh ninja
    "${python}" -m pip install --no-cache-dir \
        'git+https://github.com/mworchel/meshzoo.git@c2767359737f4a76a4f65da4bd4a694d61303f7b'
    CUDA_HOME="${env_dir}" "${python}" -m pip install --no-cache-dir --no-build-isolation \
        'git+https://github.com/NVlabs/nvdiffrast.git@20c4135fda78ae29d7658bea28b6c1d5b5e103e5'
    checkout https://github.com/fraunhoferhhi/neural-deferred-shading.git \
        760e4549f59adaed9adf1bd705599786a00ba6b8 "${EXTERNAL_ROOT}/neural-deferred-shading"
}

setup_nerf2mesh() {
    local env_dir python
    env_dir="$(clone_env future_nerf2mesh)"; python="${env_dir}/bin/python"
    "${CONDA}" install -y -n future_nerf2mesh -c nvidia "cuda-nvcc=${CUDA_VERSION}" "cuda-cudart-dev=${CUDA_VERSION}"
    "${python}" -m pip install --no-cache-dir 'setuptools<81' ninja rich tqdm scipy lpips pandas trimesh \
        PyMCubes torch-ema packaging matplotlib tensorboardX opencv-python imageio imageio-ffmpeg \
        pymeshlab xatlas scikit-learn torchmetrics dearpygui
    CUDA_HOME="${env_dir}" "${python}" -m pip install --no-cache-dir --no-build-isolation \
        'git+https://github.com/NVlabs/tiny-cuda-nn.git@749dd70c5afc5a9dadb85e5652ed65d55e0ba187#subdirectory=bindings/torch'
    CUDA_HOME="${env_dir}" "${python}" -m pip install --no-cache-dir --no-build-isolation \
        'git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae'
    CUDA_HOME="${env_dir}" "${python}" -m pip install --no-cache-dir --no-build-isolation \
        'git+https://github.com/facebookresearch/pytorch3d.git@3143b3baf8ef8b1023ed76f225af59e2e8a71e06' torch-scatter
    checkout https://github.com/ashawkey/nerf2mesh.git \
        ec7f930ccf768ba4d6e602360b4a6ff0300fe9c8 "${EXTERNAL_ROOT}/nerf2mesh"
}

setup_exmesh() {
    local env_dir python
    env_dir="$(clone_env future_exmesh)"; python="${env_dir}/bin/python"
    "${CONDA}" install -y -n future_exmesh -c nvidia "cuda-nvcc=${CUDA_VERSION}" "cuda-cudart-dev=${CUDA_VERSION}"
    "${python}" -m pip install --no-cache-dir ninja open3d plyfile opencv-python lpips trimesh \
        xatlas imageio scikit-image tensorboard
    CUDA_HOME="${env_dir}" "${python}" -m pip install --no-cache-dir --no-build-isolation \
        'git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae'
    checkout https://github.com/Fan-Treasure/ExMesh.git \
        09950d283fc5372a09079e30c88d998f1c40b2d0 "${EXTERNAL_ROOT}/ExMesh"
}

setup_da3() {
    local env_dir python
    env_dir="$(clone_env future_da3)"; python="${env_dir}/bin/python"
    checkout https://github.com/ByteDance-Seed/depth-anything-3.git \
        3d835ec1a5802d64a8b8b15f817a1ab54809bfe4 "${EXTERNAL_ROOT}/depth-anything-3"
    "${python}" -m pip install --no-cache-dir 'numpy<2' einops huggingface_hub imageio \
        opencv-python xformers==0.0.29.post1 open3d fastapi uvicorn requests typer pillow \
        omegaconf evo e3nn moviepy==1.0.3 plyfile pillow_heif safetensors pycolmap hatchling hatch-vcs
    "${python}" -m pip install --no-cache-dir --no-deps -e "${EXTERNAL_ROOT}/depth-anything-3"
}

setup_openmvs() {
    local tools vcpkg source build interface refine
    tools="$(clone_env future_openmvs_tools)"
    "${CONDA}" install -y -n future_openmvs_tools -c conda-forge cmake ninja pkg-config nasm bison libtool autoconf automake
    vcpkg="${EXTERNAL_ROOT}/vcpkg-openmvs"
    source="${EXTERNAL_ROOT}/openMVS"
    build="${source}/make-local"
    checkout https://github.com/cdcseacave/openMVS.git \
        b2f21a032376972dcafc3a402a4618ecd6f35b73 "${source}"
    checkout https://github.com/microsoft/vcpkg.git \
        56bb2411609227288b70117ead2c47585ba07713 "${vcpkg}"
    "${vcpkg}/bootstrap-vcpkg.sh" -disableMetrics
    env -u VCPKG_ROOT PATH="${tools}/bin:${PATH}" cmake -S "${source}" -B "${build}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="${vcpkg}/scripts/buildsystems/vcpkg.cmake" \
        -DVCPKG_TARGET_TRIPLET=x64-linux-release \
        -DOpenMVS_USE_CUDA=OFF \
        -DOpenMVS_HEADLESS_DEBUG=ON \
        -DOpenMVS_USE_VIEWER=OFF
    env -u VCPKG_ROOT PATH="${tools}/bin:${PATH}" cmake --build "${build}" --parallel "${F2K_BUILD_JOBS:-8}"
    interface="$(find "${build}/bin" -type f -name InterfaceCOLMAP -perm -u+x -print -quit)"
    refine="$(find "${build}/bin" -type f -name RefineMesh -perm -u+x -print -quit)"
    test -n "${interface}"
    test -n "${refine}"
    printf 'InterfaceCOLMAP=%s\nRefineMesh=%s\n' "${interface}" "${refine}"
}

case "${METHOD}" in
    list) printf '%s\n' openmvs nds nerf2mesh exmesh da3 all ;;
    openmvs) setup_openmvs ;;
    nds) setup_nds ;;
    nerf2mesh) setup_nerf2mesh ;;
    exmesh) setup_exmesh ;;
    da3) setup_da3 ;;
    all) setup_openmvs; setup_nds; setup_nerf2mesh; setup_exmesh; setup_da3 ;;
    *) echo "usage: $0 openmvs|nds|nerf2mesh|exmesh|da3|all|list" >&2; exit 2 ;;
esac
