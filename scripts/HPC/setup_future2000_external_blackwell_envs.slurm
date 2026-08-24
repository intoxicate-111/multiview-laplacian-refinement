#!/bin/bash
#SBATCH --job-name=f2k_ext_bw_envs
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

BASE=/networkhome/WMGDS/zhou_c/miniconda3/envs
CONDA=/networkhome/WMGDS/zhou_c/miniconda3/bin/conda
NVD_SOURCE=/networkhome/WMGDS/zhou_c/external_baselines/nvdiffrast-v035
TCNN_SOURCE=/networkhome/WMGDS/zhou_c/external_baselines/tiny-cuda-nn
CUDA_HOME=/usr/local/cuda-13.1
TORCH_INDEX=https://download.pytorch.org/whl/cu130
NDS=${BASE}/f2k8_nds_blackwell_20260822
NVD=${BASE}/f2k8_nvdiffrec_blackwell_20260822
EXMESH=${BASE}/f2k8_exmesh_blackwell_20260822
DA3=${BASE}/f2k8_da3_blackwell_20260822
PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement

test "$(git -C "${NVD_SOURCE}" rev-parse HEAD)" = 20c4135fda78ae29d7658bea28b6c1d5b5e103e5
mkdir -p "${PROJECT}/slurm_logs"

if [[ ! -x "${NDS}/bin/python" ]]; then
    "${CONDA}" create -y -p "${NDS}" --clone "${BASE}/future_nds"
fi
if [[ ! -x "${NVD}/bin/python" ]]; then
    "${CONDA}" create -y -p "${NVD}" --clone "${BASE}/nvdiffrec_official"
fi
if [[ ! -x "${DA3}/bin/python" ]]; then
    "${CONDA}" create -y -p "${DA3}" --clone "${BASE}/future_da3"
fi
if [[ ! -x "${EXMESH}/bin/python" ]]; then
    "${CONDA}" create -y -p "${EXMESH}" --clone "${BASE}/exmesh_official"
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"

for env in "${NDS}" "${NVD}" "${EXMESH}" "${DA3}"; do
    "${env}/bin/python" -m pip install --disable-pip-version-check --upgrade \
        --index-url "${TORCH_INDEX}" 'torch==2.13.0' 'torchvision==0.28.0'
    "${env}/bin/python" -m pip uninstall -y torchaudio || true
done

# NDS and nvdiffrec use nvdiffrast's official RasterizeGLContext.  gpu-04 does
# not expose system EGL development headers, so keep the SDK isolated in the
# corresponding environments.
"${CONDA}" install -y -p "${NDS}" -c conda-forge 'libegl-devel=1.7.0'
"${CONDA}" install -y -p "${NVD}" -c conda-forge 'libegl-devel=1.7.0'

"${DA3}/bin/python" -m pip uninstall -y xformers || true
"${EXMESH}/bin/python" -m pip install --disable-pip-version-check ninja

for env in "${NDS}" "${NVD}" "${EXMESH}"; do
    "${env}/bin/python" -m pip uninstall -y nvdiffrast || true
    "${env}/bin/python" -m pip install --disable-pip-version-check \
        --no-build-isolation --no-deps "${NVD_SOURCE}"
done

test "$(git -C "${TCNN_SOURCE}" rev-parse HEAD)" = 749dd70c5afc5a9dadb85e5652ed65d55e0ba187
export TCNN_CUDA_ARCHITECTURES=120
"${NVD}/bin/python" -m pip uninstall -y tinycudann || true
"${NVD}/bin/python" -m pip install --disable-pip-version-check \
    --force-reinstall --no-build-isolation --no-deps \
    "${TCNN_SOURCE}/bindings/torch"

for env in "${NDS}" "${NVD}" "${EXMESH}" "${DA3}"; do
    "${env}/bin/python" -c 'import torch; assert "sm_120" in torch.cuda.get_arch_list(); x=torch.randn(256,256,device="cuda"); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(),float((x@x).mean()))'
done

"${NDS}/bin/python" -c 'import nvdiffrast.torch as dr, gpytoolbox, meshzoo; print("NDS dependencies: OK")'
"${NVD}/bin/python" -c 'import nvdiffrast.torch as dr, xatlas; print("nvdiffrec dependencies: OK")'
"${NVD}/bin/python" -c 'import tinycudann as tcnn; print("tiny-cuda-nn Blackwell binding: OK")'
"${EXMESH}/bin/python" -c 'import nvdiffrast.torch as dr, lpips, open3d, xatlas; print("ExMesh dependencies: OK")'
PYTHONPATH=/networkhome/WMGDS/zhou_c/external_baselines/depth-anything-3/src \
    "${DA3}/bin/python" -c 'from depth_anything_3.api import DepthAnything3; print("DA3 dependencies: OK")'
