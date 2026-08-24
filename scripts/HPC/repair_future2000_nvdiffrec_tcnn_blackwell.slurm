#!/bin/bash
#SBATCH --job-name=f2k_tcnn_bw_fix
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
SOURCE=/networkhome/WMGDS/zhou_c/external_baselines/tiny-cuda-nn
CUDA_HOME=/usr/local/cuda-13.1

test "$(git -C "${SOURCE}" rev-parse HEAD)" = 749dd70c5afc5a9dadb85e5652ed65d55e0ba187
export CUDA_HOME
export PATH=${ENV}/bin:${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=12.0
export TCNN_CUDA_ARCHITECTURES=120
export CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_PER_TASK}
export MAX_JOBS=${SLURM_CPUS_PER_TASK}

"${ENV}/bin/python" -m pip uninstall -y tinycudann || true
"${ENV}/bin/python" -m pip install --disable-pip-version-check \
    --force-reinstall --no-build-isolation --no-deps \
    "${SOURCE}/bindings/torch"
"${ENV}/bin/python" -c \
    'import tinycudann as tcnn, torch; assert torch.cuda.get_device_capability() == (12, 0); print(torch.__version__, torch.version.cuda, tcnn.__file__)'
