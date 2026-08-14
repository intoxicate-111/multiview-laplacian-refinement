#!/bin/bash
#SBATCH --job-name=f2k_openmvs_build
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
SOURCE="/networkhome/WMGDS/zhou_c/external_baselines/openMVS"
VCPKG="/networkhome/WMGDS/zhou_c/external_baselines/vcpkg-openmvs"
TOOLS="/networkhome/WMGDS/zhou_c/miniconda3/envs/future_openmvs_tools"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
BUILD="${SOURCE}/make-future2000-v2"
VCPKG_COMMIT="56bb2411609227288b70117ead2c47585ba07713"

mkdir -p "${REPO}/slurm_logs"
if [[ ! -x "${TOOLS}/bin/cmake" ]]; then
    "${CONDA}" create -y -n future_openmvs_tools -c conda-forge \
        cmake ninja pkg-config nasm bison libtool autoconf automake
fi
export PATH="${TOOLS}/bin:${PATH}"
export VCPKG_ROOT="${VCPKG}"
if [[ ! -d "${VCPKG}/.git" ]]; then
    git clone https://github.com/microsoft/vcpkg.git "${VCPKG}"
fi
git -C "${VCPKG}" fetch origin "${VCPKG_COMMIT}"
git -C "${VCPKG}" checkout --detach "${VCPKG_COMMIT}"
"${VCPKG}/bootstrap-vcpkg.sh" -disableMetrics

cmake -S "${SOURCE}" -B "${BUILD}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DVCPKG_ROOT="${VCPKG}" \
    -DCMAKE_TOOLCHAIN_FILE="${VCPKG}/scripts/buildsystems/vcpkg.cmake" \
    -DVCPKG_TARGET_TRIPLET=x64-linux-release \
    -DOpenMVS_USE_CUDA=OFF \
    -DOpenMVS_HEADLESS_DEBUG=ON \
    -DOpenMVS_USE_VIEWER=OFF
cmake --build "${BUILD}" --parallel "${SLURM_CPUS_PER_TASK}"

INTERFACE="$(find "${BUILD}/bin" -type f -name InterfaceCOLMAP -perm -u+x | head -1)"
REFINE="$(find "${BUILD}/bin" -type f -name RefineMesh -perm -u+x | head -1)"
test -n "${INTERFACE}"
test -n "${REFINE}"
"${INTERFACE}" --help >/dev/null 2>&1 || test "$?" -le 1
"${REFINE}" --help >/dev/null 2>&1 || test "$?" -le 1
mkdir -p "${REPO}/runs/baselines/future2000_gt_adaptive_2000mesh_external_baselines/setup"
printf '{"status":"completed","method":"openmvs_refinemesh","commit":"%s","interface_colmap":"%s","refine_mesh":"%s","cuda":false}\n' \
    "$(git -C "${SOURCE}" rev-parse HEAD)" "${INTERFACE}" "${REFINE}" \
    > "${REPO}/runs/baselines/future2000_gt_adaptive_2000mesh_external_baselines/setup/openmvs_refinemesh.json"
