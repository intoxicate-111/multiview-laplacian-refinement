#!/bin/bash
#SBATCH --job-name=f2k1000_ext_bw
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

METHOD=${1:?usage: sbatch run_future2000_same_initial_full1000_external_blackwell.slurm nds|nds_28v_full|nvdiffrec|exmesh}
PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
EXTERNAL=/networkhome/WMGDS/zhou_c/external_baselines
PROJECT_PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
CONFIG=${PROJECT}/configs/baselines/future2000_same_initial_full1000_blackwell.json
ROOT=${PROJECT}/runs/benchmarks/future2000_same_initial_full1000_20260822
INDEX=${SLURM_ARRAY_TASK_ID}
ARGS=()

case "${METHOD}" in
    nds|nds_28v_full)
        SOURCE=${EXTERNAL}/neural-deferred-shading
        ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/f2k8_nds_blackwell_20260822
        STAGE=${SLURM_TMPDIR:-/tmp}/f2k1000-nds-${SLURM_JOB_ID}-${INDEX}
        ROOT_METHOD=${STAGE}/neural-deferred-shading
        mkdir -p "${STAGE}"
        cp -R --no-preserve=ownership,mode,timestamps "${SOURCE}" "${ROOT_METHOD}"
        patch --batch --forward -d "${ROOT_METHOD}" -p1 \
            < "${PROJECT}/scripts/HPC/patches/nds_allow_nonmanifold_edges.patch"
        ;;
    nvdiffrec)
        ROOT_METHOD=${EXTERNAL}/nvdiffrec
        ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/f2k8_nvdiffrec_blackwell_20260822
        ;;
    exmesh)
        ROOT_METHOD=${EXTERNAL}/ExMesh
        ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/f2k8_exmesh_blackwell_20260822
        ARGS+=(--exmesh-depth-root "${ROOT}/exmesh_da3_priors")
        ;;
    *) echo "unsupported method ${METHOD}" >&2; exit 2 ;;
esac

export CUDA_HOME=/usr/local/cuda-13.1
export PATH=${ENV}/bin:${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${ENV}/lib:${ENV}/lib64:${LIBRARY_PATH:-}
if [[ "${METHOD}" == nds || "${METHOD}" == nds_28v_full || "${METHOD}" == nvdiffrec ]]; then
    export CPATH=${ENV}/include:${CPATH:-}
else
    unset CPATH
fi
export TORCH_CUDA_ARCH_LIST=12.0
export TCNN_CUDA_ARCHITECTURES=120
export TORCH_EXTENSIONS_DIR=${SLURM_TMPDIR:-/tmp}/torch-extensions-${METHOD}-${SLURM_JOB_ID}-${INDEX}
export PYTHONPATH=${PROJECT}/src:${TORCH_EXTENSIONS_DIR}/renderutils_plugin:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONHASHSEED=7
mkdir -p "${PROJECT}/slurm_logs" "${ROOT}/results" "${TORCH_EXTENSIONS_DIR}"
cd "${PROJECT}"
"${PROJECT_PYTHON}" scripts/evaluate_future2000_external_baseline.py \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --method "${METHOD}" \
    --external-root "${ROOT_METHOD}" \
    --external-python "${ENV}/bin/python" \
    --output-dir "${ROOT}/results" \
    --shard-index "${INDEX}" \
    --shard-count 8 \
    --expected-test-samples 1000 \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01 \
    --retry-failed \
    --fail-on-sample-error \
    "${ARGS[@]}"
