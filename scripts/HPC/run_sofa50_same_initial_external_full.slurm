#!/bin/bash
#SBATCH --job-name=s50_same_ext_full
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

METHOD=${1:?usage: sbatch run_sofa50_same_initial_external_full.slurm METHOD}
PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
EXTERNAL=/networkhome/WMGDS/zhou_c/external_baselines
MANIFEST=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820/benchmark_manifest.json
CONFIG=${PROJECT}/configs/baselines/sofa50_same_initial_benchmark.json
GATE=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820/SANITY_GATE.json
OUTPUT=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820/full
PROJECT_PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
ARGS=()

case "${METHOD}" in
    nds)
        NDS_SOURCE=${EXTERNAL}/neural-deferred-shading
        export PATH=/networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds/bin:${PATH}
        STAGE_ROOT=${SLURM_TMPDIR:-/tmp}/sofa50-nds-${SLURM_JOB_ID}
        ROOT=${STAGE_ROOT}/neural-deferred-shading
        mkdir -p "${STAGE_ROOT}"
        cp -a "${NDS_SOURCE}" "${ROOT}"
        patch --batch --forward -d "${ROOT}" -p1 < "${PROJECT}/scripts/HPC/patches/nds_allow_nonmanifold_edges.patch"
        ARGS+=(--external-python /networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds/bin/python)
        ;;
    exmesh)
        ROOT=${EXTERNAL}/ExMesh
        ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official
        export CUDA_HOME=${ENV}
        export PATH=${ENV}/bin:${PATH}
        export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/lib64:${LD_LIBRARY_PATH:-}
        ARGS+=(--external-python ${ENV}/bin/python)
        ARGS+=(--exmesh-depth-root ${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820/full_exmesh_da3_priors)
        ;;
    nvdiffrec)
        ROOT=${EXTERNAL}/nvdiffrec
        ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/nvdiffrec_official
        export CUDA_HOME=${ENV}
        export PATH=${ENV}/bin:${PATH}
        export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/lib64:${LD_LIBRARY_PATH:-}
        export TCNN_CUDA_ARCHITECTURES=89
        export TORCH_CUDA_ARCH_LIST=8.9
        ARGS+=(--external-python ${ENV}/bin/python)
        ;;
    *) echo "Unsupported controlled method: ${METHOD}" >&2; exit 2 ;;
esac

test -f "${MANIFEST}"
test -f "${CONFIG}"
test -f "${GATE}"
"${PROJECT_PYTHON}" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["full_benchmark_submission_allowed"] else 2)' "${GATE}"
mkdir -p "${PROJECT}/slurm_logs" "${OUTPUT}"
cd "${PROJECT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONHASHSEED=7
PYTHONPATH=src "${PROJECT_PYTHON}" scripts/evaluate_future2000_external_baseline.py \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --method "${METHOD}" \
    --external-root "${ROOT}" \
    --output-dir "${OUTPUT}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 25 \
    --expected-test-samples 25 \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01 \
    "${ARGS[@]}"
