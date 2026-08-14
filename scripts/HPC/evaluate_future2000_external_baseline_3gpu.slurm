#!/bin/bash
#SBATCH --job-name=f2k_external
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=14-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

METHOD="${1:?usage: sbatch evaluate_future2000_external_baseline_3gpu.slurm METHOD}"
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
EXTERNAL="/networkhome/WMGDS/zhou_c/external_baselines"
MANIFEST="${MANIFEST:-/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json}"
CONFIG="${REPO}/configs/baselines/future2000_external_baselines.json"
PROJECT_PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
OUTPUT="${REPO}/runs/baselines/future2000_gt_adaptive_2000mesh_external_baselines"
ARGS=()

case "${METHOD}" in
    openmvs_refinemesh)
        ROOT="${EXTERNAL}/openMVS"
        INTERFACE="$(find "${ROOT}/make-future2000-v2/bin" -type f -name InterfaceCOLMAP -perm -u+x | head -1)"
        REFINE="$(find "${ROOT}/make-future2000-v2/bin" -type f -name RefineMesh -perm -u+x | head -1)"
        ARGS+=(--interface-colmap "${INTERFACE}" --refine-mesh "${REFINE}")
        ;;
    nds)
        NDS_SOURCE="${EXTERNAL}/neural-deferred-shading"
        export PATH="/networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds/bin:${PATH}"
        NDS_STAGE_ROOT="${SLURM_TMPDIR:-/tmp}/future2000-nds-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
        ROOT="${NDS_STAGE_ROOT}/neural-deferred-shading"
        mkdir -p "${NDS_STAGE_ROOT}"
        cp -a "${NDS_SOURCE}" "${ROOT}"
        patch --batch --forward -d "${ROOT}" -p1 \
            < "${REPO}/scripts/HPC/patches/nds_allow_nonmanifold_edges.patch"
        ARGS+=(--external-python /networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds/bin/python)
        ARGS+=(--retry-failed)
        ;;
    nerf2mesh)
        ROOT="${EXTERNAL}/nerf2mesh"
        ARGS+=(--external-python /networkhome/WMGDS/zhou_c/miniconda3/envs/future_nerf2mesh/bin/python)
        ;;
    exmesh)
        ROOT="${EXTERNAL}/ExMesh"
        ARGS+=(--external-python /networkhome/WMGDS/zhou_c/miniconda3/envs/future_exmesh/bin/python)
        ARGS+=(--exmesh-depth-root "${OUTPUT}/exmesh_da3_priors")
        ;;
    *) echo "Unsupported method: ${METHOD}" >&2; exit 2 ;;
esac

test -f "${MANIFEST}"
test -f "${CONFIG}"
test -x "${PROJECT_PYTHON}"
mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
cd "${REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CUDA_HOME="/networkhome/WMGDS/zhou_c/miniconda3/envs/future_${METHOD}"
nvidia-smi

PYTHONPATH=src "${PROJECT_PYTHON}" scripts/evaluate_future2000_external_baseline.py \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --method "${METHOD}" \
    --external-root "${ROOT}" \
    --output-dir "${OUTPUT}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 3 \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01 \
    "${ARGS[@]}"
