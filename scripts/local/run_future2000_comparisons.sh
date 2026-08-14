#!/bin/bash

set -euo pipefail

TASK="${1:-list}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_ROOT="${F2K_LOCAL_ROOT:-${REPO}/.external/future2000_local}"
DATA_ROOT="${F2K_DATA_ROOT:-${LOCAL_ROOT}/future2000_gt_adaptive_synthetic_current_28view_v2}"
MANIFEST="${F2K_MANIFEST:-${DATA_ROOT}/manifest.json}"
LAPLACIAN_RUN="${F2K_LAPLACIAN_RUN:-${LOCAL_ROOT}/runs/laplacian}"
DISPLACEMENT_RUN="${F2K_DISPLACEMENT_RUN:-${LOCAL_ROOT}/runs/displacement}"
LEARNED_ANALYSIS="${F2K_LEARNED_ANALYSIS:-${LOCAL_ROOT}/outputs/learned_analysis}"
EXTERNAL_OUTPUT="${F2K_EXTERNAL_OUTPUT:-${LOCAL_ROOT}/outputs/external}"
QUALITATIVE_DIR="${F2K_QUALITATIVE_DIR:-${LOCAL_ROOT}/outputs/qualitative}"
REPORT_DIR="${F2K_REPORT_DIR:-${LOCAL_ROOT}/outputs/report}"
LOG_DIR="${F2K_LOG_DIR:-${LOCAL_ROOT}/logs}"
CONFIG="${F2K_EXTERNAL_CONFIG:-${REPO}/configs/baselines/future2000_external_baselines.json}"
PROJECT_PYTHON="${F2K_PROJECT_PYTHON:-${REPO}/../miniconda3/envs/test/bin/python}"
CONDA_ROOT="${F2K_CONDA_ROOT:-${REPO}/../miniconda3}"
EXTERNAL_ROOT="${F2K_EXTERNAL_ROOT:-${REPO}/.external}"
OPENMVS_ROOT="${F2K_OPENMVS_ROOT:-${EXTERNAL_ROOT}/openMVS}"
NDS_ROOT="${F2K_NDS_ROOT:-${EXTERNAL_ROOT}/neural-deferred-shading}"
NERF2MESH_ROOT="${F2K_NERF2MESH_ROOT:-${EXTERNAL_ROOT}/nerf2mesh}"
EXMESH_ROOT="${F2K_EXMESH_ROOT:-${EXTERNAL_ROOT}/ExMesh}"
DA3_ROOT="${F2K_DA3_ROOT:-${EXTERNAL_ROOT}/depth-anything-3}"
OPENMVS_INTERFACE="${F2K_OPENMVS_INTERFACE:-${OPENMVS_ROOT}/make-local/bin/InterfaceCOLMAP}"
OPENMVS_REFINE="${F2K_OPENMVS_REFINE:-${OPENMVS_ROOT}/make-local/bin/RefineMesh}"
NDS_PYTHON="${F2K_NDS_PYTHON:-${CONDA_ROOT}/envs/future_nds/bin/python}"
NERF2MESH_PYTHON="${F2K_NERF2MESH_PYTHON:-${CONDA_ROOT}/envs/future_nerf2mesh/bin/python}"
EXMESH_PYTHON="${F2K_EXMESH_PYTHON:-${CONDA_ROOT}/envs/future_exmesh/bin/python}"
DA3_PYTHON="${F2K_DA3_PYTHON:-${CONDA_ROOT}/envs/future_da3/bin/python}"

IFS=',' read -r -a GPUS <<< "${F2K_GPUS:-0}"
SHARD_COUNT="${F2K_SHARD_COUNT:-${#GPUS[@]}}"
if (( SHARD_COUNT < 1 || ${#GPUS[@]} < 1 )); then
    echo "F2K_GPUS and F2K_SHARD_COUNT must describe at least one local GPU." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

checkpoint_exists() {
    [[ -f "$1/checkpoint_best.pt" || -f "$1/checkpoint_latest.pt" || -f "$1/best.pt" ]]
}

require_file() {
    [[ -f "$1" ]] || { echo "Missing file: $1" >&2; return 1; }
}

require_executable() {
    [[ -x "$1" ]] || { echo "Missing executable: $1" >&2; return 1; }
}

preflight_gpu() {
    require_executable "${PROJECT_PYTHON}"
    "${PROJECT_PYTHON}" -c 'import torch,sys; print("torch",torch.__version__,"cuda",torch.cuda.is_available(),"devices",torch.cuda.device_count()); sys.exit(0 if torch.cuda.is_available() else 1)'
    local available
    available="$(CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")" "${PROJECT_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
    if (( available < ${#GPUS[@]} )); then
        echo "Requested ${#GPUS[@]} GPUs but only ${available} are visible." >&2
        return 1
    fi
}

preflight_data() {
    require_file "${MANIFEST}"
    require_file "${CONFIG}"
}

preflight_learned_runs() {
    require_file "${LAPLACIAN_RUN}/run_config.json"
    require_file "${DISPLACEMENT_RUN}/run_config.json"
    checkpoint_exists "${LAPLACIAN_RUN}" || { echo "Missing Laplacian checkpoint" >&2; return 1; }
    checkpoint_exists "${DISPLACEMENT_RUN}" || { echo "Missing displacement checkpoint" >&2; return 1; }
}

discover_openmvs_binaries() {
    if [[ ! -x "${OPENMVS_INTERFACE}" ]]; then
        OPENMVS_INTERFACE="$(find "${OPENMVS_ROOT}/make-local/bin" -type f -name InterfaceCOLMAP -perm -u+x -print -quit 2>/dev/null || true)"
    fi
    if [[ ! -x "${OPENMVS_REFINE}" ]]; then
        OPENMVS_REFINE="$(find "${OPENMVS_ROOT}/make-local/bin" -type f -name RefineMesh -perm -u+x -print -quit 2>/dev/null || true)"
    fi
}

preflight_external() {
    local method="${1:-all}"
    case "${method}" in
        openmvs_refinemesh)
            discover_openmvs_binaries
            require_executable "${OPENMVS_INTERFACE}"
            require_executable "${OPENMVS_REFINE}"
            ;;
        nds)
            require_executable "${NDS_PYTHON}"
            require_file "${NDS_ROOT}/reconstruct.py"
            ;;
        nerf2mesh)
            require_executable "${NERF2MESH_PYTHON}"
            require_file "${NERF2MESH_ROOT}/main.py"
            ;;
        exmesh)
            require_executable "${EXMESH_PYTHON}"
            require_file "${EXMESH_ROOT}/train.py"
            ;;
        da3)
            require_executable "${DA3_PYTHON}"
            require_file "${DA3_ROOT}/pyproject.toml"
            ;;
        all)
            preflight_external openmvs_refinemesh
            preflight_external nds
            preflight_external nerf2mesh
            preflight_external da3
            preflight_external exmesh
            ;;
        *) echo "Unknown preflight method: ${method}" >&2; return 2 ;;
    esac
}

run_shards() {
    local label="$1"
    shift
    local -a base=("$@")
    local -a pids=()
    local shard gpu log
    for ((shard=0; shard<SHARD_COUNT; shard++)); do
        gpu="${GPUS[$((shard % ${#GPUS[@]}))]}"
        log="${LOG_DIR}/${label}_shard_$(printf '%03d' "${shard}").log"
        echo "Starting ${label} shard=${shard}/${SHARD_COUNT} gpu=${gpu} log=${log}"
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            "${base[@]}" --shard-index "${shard}" --shard-count "${SHARD_COUNT}"
        ) >"${log}" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || { echo "${label} failed; inspect ${LOG_DIR}." >&2; return 1; }
}

run_learned() {
    preflight_data
    preflight_learned_runs
    preflight_gpu
    mkdir -p "${LEARNED_ANALYSIS}/shards"
    run_shards learned \
        "${PROJECT_PYTHON}" "${REPO}/scripts/evaluate_future2000_laplacian_vs_displacement.py" \
        --manifest "${MANIFEST}" \
        --laplacian-run "${LAPLACIAN_RUN}" \
        --displacement-run "${DISPLACEMENT_RUN}" \
        --output-dir "${LEARNED_ANALYSIS}" \
        --device cuda \
        --surface-samples 3000 \
        --metric-seed 7 \
        --fscore-threshold 0.01
    "${PROJECT_PYTHON}" "${REPO}/scripts/merge_future2000_learned_evaluation.py" \
        --manifest "${MANIFEST}" \
        --output-dir "${LEARNED_ANALYSIS}" \
        --shard-count "${SHARD_COUNT}"
}

run_external() {
    local method="$1"
    preflight_data
    preflight_gpu
    preflight_external "${method}"
    local root external_python=""
    local -a extra=()
    case "${method}" in
        openmvs_refinemesh)
            root="${OPENMVS_ROOT}"
            extra+=(--interface-colmap "${OPENMVS_INTERFACE}" --refine-mesh "${OPENMVS_REFINE}")
            ;;
        nds)
            root="${NDS_ROOT}"
            external_python="${NDS_PYTHON}"
            ;;
        nerf2mesh)
            root="${NERF2MESH_ROOT}"
            external_python="${NERF2MESH_PYTHON}"
            ;;
        exmesh)
            root="${EXMESH_ROOT}"
            external_python="${EXMESH_PYTHON}"
            extra+=(--exmesh-depth-root "${EXTERNAL_OUTPUT}/exmesh_da3_priors")
            ;;
        *) echo "Unknown external method: ${method}" >&2; return 2 ;;
    esac
    if [[ -n "${external_python}" ]]; then
        extra+=(--external-python "${external_python}")
    fi
    run_shards "${method}" \
        "${PROJECT_PYTHON}" "${REPO}/scripts/evaluate_future2000_external_baseline.py" \
        --manifest "${MANIFEST}" \
        --config "${CONFIG}" \
        --method "${method}" \
        --external-root "${root}" \
        --output-dir "${EXTERNAL_OUTPUT}" \
        --surface-samples 3000 \
        --metric-seed 7 \
        --fscore-threshold 0.01 \
        "${extra[@]}"
    "${PROJECT_PYTHON}" "${REPO}/scripts/merge_future2000_external_baseline.py" \
        --manifest "${MANIFEST}" \
        --output-dir "${EXTERNAL_OUTPUT}" \
        --method "${method}" \
        --shard-count "${SHARD_COUNT}"
}

run_da3() {
    preflight_data
    preflight_gpu
    preflight_external da3
    run_shards da3 \
        "${DA3_PYTHON}" "${REPO}/scripts/prepare_future2000_exmesh_da3_priors.py" \
        --manifest "${MANIFEST}" \
        --config "${CONFIG}" \
        --da3-root "${DA3_ROOT}" \
        --output-dir "${EXTERNAL_OUTPUT}/exmesh_da3_priors" \
        --device cuda
}

run_qualitative() {
    "${PROJECT_PYTHON}" "${REPO}/scripts/render_future2000_unified_qualitative.py" \
        --manifest "${MANIFEST}" \
        --learned-analysis "${LEARNED_ANALYSIS}" \
        --external-output "${EXTERNAL_OUTPUT}" \
        --output-dir "${QUALITATIVE_DIR}" \
        --count 4 \
        --image-size 320
}

run_report() {
    "${PROJECT_PYTHON}" "${REPO}/scripts/generate_future2000_unified_report.py" \
        --laplacian-run "${LAPLACIAN_RUN}" \
        --displacement-run "${DISPLACEMENT_RUN}" \
        --learned-analysis "${LEARNED_ANALYSIS}" \
        --external-output "${EXTERNAL_OUTPUT}" \
        --qualitative-dir "${QUALITATIVE_DIR}" \
        --output-dir "${REPORT_DIR}"
}

list_tasks() {
    printf '%s\n' \
        'preflight          validate local data, checkpoints, GPU and runtimes' \
        'learned            learned-Laplacian vs direct-displacement evaluation + merge' \
        'openmvs            OpenMVS RefineMesh evaluation + merge' \
        'nds                NDS evaluation + merge' \
        'nerf2mesh          NeRF2Mesh evaluation + merge' \
        'da3                prepare RGB-only DA3 priors for 200 test objects' \
        'exmesh             ExMesh evaluation + merge (requires da3)' \
        'qualitative        render at least four unified comparison figures' \
        'report             generate final Markdown/JSON/status artifacts' \
        'all                run the strict sequence above locally'
}

case "${TASK}" in
    list) list_tasks ;;
    preflight) preflight_data; preflight_learned_runs; preflight_gpu; preflight_external all ;;
    learned) run_learned ;;
    openmvs) run_external openmvs_refinemesh ;;
    nds) run_external nds ;;
    nerf2mesh) run_external nerf2mesh ;;
    da3) run_da3 ;;
    exmesh) run_external exmesh ;;
    qualitative) run_qualitative ;;
    report) run_report ;;
    all)
        preflight_data
        preflight_learned_runs
        preflight_gpu
        preflight_external all
        run_learned
        run_external openmvs_refinemesh
        run_external nds
        run_external nerf2mesh
        run_da3
        run_external exmesh
        run_qualitative
        run_report
        ;;
    *) echo "Unknown task: ${TASK}" >&2; list_tasks; exit 2 ;;
esac
