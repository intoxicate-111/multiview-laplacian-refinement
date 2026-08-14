#!/bin/bash

set -euo pipefail

TASK="${1:-all}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_ROOT="${F2K_LOCAL_ROOT:-${REPO}/.external/future2000_local}"
REMOTE_HOST="${F2K_REMOTE_HOST:-zhou_c@hpc-login-03}"
REMOTE_DATA="${F2K_REMOTE_DATA:-/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2}"
REMOTE_RGB="${F2K_REMOTE_RGB:-/networkhome/WMGDS/zhou_c/future2000_compact/multiview_960/rendered}"
REMOTE_RUNS="${F2K_REMOTE_RUNS:-/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian}"
LAPLACIAN_NAME="future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k_seed7"
DISPLACEMENT_NAME="future2000_gt_adaptive_2000mesh_expanded_current_28view_displacement_20k_seed7"
DATA_ROOT="${LOCAL_ROOT}/future2000_gt_adaptive_synthetic_current_28view_v2"
RGB_ROOT="${LOCAL_ROOT}/future2000_compact/multiview_960/rendered"

sync_data() {
    local metadata
    mkdir -p "${DATA_ROOT}/prepared/test" "${RGB_ROOT}"
    for metadata in \
        manifest.json \
        nested_camera_layout_28.json \
        oracle_validation.json \
        gt_adaptive_validation.json; do
        rsync -a --info=progress2 \
            "${REMOTE_HOST}:${REMOTE_DATA}/${metadata}" \
            "${DATA_ROOT}/${metadata}"
    done
    rsync -a --info=progress2 \
        "${REMOTE_HOST}:${REMOTE_DATA}/prepared/test/" \
        "${DATA_ROOT}/prepared/test/"
    # The prepared samples use ../future2000_compact/... lazy paths.  The full
    # rendered RGB tree is only ~642 MB and preserves that relative contract.
    rsync -a --info=progress2 \
        "${REMOTE_HOST}:${REMOTE_RGB}/" \
        "${RGB_ROOT}/"
}

sync_run() {
    local remote_name="$1" local_name="$2"
    mkdir -p "${LOCAL_ROOT}/runs/${local_name}"
    rsync -a --info=progress2 \
        --include='config.json' \
        --include='run_config.json' \
        --include='metrics.json' \
        --include='training_history.json' \
        --include='best.pt' \
        --include='checkpoint_best.pt' \
        --include='checkpoint_latest.pt' \
        --exclude='*' \
        "${REMOTE_HOST}:${REMOTE_RUNS}/${remote_name}/" \
        "${LOCAL_ROOT}/runs/${local_name}/"
}

case "${TASK}" in
    data) sync_data ;;
    laplacian) sync_run "${LAPLACIAN_NAME}" laplacian ;;
    displacement) sync_run "${DISPLACEMENT_NAME}" displacement ;;
    all)
        sync_data
        sync_run "${LAPLACIAN_NAME}" laplacian
        sync_run "${DISPLACEMENT_NAME}" displacement
        ;;
    *) echo "usage: $0 data|laplacian|displacement|all" >&2; exit 2 ;;
esac
