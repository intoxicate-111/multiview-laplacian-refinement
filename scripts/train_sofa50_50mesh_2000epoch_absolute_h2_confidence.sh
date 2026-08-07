#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY_ROOT="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/multiview-laplacian-refinement"
readonly MANIFEST="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"
readonly CONFIG="${REPOSITORY_ROOT}/configs/learned_laplacian/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"
readonly OUTPUT="${REPOSITORY_ROOT}/runs/learned_laplacian/sofa50_50mesh_2000epoch_absolute_h2_confidence"

cd "${REPOSITORY_ROOT}"
exec /home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/miniconda3/bin/conda run \
  --no-capture-output -n test \
  python scripts/train_multi_mesh_laplacian.py \
  --manifest "${MANIFEST}" \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT}" \
  --device cuda \
  "$@"
