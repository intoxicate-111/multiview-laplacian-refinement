#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/multiview-laplacian-refinement"
readonly CONDA_SETUP="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/miniconda3/etc/profile.d/conda.sh"
readonly QUERY_MANIFEST="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/thingi10k50/sample_50_960/gt_query_full/prepared_manifest.json"
readonly CONFIG="${REPO_ROOT}/configs/learned_laplacian/train_gt_query_50_960_weighted.json"
readonly OUTPUT_DIR="${REPO_ROOT}/runs/learned_laplacian/thingi10k50_gt_query_960_weighted_lr1e4_20260806"

source "${CONDA_SETUP}"
conda activate test
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

test -f "${QUERY_MANIFEST}"
test -f "${CONFIG}"
python -m json.tool "${QUERY_MANIFEST}" >/dev/null
python -m json.tool "${CONFIG}" >/dev/null
python -c 'import sys, torch; ok=torch.cuda.is_available(); print(f"CUDA available: {ok}"); print(f"CUDA device: {torch.cuda.get_device_name(0) if ok else None}"); sys.exit(0 if ok else 1)'

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
python scripts/train_multi_mesh_laplacian.py \
  --manifest "${QUERY_MANIFEST}" \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  2>&1 | tee "${OUTPUT_DIR}/console.log"
