#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/multiview-laplacian-refinement"
readonly CONDA_SETUP="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/miniconda3/etc/profile.d/conda.sh"
readonly DATA_ROOT="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/thingi10k50/sample_50_960"
readonly SOURCE_MANIFEST="${DATA_ROOT}/prepared_manifest.json"
readonly QUERY_ROOT="${DATA_ROOT}/gt_query_smoke"
readonly QUERY_MANIFEST="${QUERY_ROOT}/prepared_manifest.json"
readonly CONFIG="${REPO_ROOT}/configs/learned_laplacian/train_gt_query_960_smoke.json"
readonly OUTPUT_DIR="${REPO_ROOT}/runs/learned_laplacian/gt_query_960_smoke"

source "${CONDA_SETUP}"
conda activate test
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

test -f "${SOURCE_MANIFEST}"
test -f "${CONFIG}"
python -m json.tool "${CONFIG}" >/dev/null
python -c 'import sys, torch; ok = torch.cuda.is_available(); print(f"CUDA available: {ok}"); print(f"CUDA device: {torch.cuda.get_device_name(0) if ok else None}"); sys.exit(0 if ok else 1)'

if [[ "${1:-}" == "--check" ]]; then
  echo "GT-query smoke configuration check passed."
  exit 0
fi

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi

python scripts/prepare_gt_query_manifest.py \
  --source-manifest "${SOURCE_MANIFEST}" \
  --output-manifest "${QUERY_MANIFEST}" \
  --train-limit 2 \
  --validation-limit 1 \
  --test-limit 0 \
  --image-size 256 \
  --overwrite

mkdir -p "${OUTPUT_DIR}"
python scripts/train_multi_mesh_laplacian.py \
  --manifest "${QUERY_MANIFEST}" \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  2>&1 | tee "${OUTPUT_DIR}/console.log"
