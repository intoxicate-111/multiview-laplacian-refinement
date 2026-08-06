#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/multiview-laplacian-refinement"
readonly CONDA_SETUP="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/miniconda3/etc/profile.d/conda.sh"
readonly DATA_ROOT="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/thingi10k50/sample_50_960"
readonly SOURCE_MANIFEST="${DATA_ROOT}/prepared_manifest.json"
readonly QUERY_ROOT="${DATA_ROOT}/gt_query_full"
readonly QUERY_MANIFEST="${QUERY_ROOT}/prepared_manifest.json"
readonly CONFIG="${REPO_ROOT}/configs/learned_laplacian/train_gt_query_50_960.json"
readonly OUTPUT_DIR="${REPO_ROOT}/runs/learned_laplacian/thingi10k50_gt_query_960_full"

source "${CONDA_SETUP}"
conda activate test
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

test -f "${SOURCE_MANIFEST}"
test -f "${CONFIG}"
python -m json.tool "${SOURCE_MANIFEST}" >/dev/null
python -m json.tool "${CONFIG}" >/dev/null
python -c 'import sys, torch; ok = torch.cuda.is_available(); print(f"CUDA available: {ok}"); print(f"CUDA device: {torch.cuda.get_device_name(0) if ok else None}"); sys.exit(0 if ok else 1)'

if [[ "${1:-}" == "--check" ]]; then
  echo "Full GT-query training configuration check passed."
  echo "Source manifest: ${SOURCE_MANIFEST}"
  echo "GT-query manifest: ${QUERY_MANIFEST}"
  echo "Output directory: ${OUTPUT_DIR}"
  exit 0
fi

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi

if [[ ! -f "${QUERY_MANIFEST}" ]]; then
  echo "Preparing all 50 GT-query samples at full 960 resolution..."
  python scripts/prepare_gt_query_manifest.py \
    --source-manifest "${SOURCE_MANIFEST}" \
    --output-manifest "${QUERY_MANIFEST}" \
    --overwrite
else
  echo "Reusing existing GT-query manifest: ${QUERY_MANIFEST}"
fi

python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); m=json.loads(p.read_text()); counts={s:sum(x.get("split")==s for x in m["samples"]) for s in ("train","validation","test")}; print(f"GT-query split counts: {counts}"); assert counts == {"train":40,"validation":5,"test":5}; assert m.get("query_training_mode") == "gt_vertex_perturbation_v1"' "${QUERY_MANIFEST}"

mkdir -p "${OUTPUT_DIR}"
python scripts/train_multi_mesh_laplacian.py \
  --manifest "${QUERY_MANIFEST}" \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  2>&1 | tee "${OUTPUT_DIR}/console.log"
