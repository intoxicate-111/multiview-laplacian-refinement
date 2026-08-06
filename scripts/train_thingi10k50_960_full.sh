#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/multiview-laplacian-refinement"
readonly CONDA_SETUP="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/miniconda3/etc/profile.d/conda.sh"
readonly MANIFEST="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/thingi10k50/sample_50_960/prepared_manifest.json"
readonly CONFIG="${REPO_ROOT}/configs/learned_laplacian/train_multi_mesh_edge_normalized_50_960.json"
readonly OUTPUT_DIR="${REPO_ROOT}/runs/learned_laplacian/thingi10k50_960_full"

source "${CONDA_SETUP}"
conda activate test
cd "${REPO_ROOT}"

test -f "${MANIFEST}"
test -f "${CONFIG}"
python -m json.tool "${CONFIG}" >/dev/null
python -c 'import sys, torch; available = torch.cuda.is_available(); print(f"CUDA available: {available}"); print(f"CUDA device: {torch.cuda.get_device_name(0) if available else None}"); sys.exit(0 if available else 1)'

if [[ "${1:-}" == "--check" ]]; then
  echo "Training configuration check passed."
  exit 0
fi

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/train_multi_mesh_laplacian.py \
  --manifest "${MANIFEST}" \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  2>&1 | tee "${OUTPUT_DIR}/console.log"
