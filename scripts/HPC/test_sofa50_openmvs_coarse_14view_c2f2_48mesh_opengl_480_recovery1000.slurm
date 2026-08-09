#!/bin/bash
#SBATCH --job-name=openmvs48_c2f2_gl_r1000
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/openmvs_coarse14_c2f2_gl480_r1000_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/openmvs_coarse14_c2f2_gl480_r1000_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

SCRIPT="${REPO}/scripts/test_sofa50_openmvs48_c2f2_3seed.py"

ORIGINAL_C2F2_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_50000step_3seed"
SOURCE_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"
COARSE_MODELS_ROOT="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/openmvs_texture_test_v6_48view/reconstruction/models"

OUTPUT_DIR="${REPO}/runs/learned_laplacian/sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480_recovery1000"

VISIBILITY_SIZE="${VISIBILITY_SIZE:-480}"
RECOVERY_ITERS="${RECOVERY_ITERS:-1000}"

RUNTIME_C2F2_ROOT="${OUTPUT_DIR}/runtime_c2f2_recovery_${RECOVERY_ITERS}"

cd "${REPO}"
mkdir -p "${REPO}/slurm_logs" "${OUTPUT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

printf '%s\n' \
  "============================================================" \
  "OpenMVS coarse -> original 14-view C2F2 inference" \
  "Visibility backend: OpenGL" \
  "Visibility size:    ${VISIBILITY_SIZE}" \
  "Recovery iterations:${RECOVERY_ITERS}" \
  "Host:               $(hostname)" \
  "SLURM job:          ${SLURM_JOB_ID}" \
  "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" \
  "Original C2F2 root: ${ORIGINAL_C2F2_ROOT}" \
  "Source manifest:    ${SOURCE_MANIFEST}" \
  "Coarse models:      ${COARSE_MODELS_ROOT}" \
  "Output:             ${OUTPUT_DIR}" \
  "============================================================"

test -f "${SCRIPT}"
test -f "${SOURCE_MANIFEST}"
test -d "${COARSE_MODELS_ROOT}"

for SEED in 7 17 27; do
  test -d "${ORIGINAL_C2F2_ROOT}/seed_${SEED}"
done

MESH_COUNT="$(find "${COARSE_MODELS_ROOT}" -mindepth 2 -maxdepth 2 -name coarse.obj | wc -l)"
echo "Found coarse meshes: ${MESH_COUNT}"

if [[ "${MESH_COUNT}" -eq 0 ]]; then
  echo "ERROR: no coarse.obj files found under ${COARSE_MODELS_ROOT}" >&2
  exit 2
fi

echo
echo "============================================================"
echo "Preparing runtime C2F2 configs"
echo "============================================================"

rm -rf "${RUNTIME_C2F2_ROOT}"
mkdir -p "${RUNTIME_C2F2_ROOT}"

for SEED in 7 17 27; do
  SRC="${ORIGINAL_C2F2_ROOT}/seed_${SEED}"
  DST="${RUNTIME_C2F2_ROOT}/seed_${SEED}"
  mkdir -p "${DST}"

  "${CONDA}" run --no-capture-output -n test \
    python - "${SRC}" "${DST}" "${RECOVERY_ITERS}" <<'PY'
import json
import os
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()
num_iters = int(sys.argv[3])

config_path = None
for name in ("config.json", "launch_config.json", "run_config.json"):
    candidate = src / name
    if candidate.is_file():
        config_path = candidate
        break
if config_path is None:
    raise FileNotFoundError(f"No config file found under {src}")

checkpoint_path = None
for name in ("checkpoint_best.pt", "best.pt"):
    candidate = src / name
    if candidate.is_file():
        checkpoint_path = candidate
        break
if checkpoint_path is None:
    raise FileNotFoundError(f"No best checkpoint found under {src}")

cfg = json.loads(config_path.read_text(encoding="utf-8"))
old_iters = cfg.get("recovery", {}).get("num_iters")
cfg.setdefault("recovery", {})["num_iters"] = num_iters

runtime_config = dst / "config.json"
runtime_config.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

runtime_checkpoint = dst / checkpoint_path.name
if runtime_checkpoint.exists() or runtime_checkpoint.is_symlink():
    runtime_checkpoint.unlink()
os.symlink(checkpoint_path, runtime_checkpoint)

print(f"[runtime-config] seed={cfg.get('seed')} recovery.num_iters={old_iters} -> {num_iters}")
print(f"[runtime-config] config={runtime_config}")
print(f"[runtime-config] checkpoint={runtime_checkpoint} -> {checkpoint_path}")
PY
done

C2F2_ROOT="${RUNTIME_C2F2_ROOT}"

echo
echo "Runtime C2F2 root: ${C2F2_ROOT}"

for SEED in 7 17 27; do
  test -f "${C2F2_ROOT}/seed_${SEED}/config.json"
  if [[ ! -f "${C2F2_ROOT}/seed_${SEED}/checkpoint_best.pt" && ! -f "${C2F2_ROOT}/seed_${SEED}/best.pt" ]]; then
    echo "ERROR: runtime checkpoint missing for seed ${SEED}" >&2
    exit 2
  fi
done

echo
echo "============================================================"
echo "GPU"
echo "============================================================"
nvidia-smi

echo
echo "============================================================"
echo "Starting OpenGL visibility + CUDA prediction + recovery"
echo "============================================================"

PYTHONPATH="${REPO}/src" "${CONDA}" run \
  --no-capture-output \
  -n test \
  python "${SCRIPT}" \
    --repo-root "${REPO}" \
    --c2f2-root "${C2F2_ROOT}" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --coarse-models-root "${COARSE_MODELS_ROOT}" \
    --mesh-name coarse.obj \
    --output-dir "${OUTPUT_DIR}" \
    --split all \
    --device cuda \
    --visibility-backend opengl \
    --visibility-size "${VISIBILITY_SIZE}"

printf '%s\n' \
  "============================================================" \
  "Finished" \
  "Recovery iterations: ${RECOVERY_ITERS}" \
  "Summary:   ${OUTPUT_DIR}/summary.json" \
  "Per mesh:  ${OUTPUT_DIR}/per_mesh_metrics.csv" \
  "Aggregate: ${OUTPUT_DIR}/aggregate_metrics.csv" \
  "Pred:      ${OUTPUT_DIR}/predictions" \
  "Recovered: ${OUTPUT_DIR}/recovered" \
  "Runtime configs: ${RUNTIME_C2F2_ROOT}" \
  "============================================================"
