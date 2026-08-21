#!/bin/bash
#SBATCH --job-name=exmesh_nvdr_s24
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

SCENE_ID=24
SEED=7
PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
NVDIFFREC=/networkhome/WMGDS/zhou_c/external_baselines/nvdiffrec
ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/nvdiffrec_official
PYTHON=${ENV}/bin/python
EXMESH=/networkhome/WMGDS/zhou_c/external_baselines/ExMesh
EXMESH_PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official/bin/python
DTU=${EXMESH}/workdir/DTU
ROOT_OUTPUT=${PROJECT}/runs/exmesh_baselines
OUTPUT=${ROOT_OUTPUT}/nvdiffrec/scan${SCENE_ID}
CONTRACT=${ROOT_OUTPUT}/common_contract.json
CONFIG=${PROJECT}/configs/baselines/nvdiffrec_exmesh_scan24.json
WORKDIR=${OUTPUT}/workdir
FINAL_NATIVE=${WORKDIR}/out/native_run/mesh/mesh.obj
EXPECTED_COMMIT=abf3a34b1eb6e782abffefc2462c7e9bcd89f9bb

mkdir -p "${OUTPUT}/logs" "${OUTPUT}/configs" "${OUTPUT}/eval" \
    "${OUTPUT}/meshes" "${WORKDIR}"
ln -sfn "${NVDIFFREC}/data" "${WORKDIR}/data"
if [[ -f "${OUTPUT}/status.json" ]]; then
    cp "${OUTPUT}/status.json" "${OUTPUT}/logs/status.before_${SLURM_JOB_ID}.json"
fi
test -x "${PYTHON}"
test -f "${CONTRACT}"
test -f "${CONFIG}"
test -f "${NVDIFFREC}/data/tets/128_tets.npz"
test "$(git -C "${NVDIFFREC}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${NVDIFFREC}" status --short --untracked-files=no)"
"${PYTHON}" - <<'PY'
import nvdiffrast.torch, tinycudann, torch, xatlas
assert torch.cuda.is_available()
PY

STARTED=$(date +%s)
STATUS=${OUTPUT}/status.json
finish_failure() {
    code=$?
    if [[ ${code} -ne 0 ]]; then
        STATUS_PATH="${STATUS}" EXIT_VALUE="${code}" "${EXMESH_PYTHON}" - <<'PY'
import json, os, pathlib
path = pathlib.Path(os.environ["STATUS_PATH"])
path.write_text(json.dumps({
    "scene_id": 24,
    "method": "nvdiffrec",
    "success": False,
    "failure_stage": "scan24_sanity",
    "notes": f"command failed with exit code {os.environ['EXIT_VALUE']}",
}, indent=2) + "\n")
PY
    fi
}
trap finish_failure EXIT

cd "${PROJECT}"
"${EXMESH_PYTHON}" scripts/prepare_exmesh_nvdiffrec_scene.py \
    --contract "${CONTRACT}" --scene-id "${SCENE_ID}" \
    --output-dir "${OUTPUT}/input" --image-mode symlink \
    > "${OUTPUT}/logs/adapter_${SLURM_JOB_ID}.out"
cp "${CONFIG}" "${OUTPUT}/configs/"
cp "${OUTPUT}/input/adapter_manifest.json" "${OUTPUT}/configs/"
"${PYTHON}" -m pip freeze > "${OUTPUT}/configs/environment_pip_freeze.txt"

cat > "${OUTPUT}/command.sh" <<EOF
#!/bin/bash
set -euo pipefail
export PYTHONHASHSEED=${SEED}
export CUDA_HOME=${ENV}
export PATH=${ENV}/bin:\${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/lib64:\${LD_LIBRARY_PATH:-}
export PYTHONPATH=${NVDIFFREC}
export TCNN_CUDA_ARCHITECTURES=89
export TORCH_CUDA_ARCH_LIST=8.9
cd ${WORKDIR}
${PYTHON} ${PROJECT}/scripts/run_nvdiffrec_exmesh.py \
  --seed ${SEED} --nvdiffrec-root ${NVDIFFREC} \
  --config ${CONFIG}
EOF
chmod +x "${OUTPUT}/command.sh"

MONITOR=${OUTPUT}/logs/gpu_memory_${SLURM_JOB_ID}.csv
GPU_INVENTORY=${OUTPUT}/logs/gpu_inventory_${SLURM_JOB_ID}.csv
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?Slurm did not expose an allocated GPU}" \
    --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader,nounits > "${GPU_INVENTORY}"
nvidia-smi --id="${CUDA_VISIBLE_DEVICES}" \
    --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -l 1 \
    > "${MONITOR}" &
MONITOR_PID=$!
set +e
/usr/bin/time -v "${OUTPUT}/command.sh" \
    > "${OUTPUT}/logs/pipeline_${SLURM_JOB_ID}.out" \
    2> "${OUTPUT}/logs/pipeline_${SLURM_JOB_ID}.err"
CODE=$?
set -e
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
test "${CODE}" -eq 0
test -f "${FINAL_NATIVE}"
cp "${FINAL_NATIVE}" "${OUTPUT}/meshes/final_mesh.obj"

export PATH=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official/bin:${PATH}
cd "${EXMESH}"
"${EXMESH_PYTHON}" scripts/eval_dtu/evaluate_single_scene.py \
    --input_mesh "${OUTPUT}/meshes/final_mesh.obj" \
    --scan_id "${SCENE_ID}" --output_dir "${OUTPUT}/eval" \
    --mask_dir "${DTU}" --DTU "${DTU}" \
    > "${OUTPUT}/logs/evaluation_${SLURM_JOB_ID}.out" \
    2> "${OUTPUT}/logs/evaluation_${SLURM_JOB_ID}.err"
test -f "${OUTPUT}/eval/results.json"

FINISHED=$(date +%s)
STATUS_PATH="${STATUS}" OUTPUT_PATH="${OUTPUT}" START_VALUE="${STARTED}" \
FINISH_VALUE="${FINISHED}" MONITOR_PATH="${MONITOR}" \
GPU_INVENTORY_PATH="${GPU_INVENTORY}" NVDIFFREC_REPO="${NVDIFFREC}" \
"${EXMESH_PYTHON}" - <<'PY'
import json, os, pathlib, subprocess
import trimesh

output = pathlib.Path(os.environ["OUTPUT_PATH"])
metrics = json.loads((output / "eval/results.json").read_text())
manifest = json.loads((output / "input/adapter_manifest.json").read_text())
mesh = trimesh.load(output / "meshes/final_mesh.obj", process=False)
values = []
for line in pathlib.Path(os.environ["MONITOR_PATH"]).read_text().splitlines():
    try: values.append(float(line.rsplit(",", 1)[-1].strip()))
    except (ValueError, IndexError): pass
payload = {
    "scene_id": 24,
    "method": "nvdiffrec",
    "success": True,
    "initialization": "official DMTet grid-128 reconstruction; no GT geometry",
    "num_views": manifest["num_views"],
    "metrics": metrics,
    "vertices": int(len(mesh.vertices)),
    "faces": int(len(mesh.faces)),
    "runtime_sec": int(os.environ["FINISH_VALUE"]) - int(os.environ["START_VALUE"]),
    "peak_gpu_memory": max(values) if values else None,
    "gpu_inventory": pathlib.Path(os.environ["GPU_INVENTORY_PATH"]).read_text().strip(),
    "source_repository": "https://github.com/NVlabs/nvdiffrec.git",
    "source_commit": subprocess.check_output(
        ["git", "-C", os.environ["NVDIFFREC_REPO"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "seed": 7,
    "official_iterations_per_pass": 5000,
    "passes": 2,
    "batch": 1,
    "protocol_differences": [
        "batch reduced from the example config's 8 to 1 for native 1554x1162 observations",
        "exact-camera DatasetNERF overlay supports per-view fx/fy/cx/cy; optimization code is unchanged"
    ],
    "adapter_manifest": str(output / "input/adapter_manifest.json"),
    "same_rgb_bytes": manifest["image_bytes_unchanged"],
    "mask_source": manifest["mask_source"],
    "resampling": manifest["resampling"],
    "evaluation_gt_used_as_method_input": False,
    "command": str(output / "command.sh"),
    "final_mesh": str(output / "meshes/final_mesh.obj"),
    "notes": "Official nvdiffrec DMTet/material/light optimization; only the image-camera loader is adapted to retain exact ExMesh intrinsics and frame.",
    "contract_audit": True,
}
pathlib.Path(os.environ["STATUS_PATH"]).write_text(json.dumps(payload, indent=2) + "\n")
PY

trap - EXIT
