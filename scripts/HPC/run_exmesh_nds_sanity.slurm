#!/bin/bash
#SBATCH --job-name=exmesh_nds_s24
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
NDS=/networkhome/WMGDS/zhou_c/external_baselines/neural-deferred-shading
NDS_PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds/bin/python
EXMESH=/networkhome/WMGDS/zhou_c/external_baselines/ExMesh
EXMESH_PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official/bin/python
DTU=${EXMESH}/workdir/DTU
ROOT_OUTPUT=${PROJECT}/runs/exmesh_baselines
OUTPUT=${ROOT_OUTPUT}/neural_deferred_shading/scan${SCENE_ID}
CONTRACT=${ROOT_OUTPUT}/common_contract.json
NDS_RUN=${OUTPUT}/native_run
FINAL_MESH=${NDS_RUN}/scan${SCENE_ID}/meshes/mesh_002000.obj
EXPECTED_NDS_COMMIT=760e4549f59adaed9adf1bd705599786a00ba6b8

mkdir -p "${OUTPUT}/logs" "${OUTPUT}/configs" "${OUTPUT}/eval" "${OUTPUT}/meshes"
if [[ -f "${OUTPUT}/status.json" ]]; then
    cp "${OUTPUT}/status.json" "${OUTPUT}/logs/status.before_${SLURM_JOB_ID}.json"
fi
test -x "${NDS_PYTHON}"
test -x "${EXMESH_PYTHON}"
test -f "${CONTRACT}"
test "$(git -C "${NDS}" rev-parse HEAD)" = "${EXPECTED_NDS_COMMIT}"
test -z "$(git -C "${NDS}" status --short --untracked-files=no)"

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
    "method": "neural_deferred_shading",
    "success": False,
    "failure_stage": "scan24_sanity",
    "notes": f"command failed with exit code {os.environ['EXIT_VALUE']}",
}, indent=2) + "\n")
PY
    fi
}
trap finish_failure EXIT

cd "${PROJECT}"
"${EXMESH_PYTHON}" scripts/prepare_exmesh_nds_scene.py \
    --contract "${CONTRACT}" \
    --scene-id "${SCENE_ID}" \
    --output-dir "${OUTPUT}/input" \
    --image-mode symlink \
    > "${OUTPUT}/logs/adapter.out"

cp "${PROJECT}/configs/baselines/exmesh_official_suite.json" "${OUTPUT}/configs/"
cp "${OUTPUT}/input/adapter_manifest.json" "${OUTPUT}/configs/"
"${NDS_PYTHON}" -m pip freeze > "${OUTPUT}/configs/environment_pip_freeze.txt"
"${NDS_PYTHON}" - <<'PY' > "${OUTPUT}/configs/environment_runtime.txt"
import platform, torch
print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())
PY

cat > "${OUTPUT}/command.sh" <<EOF
#!/bin/bash
set -euo pipefail
export PYTHONHASHSEED=${SEED}
export CUDA_HOME=/networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds
export PATH=/networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds/bin:\${PATH}
export LD_LIBRARY_PATH=/networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds/lib:/networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds/lib64:\${LD_LIBRARY_PATH:-}
export PYTHONPATH=${NDS}
cd ${NDS}
${NDS_PYTHON} ${PROJECT}/scripts/run_seeded_entrypoint.py \
  --seed ${SEED} --script ${NDS}/reconstruct.py \
  --input_dir ${OUTPUT}/input/views \
  --input_bbox ${OUTPUT}/input/bbox.txt \
  --output_dir ${NDS_RUN} \
  --initial_mesh vh32 \
  --image_scale 1 \
  --iterations 2000 \
  --run_name scan${SCENE_ID} \
  --device 0
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
test -f "${FINAL_MESH}"
cp "${FINAL_MESH}" "${OUTPUT}/meshes/final_mesh.obj"

export PATH=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official/bin:${PATH}
cd "${EXMESH}"
"${EXMESH_PYTHON}" scripts/eval_dtu/evaluate_single_scene.py \
    --input_mesh "${OUTPUT}/meshes/final_mesh.obj" \
    --scan_id "${SCENE_ID}" \
    --output_dir "${OUTPUT}/eval" \
    --mask_dir "${DTU}" \
    --DTU "${DTU}" \
    > "${OUTPUT}/logs/evaluation_${SLURM_JOB_ID}.out" \
    2> "${OUTPUT}/logs/evaluation_${SLURM_JOB_ID}.err"
test -f "${OUTPUT}/eval/results.json"

FINISHED=$(date +%s)
STATUS_PATH="${STATUS}" OUTPUT_PATH="${OUTPUT}" START_VALUE="${STARTED}" \
FINISH_VALUE="${FINISHED}" MONITOR_PATH="${MONITOR}" \
GPU_INVENTORY_PATH="${GPU_INVENTORY}" NDS_REPO="${NDS}" \
NDS_PYTHON_VALUE="${NDS_PYTHON}" "${EXMESH_PYTHON}" - <<'PY'
import json, os, pathlib, subprocess
import trimesh

output = pathlib.Path(os.environ["OUTPUT_PATH"])
metrics = json.loads((output / "eval/results.json").read_text())
manifest = json.loads((output / "input/adapter_manifest.json").read_text())
mesh = trimesh.load(output / "meshes/final_mesh.obj", process=False)
values = []
for line in pathlib.Path(os.environ["MONITOR_PATH"]).read_text().splitlines():
    try:
        values.append(float(line.rsplit(",", 1)[-1].strip()))
    except (ValueError, IndexError):
        pass
payload = {
    "scene_id": 24,
    "method": "neural_deferred_shading",
    "success": True,
    "initialization": "official NDS vh32 visual hull in the ExMesh normalized scene bbox",
    "num_views": manifest["num_views"],
    "metrics": metrics,
    "vertices": int(len(mesh.vertices)),
    "faces": int(len(mesh.faces)),
    "runtime_sec": int(os.environ["FINISH_VALUE"]) - int(os.environ["START_VALUE"]),
    "peak_gpu_memory": max(values) if values else None,
    "gpu_inventory": pathlib.Path(os.environ["GPU_INVENTORY_PATH"]).read_text().strip(),
    "source_repository": "https://github.com/fraunhoferhhi/neural-deferred-shading.git",
    "source_commit": subprocess.check_output(
        ["git", "-C", os.environ["NDS_REPO"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "seed": 7,
    "iterations": 2000,
    "image_scale": 1,
    "views_per_iteration": 1,
    "adapter_manifest": str(output / "input/adapter_manifest.json"),
    "same_rgb_bytes": manifest["image_bytes_unchanged"],
    "mask_source": manifest["mask_source"],
    "resampling": manifest["resampling"],
    "evaluation_gt_used_as_method_input": False,
    "command": str(output / "command.sh"),
    "final_mesh": str(output / "meshes/final_mesh.obj"),
    "notes": "Unmodified official NDS objective/settings with its official vh32 initialization; ExMesh observations and normalized cameras are deterministically adapted. Official ExMesh evaluator used after export.",
    "contract_audit": True,
}
pathlib.Path(os.environ["STATUS_PATH"]).write_text(json.dumps(payload, indent=2) + "\n")
PY

trap - EXIT
