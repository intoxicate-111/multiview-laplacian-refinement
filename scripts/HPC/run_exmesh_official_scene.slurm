#!/bin/bash
#SBATCH --job-name=exmesh_repro
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

SCENES=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
SMOOTH=(3000 3000 3000 3000 3000 3000 1000 3000 3000 3000 3000 1000 3000 1000 1000)
DEPTH=(0.1 0.1 0.03 0.01 0.03 0.01 0.1 0.01 0.01 0.03 0.1 0.01 0.03 0.01 0.03)
INDEX="${SLURM_ARRAY_TASK_ID:?This script must run as an array job.}"
SCENE_ID="${SCENES[$INDEX]}"

PROJECT="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
EXMESH="/networkhome/WMGDS/zhou_c/external_baselines/ExMesh"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official/bin/python"
DTU="${EXMESH}/workdir/DTU"
SCENE="${DTU}/scan${SCENE_ID}"
ROOT_OUTPUT="${PROJECT}/runs/exmesh_baselines"
OUTPUT="${ROOT_OUTPUT}/exmesh_official/scan${SCENE_ID}"
INITIAL_OUTPUT="${ROOT_OUTPUT}/exmesh_initial/scan${SCENE_ID}"
PGSR_MODEL="${EXMESH}/outputs/coarse_meshes/dtu_scan${SCENE_ID}/test"
PGSR_MESH="${PGSR_MODEL}/mesh/tsdf_fusion_post.ply"
MODEL="${OUTPUT}/model"
LPIPS_VGG="/networkhome/WMGDS/zhou_c/.cache/torch/hub/checkpoints/vgg16-397923af.pth"
EXPECTED_COMMIT="09950d283fc5372a09079e30c88d998f1c40b2d0"

mkdir -p \
    "${OUTPUT}/logs" "${OUTPUT}/configs" "${OUTPUT}/meshes" "${OUTPUT}/eval" \
    "${INITIAL_OUTPUT}/logs" "${INITIAL_OUTPUT}/configs" \
    "${INITIAL_OUTPUT}/meshes" "${INITIAL_OUTPUT}/eval"
test -x "${PYTHON}"
test "$(git -C "${EXMESH}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${EXMESH}" status --short --untracked-files=no)"
test -d "${SCENE}/images"
test -d "${SCENE}/mask"
test -d "${SCENE}/mono_priors/da3"
test -f "${DTU}/Points/stl/stl$(printf '%03d' "${SCENE_ID}")_total.ply"
test -f "${DTU}/ObsMask/ObsMask${SCENE_ID}_10.mat"

STARTED="$(date +%s)"
STATUS="${OUTPUT}/status.json"
finish_failure() {
    code=$?
    if [[ ${code} -ne 0 ]]; then
        STATUS_PATH="${STATUS}" SCENE_VALUE="${SCENE_ID}" EXIT_VALUE="${code}" "${PYTHON}" - <<'PY'
import json, os, pathlib
path = pathlib.Path(os.environ["STATUS_PATH"])
path.write_text(json.dumps({
    "scene_id": int(os.environ["SCENE_VALUE"]),
    "method": "exmesh_official",
    "success": False,
    "failure_stage": "official_reproduction",
    "notes": f"command failed with exit code {os.environ['EXIT_VALUE']}",
}, indent=2) + "\n")
PY
    fi
}
trap finish_failure EXIT

cat > "${OUTPUT}/command.sh" <<EOF
#!/bin/bash
set -euo pipefail
export CUDA_HOME=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official
export PATH=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official/bin:\${PATH}
export LD_LIBRARY_PATH=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official/lib:/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official/lib64:\${LD_LIBRARY_PATH:-}
date +%s > ${INITIAL_OUTPUT}/started_epoch.txt
cd ${EXMESH}/PGSR
${PYTHON} train.py -s ../workdir/DTU/scan${SCENE_ID} -m ../outputs/coarse_meshes/dtu_scan${SCENE_ID}/test --quiet -r2 --ncc_scale 0.5
${PYTHON} render.py -m ../outputs/coarse_meshes/dtu_scan${SCENE_ID}/test --quiet --num_cluster 1 --voxel_size 0.01 --max_depth 5.0 --use_depth_filter
cp ${PGSR_MESH} ${SCENE}/mesh.ply
cp ${PGSR_MESH} ${INITIAL_OUTPUT}/meshes/final_mesh.ply
date +%s > ${INITIAL_OUTPUT}/finished_epoch.txt
cd ${EXMESH}
${PYTHON} scripts/eval_dtu/evaluate_single_scene.py --input_mesh ${INITIAL_OUTPUT}/meshes/final_mesh.ply --scan_id ${SCENE_ID} --output_dir ${INITIAL_OUTPUT}/eval --mask_dir ${DTU} --DTU ${DTU}
test -f ${INITIAL_OUTPUT}/eval/results.json
${PYTHON} train.py -s ${SCENE} -m ${MODEL} --lambda_smooth ${SMOOTH[$INDEX]} --lambda_depth ${DEPTH[$INDEX]}
${PYTHON} scripts/eval_dtu/evaluate_single_scene.py --input_mesh ${MODEL}/mesh/auto_mesh_iter_10000.ply --scan_id ${SCENE_ID} --output_dir ${OUTPUT}/eval --mask_dir ${DTU} --DTU ${DTU}
test -f ${OUTPUT}/eval/results.json
EOF
chmod +x "${OUTPUT}/command.sh"

cp "${PROJECT}/configs/baselines/exmesh_official_suite.json" "${OUTPUT}/configs/"
cp "${EXMESH}/arguments/__init__.py" "${OUTPUT}/configs/exmesh_arguments.py"
cp "${EXMESH}/PGSR/arguments/__init__.py" "${OUTPUT}/configs/pgsr_arguments.py"
cp "${PROJECT}/configs/baselines/exmesh_official_suite.json" "${INITIAL_OUTPUT}/configs/"
cp "${EXMESH}/PGSR/arguments/__init__.py" "${INITIAL_OUTPUT}/configs/pgsr_arguments.py"

if [[ -d "${SCENE}/sparse/0" ]]; then
    cp -f "${SCENE}/sparse/0/"* "${SCENE}/sparse/"
fi

MONITOR="${OUTPUT}/logs/gpu_memory.csv"
GPU_INVENTORY="${OUTPUT}/logs/gpu_inventory.csv"
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?Slurm did not expose an allocated GPU}" \
    --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader,nounits > "${GPU_INVENTORY}"
nvidia-smi --id="${CUDA_VISIBLE_DEVICES}" \
    --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -l 1 \
    > "${MONITOR}" &
MONITOR_PID=$!
set +e
/usr/bin/time -v "${OUTPUT}/command.sh" \
    > "${OUTPUT}/logs/pipeline.out" 2> "${OUTPUT}/logs/pipeline.err"
CODE=$?
set -e
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
test "${CODE}" -eq 0

test -f "${PGSR_MESH}"
test -f "${MODEL}/mesh/auto_mesh_iter_10000.ply"
test -f "${OUTPUT}/eval/results.json"
test -f "${INITIAL_OUTPUT}/eval/results.json"
test -f "${LPIPS_VGG}"
cp "${PGSR_MESH}" "${SCENE}/mesh.ply"
cp "${PGSR_MESH}" "${OUTPUT}/meshes/initial_mesh.ply"
cp "${MODEL}/mesh/auto_mesh_iter_10000.ply" "${OUTPUT}/meshes/final_mesh.ply"
cp "${OUTPUT}/command.sh" "${INITIAL_OUTPUT}/command.sh"
cp "${OUTPUT}/logs/pipeline.out" "${INITIAL_OUTPUT}/logs/pipeline.out"
cp "${OUTPUT}/logs/pipeline.err" "${INITIAL_OUTPUT}/logs/pipeline.err"

FINISHED="$(date +%s)"
STATUS_PATH="${STATUS}" OUTPUT_PATH="${OUTPUT}" INITIAL_OUTPUT_PATH="${INITIAL_OUTPUT}" SCENE_VALUE="${SCENE_ID}" \
START_VALUE="${STARTED}" FINISH_VALUE="${FINISHED}" MONITOR_PATH="${MONITOR}" \
GPU_INVENTORY_PATH="${GPU_INVENTORY}" \
LPIPS_VGG_PATH="${LPIPS_VGG}" \
PYTHON_VALUE="${PYTHON}" "${PYTHON}" - <<'PY'
import hashlib, json, os, pathlib, subprocess
import trimesh

output = pathlib.Path(os.environ["OUTPUT_PATH"])
initial_output = pathlib.Path(os.environ["INITIAL_OUTPUT_PATH"])
metrics = json.loads((output / "eval/results.json").read_text())
mesh = trimesh.load(output / "meshes/final_mesh.ply", process=False)
initial_metrics = json.loads((initial_output / "eval/results.json").read_text())
initial_mesh = trimesh.load(initial_output / "meshes/final_mesh.ply", process=False)
peak = None
values = []
for line in pathlib.Path(os.environ["MONITOR_PATH"]).read_text().splitlines():
    try:
        values.append(float(line.rsplit(",", 1)[-1].strip()))
    except (ValueError, IndexError):
        pass
if values:
    peak = max(values)
lpips_vgg = pathlib.Path(os.environ["LPIPS_VGG_PATH"])
digest = hashlib.sha256()
with lpips_vgg.open("rb") as handle:
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
payload = {
    "scene_id": int(os.environ["SCENE_VALUE"]),
    "method": "exmesh_official",
    "success": True,
    "initialization": "official bundled PGSR 5000-step TSDF mesh",
    "num_views": len([
        path for path in
        (pathlib.Path("/networkhome/WMGDS/zhou_c/external_baselines/ExMesh/workdir/DTU") / f"scan{os.environ['SCENE_VALUE']}" / "images").glob("*.png")
        if not path.name.startswith("._")
    ]),
    "metrics": metrics,
    "vertices": int(len(mesh.vertices)),
    "faces": int(len(mesh.faces)),
    "runtime_sec": int(os.environ["FINISH_VALUE"]) - int(os.environ["START_VALUE"]),
    "peak_gpu_memory": peak,
    "gpu_inventory": pathlib.Path(os.environ["GPU_INVENTORY_PATH"]).read_text().strip(),
    "pretrained_models": [{
        "name": "torchvision VGG16 IMAGENET1K_V1",
        "role": "official ExMesh LPIPS feature backbone",
        "path": str(lpips_vgg),
        "sha256": digest.hexdigest(),
    }],
    "source_repository": "https://github.com/Fan-Treasure/ExMesh.git",
    "source_commit": subprocess.check_output(["git", "-C", "/networkhome/WMGDS/zhou_c/external_baselines/ExMesh", "rev-parse", "HEAD"], text=True).strip(),
    "command": str(output / "command.sh"),
    "initial_mesh": str(output / "meshes/initial_mesh.ply"),
    "final_mesh": str(output / "meshes/final_mesh.ply"),
    "notes": "Unmodified official PGSR initialization, ExMesh optimizer, culling, and DTU evaluator.",
}
pathlib.Path(os.environ["STATUS_PATH"]).write_text(json.dumps(payload, indent=2) + "\n")
initial_started = int((initial_output / "started_epoch.txt").read_text().strip())
initial_finished = int((initial_output / "finished_epoch.txt").read_text().strip())
initial_payload = {
    "scene_id": int(os.environ["SCENE_VALUE"]),
    "method": "exmesh_initial",
    "success": True,
    "initialization": "official bundled PGSR 5000-step TSDF mesh",
    "num_views": payload["num_views"],
    "metrics": initial_metrics,
    "vertices": int(len(initial_mesh.vertices)),
    "faces": int(len(initial_mesh.faces)),
    "runtime_sec": initial_finished - initial_started,
    "peak_gpu_memory": None,
    "source_repository": payload["source_repository"],
    "source_commit": payload["source_commit"],
    "command": str(initial_output / "command.sh"),
    "final_mesh": str(initial_output / "meshes/final_mesh.ply"),
    "notes": "Official bundled PGSR initialization evaluated before ExMesh optimization; peak memory was sampled only for the combined pipeline.",
}
(initial_output / "status.json").write_text(json.dumps(initial_payload, indent=2) + "\n")
PY

trap - EXIT
