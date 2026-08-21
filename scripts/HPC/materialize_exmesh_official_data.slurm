#!/bin/bash
#SBATCH --job-name=exmesh_unpack
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
EXMESH="/networkhome/WMGDS/zhou_c/external_baselines/ExMesh"
DOWNLOAD="${EXMESH}/downloads_official"
WORKDIR="${EXMESH}/workdir"
DTU="${WORKDIR}/DTU"
GT_ROOT="${EXMESH}/official_dtu_gt"
STAGING="${EXMESH}/workdir_staging_${SLURM_JOB_ID}"
GT_STAGING="${EXMESH}/official_dtu_gt_staging_${SLURM_JOB_ID}"
OUTPUT="${PROJECT}/runs/exmesh_baselines/exmesh_official"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/bin/python"

mkdir -p "${PROJECT}/slurm_logs" "${OUTPUT}"
test -s "${DOWNLOAD}/2dgs/dtu.tar.gz"
test -s "${DOWNLOAD}/exmesh_dtu_da3"
test -s "${DOWNLOAD}/SampleSet.zip"
test -s "${DOWNLOAD}/Points.zip"
sha256sum \
    "${DOWNLOAD}/2dgs/dtu.tar.gz" \
    "${DOWNLOAD}/2dgs/dtu_results.zip" \
    "${DOWNLOAD}/exmesh_dtu_da3" \
    "${DOWNLOAD}/SampleSet.zip" \
    "${DOWNLOAD}/Points.zip" \
    > "${OUTPUT}/official_data_sha256sums.txt"

if [[ -e "${DTU}" || -e "${GT_ROOT}" ]]; then
    # A previous materialization may have completed extraction but failed its
    # audit. Reuse it only when both roots exist; never merge a partial tree.
    test -d "${DTU}"
    test -d "${GT_ROOT}"
else
    mkdir -p "${STAGING}" "${GT_STAGING}"
    tar -xzf "${DOWNLOAD}/2dgs/dtu.tar.gz" -C "${STAGING}"
    unzip -nq "${DOWNLOAD}/exmesh_dtu_da3" -d "${STAGING}"
    unzip -nq "${DOWNLOAD}/SampleSet.zip" -d "${GT_STAGING}"
    unzip -nq "${DOWNLOAD}/Points.zip" -d "${GT_STAGING}"

    test -d "${STAGING}/DTU"
    mv "${GT_STAGING}" "${GT_ROOT}"
    mkdir -p "${WORKDIR}"
    mv "${STAGING}/DTU" "${DTU}"
    rmdir "${STAGING}"
fi

OBS_SOURCE="${GT_ROOT}/SampleSet/MVS Data/ObsMask"
# Points.zip contains the complete 124-scene reference set. SampleSet.zip also
# contains a two-file Points/stl example, so a generic `find -print -quit`
# silently selects the wrong directory.
POINTS_SOURCE="${GT_ROOT}/Points"
test -d "${OBS_SOURCE}"
test -d "${POINTS_SOURCE}/stl"
ln -sfn "${OBS_SOURCE}" "${DTU}/ObsMask"
ln -sfn "${POINTS_SOURCE}" "${DTU}/Points"

# Some mask folders in the official tarball contain AppleDouble sidecars named
# `._NNN.png`. OpenCV cannot decode them, while the released ExMesh culling
# script blindly globs every `*.png`. Remove only those derived sidecars from
# the materialized copy; the downloaded archive and its SHA-256 remain intact.
APPLEDOUBLE_COUNT="$(find "${DTU}" -type f -path '*/mask/._*.png' -print | wc -l)"
find "${DTU}" -type f -path '*/mask/._*.png' -delete

DTU_ROOT="${DTU}" OUTPUT_PATH="${OUTPUT}/data_audit.json" \
APPLEDOUBLE_REMOVED="${APPLEDOUBLE_COUNT}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

scene_ids = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
root = Path(os.environ["DTU_ROOT"])
scenes = []
errors = []
for scene_id in scene_ids:
    scene = root / f"scan{scene_id}"
    images = sorted(p for p in (scene / "images").glob("*.png") if not p.name.startswith("._"))
    # The official tarball contains macOS AppleDouble `._*.png` entries in
    # several mask directories. They are metadata, not additional views.
    masks = sorted(p for p in (scene / "mask").glob("*.png") if not p.name.startswith("._"))
    depth_npz = sorted((scene / "mono_priors/da3").glob("*.npz"))
    depth_png = sorted((scene / "mono_priors/da3").glob("*.png"))
    required = [
        scene / "cameras.npz",
        root / "Points/stl" / f"stl{scene_id:03d}_total.ply",
        root / "ObsMask" / f"ObsMask{scene_id}_10.mat",
        root / "ObsMask" / f"Plane{scene_id}.mat",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if not images or len(images) != len(masks) or len(images) != len(depth_npz):
        errors.append(
            f"scan{scene_id}: images={len(images)} masks={len(masks)} "
            f"depth_npz={len(depth_npz)}"
        )
    if missing:
        errors.append(f"scan{scene_id}: missing {missing}")
    scenes.append({
        "scene_id": scene_id,
        "images": len(images),
        "masks": len(masks),
        "da3_npz": len(depth_npz),
        "da3_png": len(depth_png),
        "missing": missing,
    })
payload = {
    "status": "ready" if not errors else "failed",
    "source_2dgs": "official ExMesh-linked 2DGS-preprocessed DTU archive",
    "source_da3": "official ExMesh precomputed DTU DA3 archive",
    "source_gt": "official DTU SampleSet.zip and Points.zip",
    "dtu_root": str(root),
    "materialization_only_cleanup": {
        "appledouble_mask_sidecars_removed": int(os.environ["APPLEDOUBLE_REMOVED"]),
        "raw_archives_modified": False,
    },
    "scenes": scenes,
    "errors": errors,
}
Path(os.environ["OUTPUT_PATH"]).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
if errors:
    raise SystemExit(3)
PY
