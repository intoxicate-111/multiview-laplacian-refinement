from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


METHODS = (
    "exmesh_initial",
    "ours",
    "exmesh_official",
    "neural_deferred_shading",
    "nvdiffrec",
    "neuralangelo",
    "matcha",
)
SANITY_METHODS = tuple(method for method in METHODS if method != "exmesh_initial")

SUMMARY_FIELDS = (
    "scene_id",
    "method",
    "initialization",
    "num_views",
    "chamfer",
    "p2s",
    "normal_consistency",
    "fscore",
    "vertices",
    "faces",
    "runtime_sec",
    "peak_gpu_memory",
    "success",
    "notes",
    "official_accuracy_d2s",
    "official_completeness_s2d",
    "official_overall",
)


@dataclass(frozen=True)
class ContractAudit:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def load_suite_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("benchmark") != "exmesh_official_dtu":
        raise ValueError("Expected the independent exmesh_official_dtu benchmark.")
    scene_ids = [int(value) for value in config.get("scene_ids", [])]
    if len(scene_ids) != 15 or len(set(scene_ids)) != 15:
        raise ValueError("The official ExMesh DTU contract requires 15 unique scenes.")
    if tuple(config.get("method_order", ())) != METHODS:
        raise ValueError("Method order does not match the audited suite schema.")
    return config


def extract_common_contract(
    config: Mapping[str, Any], dtu_root: str | Path
) -> dict[str, Any]:
    root = Path(dtu_root).resolve()
    scenes = [
        _extract_scene_contract(root, int(scene_id))
        for scene_id in config["scene_ids"]
    ]
    contract: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "exmesh_official_dtu",
        "source_of_truth": config["official_sources"]["exmesh"],
        "dataset_root": str(root),
        "scene_ids": [scene["scene_id"] for scene in scenes],
        "scenes": scenes,
        "coordinate_system": {
            "camera_projection": "P = world_mat_i @ scale_mat_i; OpenCV decomposition as used by ExMesh",
            "optimized_mesh_frame": "normalized ExMesh model frame",
            "evaluation_frame": "DTU world frame",
            "model_to_evaluation_transform": "scale_mat_0; official culling applies isotropic scale and translation",
            "primary_alignment": "known deterministic transform only; no ICP to GT",
        },
        "initial_geometry": {
            "source": "official ExMesh bundled PGSR at 5000 iterations followed by official TSDF extraction",
            "fixed_graph_for_ours": "scene mesh.ply before ExMesh topology adaptation",
        },
        "preprocessing": {
            "observations": "official 2DGS-preprocessed DTU RGBA images; alpha is the released ExMesh training mask",
            "auxiliary_mask_directory": "present in the archive but not consumed by the released ExMesh training loader",
            "depth_priors": "official ExMesh precomputed DA3 priors",
            "camera_order": "lexicographically sorted image paths, identical to official ExMesh loader/evaluator",
        },
        "evaluation": config["official_exmesh_protocol"]["official_evaluator"],
        "fairness": {
            "same_rgb": True,
            "same_cameras": True,
            "same_normalization": True,
            "same_evaluation_gt": True,
            "gt_geometry_allowed_as_method_input": False,
            "icp_to_gt_primary": False,
        },
    }
    audit = audit_common_contract(contract)
    contract["contract_audit"] = audit.as_dict()
    if not audit.valid:
        raise ValueError("Common contract audit failed: " + "; ".join(audit.errors))
    return contract


def audit_common_contract(contract: Mapping[str, Any]) -> ContractAudit:
    errors: list[str] = []
    warnings: list[str] = []
    if contract.get("benchmark") != "exmesh_official_dtu":
        errors.append("benchmark is not exmesh_official_dtu")
    scenes = contract.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 15:
        errors.append("exactly 15 official DTU scenes are required")
        scenes = []
    forbidden_tokens = ("sofa50", "openmvs")
    for scene in scenes:
        serialized = json.dumps(scene).lower()
        for token in forbidden_tokens:
            if token in serialized:
                errors.append(f"scene {scene.get('scene_id')} contains forbidden source {token}")
        views = scene.get("views", [])
        if not views:
            errors.append(f"scene {scene.get('scene_id')} has no views")
        if scene.get("num_views") != len(views):
            errors.append(f"scene {scene.get('scene_id')} view count mismatch")
        if any(view.get("mask_source") != "rgba_alpha" for view in views):
            errors.append(
                f"scene {scene.get('scene_id')} does not use the official RGBA alpha mask"
            )
        resolutions = {tuple(view["resolution_wh"]) for view in views}
        if len(resolutions) != 1:
            errors.append(f"scene {scene.get('scene_id')} has mixed image resolutions")
        if not scene.get("initial_mesh", {}).get("sha256"):
            errors.append(f"scene {scene.get('scene_id')} lacks the official initial mesh")
        if not scene.get("depth_prior", {}).get("files"):
            errors.append(f"scene {scene.get('scene_id')} lacks DA3 priors")
    evaluator = contract.get("evaluation", {})
    if evaluator.get("sampling_seed_control_supported") is False:
        warnings.append(
            "The released ExMesh evaluator uses an unseeded default_rng shuffle; "
            "the primary protocol is exact but not bitwise deterministic."
        )
    if not evaluator.get("normal_consistency_supported", False):
        warnings.append("Official ExMesh evaluator does not define normal consistency.")
    if not evaluator.get("fscore_supported", False):
        warnings.append("Official ExMesh evaluator does not define F-score.")
    return ContractAudit(not errors, tuple(errors), tuple(warnings))


def save_common_contract(contract: Mapping[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return path


def aggregate_results(
    config: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    root = Path(output_root)
    rows = [
        _load_result_row(root, method, int(scene_id))
        for scene_id in config["scene_ids"]
        for method in METHODS
    ]
    _write_csv(root / "summary.csv", rows)
    methods: dict[str, Any] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        successful = [row for row in method_rows if row["success"]]
        methods[method] = {
            "successful_scenes": len(successful),
            "failed_or_missing_scenes": len(method_rows) - len(successful),
            "metrics": {
                metric: _statistics(successful, metric)
                for metric in (
                    "chamfer",
                    "p2s",
                    "normal_consistency",
                    "fscore",
                    "runtime_sec",
                    "peak_gpu_memory",
                    "official_accuracy_d2s",
                    "official_completeness_s2d",
                    "official_overall",
                )
            },
            "per_scene": method_rows,
        }
    comparisons = {
        other: _paired_comparison(rows, "ours", other)
        for other in METHODS
        if other != "ours"
    }
    reproduction = _exmesh_reproduction_gate(config, rows)
    sanity_scene = int(config["sanity_scene_id"])
    sanity_rows = [
        row
        for row in rows
        if row["scene_id"] == sanity_scene and row["method"] in SANITY_METHODS
    ]
    sanity_gate = {
        "scene_id": sanity_scene,
        "passed": len(sanity_rows) == len(SANITY_METHODS)
        and all(row["success"] for row in sanity_rows),
        "methods": {row["method"]: row["success"] for row in sanity_rows},
        "requires_fixed_camera_visual_frame_audit": True,
    }
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "methods": methods,
        "paired_comparisons": comparisons,
        "official_exmesh_reproduction_gate": reproduction,
        "six_method_sanity_gate": sanity_gate,
        "full_benchmark_authorized": reproduction["passed"] and sanity_gate["passed"],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (root / "BASELINE_REPORT.md").write_text(
        _render_report(config, payload), encoding="utf-8"
    )
    return payload


def _extract_scene_contract(root: Path, scene_id: int) -> dict[str, Any]:
    scene = root / f"scan{scene_id}"
    images = sorted(
        path
        for path in (scene / "images").glob("*.png")
        if not path.name.startswith("._")
    )
    auxiliary_masks = sorted(
        path
        for path in (scene / "mask").glob("*.png")
        if not path.name.startswith("._")
    )
    if not images:
        raise FileNotFoundError(f"No official RGB images in {scene / 'images'}")
    if auxiliary_masks and len(images) != len(auxiliary_masks):
        raise ValueError(
            f"scan{scene_id}: {len(images)} images but {len(auxiliary_masks)} auxiliary masks"
        )
    camera_path = scene / "cameras.npz"
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    camera_dict = np.load(camera_path)
    views = []
    for index, image in enumerate(images):
        world_key = f"world_mat_{index}"
        scale_key = f"scale_mat_{index}"
        if world_key not in camera_dict or scale_key not in camera_dict:
            raise KeyError(f"scan{scene_id}: missing {world_key} or {scale_key}")
        world = camera_dict[world_key].astype(np.float64)
        scale = camera_dict[scale_key].astype(np.float64)
        projection = (world @ scale)[:3, :4]
        intrinsics, camera_to_world = decompose_projection_matrix(projection)
        with Image.open(image) as opened:
            resolution = [int(opened.width), int(opened.height)]
            if "A" not in opened.getbands():
                raise ValueError(
                    f"scan{scene_id}: official observation {image} has no alpha mask"
                )
        auxiliary_mask = auxiliary_masks[index] if auxiliary_masks else None
        views.append(
            {
                "index": index,
                "image_id": image.stem,
                "rgb_path": str(image.resolve()),
                "mask_path": str(image.resolve()),
                "mask_source": "rgba_alpha",
                "auxiliary_mask_path": (
                    str(auxiliary_mask.resolve()) if auxiliary_mask is not None else None
                ),
                "resolution_wh": resolution,
                "exmesh_execution_resolution_wh": [
                    1600 if resolution[0] > 1600 else resolution[0],
                    int(resolution[1] * 1600 / resolution[0])
                    if resolution[0] > 1600
                    else resolution[1],
                ],
                "intrinsics": intrinsics.tolist(),
                "camera_to_world": camera_to_world.tolist(),
                "world_to_camera": np.linalg.inv(camera_to_world).tolist(),
                "world_mat": world.tolist(),
                "scale_mat": scale.tolist(),
            }
        )
    initial_mesh = scene / "mesh.ply"
    depth_dir = scene / "mono_priors" / "da3"
    gt = root / "Points" / "stl" / f"stl{scene_id:03d}_total.ply"
    obs_mask = root / "ObsMask" / f"ObsMask{scene_id}_10.mat"
    ground_plane = root / "ObsMask" / f"Plane{scene_id}.mat"
    required = (initial_mesh, gt, obs_mask, ground_plane)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing official scene inputs: " + ", ".join(missing))
    depth_files = sorted(path for path in depth_dir.rglob("*") if path.is_file())
    return {
        "scene_id": scene_id,
        "scene_root": str(scene.resolve()),
        "num_views": len(views),
        "views": views,
        "camera_archive": str(camera_path.resolve()),
        "camera_archive_sha256": _sha256(camera_path),
        "normalization_transform": views[0]["scale_mat"],
        "initial_mesh": {
            "path": str(initial_mesh.resolve()),
            "sha256": _sha256(initial_mesh),
            "source": "official ExMesh PGSR initialization",
        },
        "depth_prior": {
            "root": str(depth_dir.resolve()),
            "source": "official ExMesh precomputed Depth Anything 3",
            "files": [str(path.resolve()) for path in depth_files],
        },
        "evaluation_gt": {
            "point_cloud": str(gt.resolve()),
            "observation_mask": str(obs_mask.resolve()),
            "ground_plane": str(ground_plane.resolve()),
            "method_input": False,
        },
    }


def prepare_nds_scene(
    contract: Mapping[str, Any],
    scene_id: int,
    output_dir: str | Path,
    *,
    image_mode: str = "symlink",
) -> dict[str, Any]:
    """Adapt one audited ExMesh scene to the official NDS view layout.

    RGB and alpha are preserved byte-for-byte. Only K/R/t text sidecars are
    generated, using the normalized ExMesh model-frame cameras from the common
    contract.
    """

    if image_mode not in {"symlink", "copy"}:
        raise ValueError("image_mode must be 'symlink' or 'copy'")
    if not contract.get("contract_audit", {}).get("valid", False):
        raise ValueError("Refusing to adapt an unaudited common contract")
    scenes = [
        scene
        for scene in contract.get("scenes", [])
        if int(scene.get("scene_id", -1)) == int(scene_id)
    ]
    if len(scenes) != 1:
        raise ValueError(f"Expected exactly one scene {scene_id} in the contract")
    scene = scenes[0]
    root = Path(output_dir)
    views_dir = root / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    bbox = np.asarray([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float64)
    _write_matrix(root / "bbox.txt", bbox)
    records: list[dict[str, Any]] = []
    for view in scene["views"]:
        if view.get("mask_source") != "rgba_alpha":
            raise ValueError("NDS adapter requires the official RGBA alpha mask")
        source = Path(view["rgb_path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        stem = str(view["image_id"])
        destination = views_dir / f"{stem}.png"
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source and _sha256(destination) != _sha256(source):
                raise FileExistsError(f"Conflicting adapted image: {destination}")
        elif image_mode == "symlink":
            destination.symlink_to(source)
        else:
            shutil.copy2(source, destination)

        intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
        world_to_camera = np.asarray(view["world_to_camera"], dtype=np.float64)
        if intrinsics.shape != (3, 3) or world_to_camera.shape != (4, 4):
            raise ValueError(f"Invalid camera shape for view {stem}")
        rotation = world_to_camera[:3, :3]
        translation = world_to_camera[:3, 3]
        expected_projection = np.asarray(view["world_mat"], dtype=np.float64) @ np.asarray(
            view["scale_mat"], dtype=np.float64
        )
        actual_projection = intrinsics @ world_to_camera[:3, :]
        projection_error = _projective_matrix_error(
            expected_projection[:3, :4], actual_projection
        )
        if projection_error > 1e-6:
            raise ValueError(
                f"Camera round-trip error {projection_error:.3g} for view {stem}"
            )
        _write_matrix(views_dir / f"{stem}_k.txt", intrinsics)
        _write_matrix(views_dir / f"{stem}_r.txt", rotation)
        _write_matrix(views_dir / f"{stem}_t.txt", translation)
        records.append(
            {
                "index": int(view["index"]),
                "image_id": stem,
                "source_rgb_alpha": str(source),
                "adapted_image": str(destination),
                "source_sha256": _sha256(source),
                "adapted_sha256": _sha256(destination),
                "resolution_wh": view["resolution_wh"],
                "mask_source": "rgba_alpha",
                "camera_projective_max_abs_error": projection_error,
            }
        )
    manifest = {
        "schema_version": 1,
        "scene_id": int(scene_id),
        "method": "neural_deferred_shading",
        "adapter": "official ExMesh normalized cameras to official NDS OpenCV K/R/t",
        "image_mode": image_mode,
        "image_bytes_unchanged": all(
            item["source_sha256"] == item["adapted_sha256"] for item in records
        ),
        "mask_source": "official ExMesh RGBA alpha",
        "bbox": {
            "path": str(root / "bbox.txt"),
            "bounds": bbox.tolist(),
            "source": "official NDS IDR normalized-scene convention applied in the ExMesh normalized model frame",
            "gt_derived": False,
        },
        "resampling": False,
        "num_views": len(records),
        "views": records,
        "initial_mesh": scene["initial_mesh"],
        "evaluation_gt_used_as_method_input": False,
        "contract_audit": True,
    }
    (root / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_nvdiffrec_scene(
    contract: Mapping[str, Any],
    scene_id: int,
    output_dir: str | Path,
    *,
    image_mode: str = "symlink",
) -> dict[str, Any]:
    """Create a byte-preserving nvdiffrec dataset with exact per-view cameras."""

    if image_mode not in {"symlink", "copy"}:
        raise ValueError("image_mode must be 'symlink' or 'copy'")
    if not contract.get("contract_audit", {}).get("valid", False):
        raise ValueError("Refusing to adapt an unaudited common contract")
    scenes = [
        scene
        for scene in contract.get("scenes", [])
        if int(scene.get("scene_id", -1)) == int(scene_id)
    ]
    if len(scenes) != 1:
        raise ValueError(f"Expected exactly one scene {scene_id} in the contract")
    scene = scenes[0]
    root = Path(output_dir)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    frames: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for view in scene["views"]:
        if view.get("mask_source") != "rgba_alpha":
            raise ValueError("nvdiffrec adapter requires the official RGBA alpha mask")
        source = Path(view["rgb_path"]).resolve()
        destination = images_dir / f"{view['image_id']}.png"
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source and _sha256(destination) != _sha256(source):
                raise FileExistsError(f"Conflicting adapted image: {destination}")
        elif image_mode == "symlink":
            destination.symlink_to(source)
        else:
            shutil.copy2(source, destination)
        intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
        world_to_camera_cv = np.asarray(view["world_to_camera"], dtype=np.float64)
        world_to_camera_gl = cv_to_gl @ world_to_camera_cv
        frames.append(
            {
                "file_path": f"images/{view['image_id']}",
                "intrinsics": intrinsics.tolist(),
                "world_to_camera_opengl": world_to_camera_gl.tolist(),
                "resolution_wh": view["resolution_wh"],
            }
        )
        records.append(
            {
                "index": int(view["index"]),
                "image_id": str(view["image_id"]),
                "source_sha256": _sha256(source),
                "adapted_sha256": _sha256(destination),
                "mask_source": "rgba_alpha",
            }
        )
    width, height = map(int, scene["views"][0]["resolution_wh"])
    nominal_fov_x = 2.0 * np.arctan(
        width / (2.0 * float(scene["views"][0]["intrinsics"][0][0]))
    )
    transforms = {
        "schema": "exmesh_exact_intrinsics_nvdiffrec_adapter_v1",
        "camera_angle_x": float(nominal_fov_x),
        "frames": frames,
    }
    for name in ("transforms_train.json", "transforms_test.json"):
        (root / name).write_text(
            json.dumps(transforms, indent=2) + "\n", encoding="utf-8"
        )
    manifest = {
        "schema_version": 1,
        "scene_id": int(scene_id),
        "method": "nvdiffrec",
        "adapter": "exact ExMesh K and OpenCV-to-OpenGL world-to-camera conversion",
        "image_mode": image_mode,
        "image_bytes_unchanged": all(
            item["source_sha256"] == item["adapted_sha256"] for item in records
        ),
        "mask_source": "official ExMesh RGBA alpha",
        "resampling": False,
        "resolution_wh": [width, height],
        "num_views": len(frames),
        "train_and_validation_frames_identical": True,
        "evaluation_gt_used_as_method_input": False,
        "camera_adapter_required": True,
        "camera_adapter_reason": "official DatasetNERF supports only centered single-FOV intrinsics",
        "views": records,
        "contract_audit": True,
    }
    (root / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def nvdiffrec_projection_from_intrinsics(
    intrinsics: np.ndarray,
    resolution_wh: Sequence[int],
    near: float,
    far: float,
) -> np.ndarray:
    """Return nvdiffrec/nvdiffrast clip projection for an exact OpenCV K."""

    k = np.asarray(intrinsics, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    width, height = map(float, resolution_wh)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    return np.asarray(
        [
            [2.0 * fx / width, 0.0, 1.0 - 2.0 * cx / width, 0.0],
            [0.0, -2.0 * fy / height, 1.0 - 2.0 * cy / height, 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def prepare_neuralangelo_scene(
    contract: Mapping[str, Any],
    scene_id: int,
    output_dir: str | Path,
    *,
    image_mode: str = "symlink",
) -> dict[str, Any]:
    """Create Neuralangelo's native DTU transforms.json in the ExMesh frame."""

    if image_mode not in {"symlink", "copy"}:
        raise ValueError("image_mode must be 'symlink' or 'copy'")
    if not contract.get("contract_audit", {}).get("valid", False):
        raise ValueError("Refusing to adapt an unaudited common contract")
    scene = next(
        (
            item
            for item in contract.get("scenes", [])
            if int(item.get("scene_id", -1)) == int(scene_id)
        ),
        None,
    )
    if scene is None:
        raise ValueError(f"Scene {scene_id} is absent from the contract")
    root = Path(output_dir)
    images_dir = root / "image"
    images_dir.mkdir(parents=True, exist_ok=True)
    cv_to_gl_columns = np.diag([1.0, -1.0, -1.0, 1.0])
    frames = []
    records = []
    intrinsics = []
    for view in scene["views"]:
        source = Path(view["rgb_path"]).resolve()
        destination = images_dir / f"{view['image_id']}.png"
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source and _sha256(destination) != _sha256(source):
                raise FileExistsError(f"Conflicting adapted image: {destination}")
        elif image_mode == "symlink":
            destination.symlink_to(source)
        else:
            shutil.copy2(source, destination)
        k = np.asarray(view["intrinsics"], dtype=np.float64)
        c2w_cv = np.asarray(view["camera_to_world"], dtype=np.float64)
        c2w_gl = c2w_cv @ cv_to_gl_columns
        frames.append(
            {
                "file_path": f"image/{view['image_id']}.png",
                "transform_matrix": c2w_gl.tolist(),
                "intrinsics": k.tolist(),
            }
        )
        intrinsics.append(k)
        records.append(
            {
                "index": int(view["index"]),
                "image_id": str(view["image_id"]),
                "source_sha256": _sha256(source),
                "adapted_sha256": _sha256(destination),
            }
        )
    k_stack = np.stack(intrinsics)
    k_mean = k_stack.mean(axis=0)
    width, height = map(int, scene["views"][0]["resolution_wh"])
    transforms = {
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "is_fisheye": False,
        "frames": frames,
        "camera_angle_x": float(2 * np.arctan(width / (2 * k_mean[0, 0]))),
        "camera_angle_y": float(2 * np.arctan(height / (2 * k_mean[1, 1]))),
        "fl_x": float(k_mean[0, 0]),
        "fl_y": float(k_mean[1, 1]),
        "cx": float(k_mean[0, 2]),
        "cy": float(k_mean[1, 2]),
        "sk_x": float(k_mean[0, 1]),
        "sk_y": float(k_mean[1, 0]),
        "w": width,
        "h": height,
        "aabb_scale": 1.0,
        "sphere_center": [0.0, 0.0, 0.0],
        "sphere_radius": 1.0,
    }
    (root / "transforms.json").write_text(
        json.dumps(transforms, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "scene_id": int(scene_id),
        "method": "neuralangelo",
        "adapter": "official Neuralangelo DTU transforms format in the ExMesh normalized frame",
        "image_bytes_unchanged": all(
            item["source_sha256"] == item["adapted_sha256"] for item in records
        ),
        "resampling_in_adapter": False,
        "native_resolution_wh": [width, height],
        "num_views": len(frames),
        "per_view_intrinsics_embedded": True,
        "global_intrinsics_max_abs_deviation_px": float(
            np.max(np.abs(k_stack - k_mean))
        ),
        "mask_policy": "official Neuralangelo loader consumes RGB and ignores alpha",
        "sphere_center": [0.0, 0.0, 0.0],
        "sphere_radius": 1.0,
        "evaluation_gt_used_as_method_input": False,
        "views": records,
        "contract_audit": True,
    }
    (root / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _projective_matrix_error(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    denominator = float(np.vdot(actual, actual))
    if denominator <= 0:
        return float("inf")
    scale = float(np.vdot(actual, expected) / denominator)
    normalized = max(float(np.max(np.abs(expected))), 1.0)
    return float(np.max(np.abs(expected - scale * actual)) / normalized)


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    payload = "\n".join(
        " ".join(f"{float(value):.17g}" for value in row)
        for row in np.atleast_2d(matrix)
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Conflicting adapted camera file: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def decompose_projection_matrix(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match ExMesh's OpenCV ``load_K_Rt_from_P`` camera convention."""

    projection = np.asarray(projection, dtype=np.float64)
    if projection.shape != (3, 4):
        raise ValueError("Projection matrix must have shape (3, 4).")
    try:
        import cv2  # type: ignore

        intrinsics, rotation, translation, *_ = cv2.decomposeProjectionMatrix(
            projection
        )
        intrinsics = intrinsics / intrinsics[2, 2]
        center = (translation[:3] / translation[3])[:, 0]
    except ImportError:
        intrinsics, rotation = _rq(projection[:, :3])
        diagonal = np.sign(np.diag(intrinsics))
        diagonal[diagonal == 0] = 1
        correction = np.diag(diagonal)
        intrinsics = intrinsics @ correction
        rotation = correction @ rotation
        if np.linalg.det(rotation) < 0:
            intrinsics[:, -1] *= -1
            rotation[-1, :] *= -1
        intrinsics = intrinsics / intrinsics[2, 2]
        center = -np.linalg.solve(projection[:, :3], projection[:, 3])
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = rotation.T
    camera_to_world[:3, 3] = center
    return intrinsics, camera_to_world


def _rq(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, r = np.linalg.qr(np.flipud(matrix).T)
    r = np.flipud(r.T)
    r = np.fliplr(r)
    q = q.T
    q = np.flipud(q)
    return r, q


def _load_result_row(root: Path, method: str, scene_id: int) -> dict[str, Any]:
    path = root / method / f"scan{scene_id}" / "status.json"
    row = {field: None for field in SUMMARY_FIELDS}
    row.update(
        {
            "scene_id": scene_id,
            "method": method,
            "initialization": "",
            "num_views": None,
            "success": False,
            "notes": "not_run: status.json is absent",
        }
    )
    if not path.is_file():
        return row
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    row.update(
        {
            "initialization": payload.get("initialization", ""),
            "num_views": payload.get("num_views"),
            "chamfer": metrics.get("chamfer", metrics.get("overall")),
            "p2s": metrics.get("p2s"),
            "normal_consistency": metrics.get("normal_consistency"),
            "fscore": metrics.get("fscore"),
            "vertices": payload.get("vertices"),
            "faces": payload.get("faces"),
            "runtime_sec": payload.get("runtime_sec"),
            "peak_gpu_memory": payload.get("peak_gpu_memory"),
            "success": payload.get("success") is True,
            "notes": payload.get("notes", ""),
            "official_accuracy_d2s": metrics.get("mean_d2s"),
            "official_completeness_s2d": metrics.get("mean_s2d"),
            "official_overall": metrics.get("overall"),
        }
    )
    return row


def _statistics(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float] | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
    }


def _paired_comparison(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[str, Any]:
    keyed = {(int(row["scene_id"]), str(row["method"])): row for row in rows}
    pairs = []
    for scene_id in sorted({int(row["scene_id"]) for row in rows}):
        a = keyed[(scene_id, left)]
        b = keyed[(scene_id, right)]
        if not a["success"] or not b["success"]:
            continue
        values: dict[str, Any] = {"scene_id": scene_id}
        for metric in ("chamfer", "p2s", "normal_consistency", "fscore"):
            if a.get(metric) is not None and b.get(metric) is not None:
                values[f"{left}_minus_{right}_{metric}"] = float(a[metric]) - float(
                    b[metric]
                )
        pairs.append(values)
    metric_summary: dict[str, Any] = {}
    for metric in ("chamfer", "p2s", "normal_consistency", "fscore"):
        field = f"{left}_minus_{right}_{metric}"
        deltas = [float(pair[field]) for pair in pairs if field in pair]
        if not deltas:
            metric_summary[metric] = None
            continue
        lower_is_better = metric in {"chamfer", "p2s"}
        better = sum(delta < 0 if lower_is_better else delta > 0 for delta in deltas)
        worse = sum(delta > 0 if lower_is_better else delta < 0 for delta in deltas)
        metric_summary[metric] = {
            "successful_pairs": len(deltas),
            "mean_left_minus_right": statistics.fmean(deltas),
            "median_left_minus_right": statistics.median(deltas),
            "left_better": better,
            "right_better": worse,
            "ties": len(deltas) - better - worse,
        }
    return {
        "successful_pairs": len(pairs),
        "metrics": metric_summary,
        "per_scene": pairs,
    }


def _exmesh_reproduction_gate(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    reproduced = {
        int(row["scene_id"]): float(row["official_overall"])
        for row in rows
        if row["method"] == "exmesh_official"
        and row["success"]
        and row["official_overall"] is not None
    }
    expected = {
        int(key): float(value)
        for key, value in config["official_exmesh_protocol"][
            "paper_overall_cd_mm"
        ].items()
    }
    threshold = config["official_exmesh_protocol"]["reproduction_gate"]
    if set(reproduced) != set(expected):
        return {
            "passed": False,
            "reason": f"requires all 15 scenes; found {len(reproduced)}",
            "successful_scenes": len(reproduced),
        }
    errors = [abs(reproduced[key] - expected[key]) for key in sorted(expected)]
    mean_reproduced = statistics.fmean(reproduced.values())
    mean_expected = float(
        config["official_exmesh_protocol"]["paper_mean_overall_cd_mm"]
    )
    mean_difference = abs(mean_reproduced - mean_expected)
    scene_mae = statistics.fmean(errors)
    passed = (
        mean_difference
        <= float(threshold["maximum_absolute_mean_difference_mm"])
        and scene_mae
        <= float(threshold["maximum_per_scene_mean_absolute_error_mm"])
    )
    return {
        "passed": passed,
        "reproduced_mean_mm": mean_reproduced,
        "paper_mean_mm": mean_expected,
        "absolute_mean_difference_mm": mean_difference,
        "per_scene_mae_mm": scene_mae,
        "thresholds": threshold,
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(config: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    sources = config["official_sources"]
    lines = [
        "# ExMesh-protocol external baseline report",
        "",
        "> This benchmark is independent from the project's synthetic-data experiments. "
        "All primary observations, cameras, normalization, initialization, and evaluation "
        "are defined by the official ExMesh DTU release.",
        "",
        "## Protocol gates",
        "",
        f"- Official ExMesh reproduction: `{summary['official_exmesh_reproduction_gate']['passed']}`",
        f"- Six-method scene-{config['sanity_scene_id']} sanity gate: `{summary['six_method_sanity_gate']['passed']}`",
        f"- Full benchmark authorized: `{summary['full_benchmark_authorized']}`",
        "",
        "## Official implementations",
        "",
        "| Method | Venue | Repository | Commit |",
        "|---|---:|---|---|",
    ]
    for method in (
        "exmesh",
        "neural_deferred_shading",
        "nvdiffrec",
        "neuralangelo",
        "matcha",
    ):
        item = sources[method]
        lines.append(
            f"| {item['paper']} | {item['venue']} {item['year']} | "
            f"{item['repository']} | `{item['commit']}` |"
        )
    lines.extend(
        [
            "",
            "## Observation, initialization, and prior audit",
            "",
            "| Method | Observations | Initialization | Additional priors |",
            "|---|---|---|---|",
        ]
    )
    for method in METHODS:
        protocol = config["method_protocols"][method]
        lines.append(
            f"| {method} | {protocol['observations']} | "
            f"{protocol['initialization']} | {protocol['additional_priors']} |"
        )
    lines.extend(
        [
            "",
            "## Metric contract",
            "",
            "The released ExMesh DTU evaluator reports accuracy (`mean_d2s`), "
            "completeness (`mean_s2d`), and their average (`overall`, called CD in "
            "the ExMesh paper). It does not implement point-to-surface distance, "
            "normal consistency, or F-score; those fields remain null unless an "
            "official common implementation is explicitly added and separately labeled.",
            "",
            "## Current result status",
            "",
            "| Method | Successful | Missing/failed | CD mean | CD median | CD std | Runtime mean (s) | Peak memory mean (MiB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        item = summary["methods"][method]
        metric = item["metrics"]["official_overall"]
        runtime = item["metrics"]["runtime_sec"]
        memory = item["metrics"]["peak_gpu_memory"]
        cd_values = ["—", "—", "—"] if metric is None else [
            f"{metric['mean']:.6f}",
            f"{metric['median']:.6f}",
            f"{metric['std']:.6f}",
        ]
        lines.append(
            f"| {method} | {item['successful_scenes']} | "
            f"{item['failed_or_missing_scenes']} | {' | '.join(cd_values)} | "
            f"{_markdown_mean(runtime)} | {_markdown_mean(memory)} |"
        )
    lines.extend(["", "## Per-scene official evaluator CD (mm)", ""])
    lines.append("| Scene | " + " | ".join(METHODS) + " |")
    lines.append("|---:|" + "---:|" * len(METHODS))
    scene_ids = [int(value) for value in config["scene_ids"]]
    for scene_id in scene_ids:
        values = []
        for method in METHODS:
            rows = summary["methods"][method]["per_scene"]
            row = next(row for row in rows if int(row["scene_id"]) == scene_id)
            value = row.get("official_overall") if row.get("success") else None
            values.append("—" if value is None else f"{float(value):.6f}")
        lines.append(f"| {scene_id} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Paired comparison against ours",
            "",
            "Negative CD deltas mean ours is lower. Counts and deltas include only "
            "scenes where both methods completed.",
            "",
            "| Comparator | Pairs | Mean ours−other CD | Ours better | Other better | Ties |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for other, comparison in summary["paired_comparisons"].items():
        metric = comparison["metrics"]["chamfer"]
        if metric is None:
            lines.append(f"| {other} | 0 | — | 0 | 0 | 0 |")
        else:
            lines.append(
                f"| {other} | {metric['successful_pairs']} | "
                f"{metric['mean_left_minus_right']:.6f} | {metric['left_better']} | "
                f"{metric['right_better']} | {metric['ties']} |"
            )
    lines.extend(
        [
            "",
            "## Known protocol limitations and blockers",
            "",
            "- The released ExMesh evaluator performs an unseeded random shuffle before "
            "radius downsampling, so exact bitwise repeatability is not exposed by the official code.",
            "- The paper describes a 7k-step 2DGS initialization, whereas the released "
            "README and bundled runner use 5k-step PGSR. This suite reproduces the released "
            "repository first and records that paper/code protocol difference.",
            "- The linked archive contains 1554×1162 images. Released ExMesh leaves these "
            "at native resolution (`resolution=-1` only limits widths above 1600), while "
            "the paper states 800×600. The released-code reproduction uses 1554×1162.",
            "- The learned method's non-evaluation training source is not defined by the "
            "scene-wise ExMesh protocol. No evaluation-scene GT may be used to resolve this.",
            "- Full external-baseline execution remains gated on the official ExMesh "
            "15-scene reproduction and the fixed-camera six-method sanity check.",
            "",
            "## Qualitative panels",
            "",
            "Panels are generated only after the sanity gate verifies a shared camera/world frame. "
            "Required order: GT, ExMesh initial, ours, ExMesh, Neural Deferred Shading, "
            "nvdiffrec, Neuralangelo, MAtCha.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_mean(value: Mapping[str, float] | None) -> str:
    return "—" if value is None else f"{value['mean']:.3f}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
