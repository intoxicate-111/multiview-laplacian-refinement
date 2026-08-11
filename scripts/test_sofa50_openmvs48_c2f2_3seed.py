#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate 48-view OpenMVS Sofa50 coarse meshes with the three C2F2 seeds.

The 48 auxiliary textured views are used only to acquire ``coarse.obj``.
Prediction uses the original Sofa50 14-view RGB/cameras from the existing
GT-query manifest.  No GT-to-current-graph Laplacian transfer is constructed.
The current-graph target fields stored in prepared samples are identity
placeholders required by the sample schema and are never used as supervision or
an oracle during inference/recovery.

Default evaluation is the held-out Sofa50 test split (5 meshes).  The script
runs seed 7/17/27 independently and then a simple prediction-space ensemble
(mean delta_hat and mean confidence), using the same current graph and renderer
visibility for all four variants.

Outputs:
  <output-dir>/prepared_query/<sample_id>.pt
  <output-dir>/visibility/<sample_id>.npz
  <output-dir>/predictions/seed_<seed>/<sample_id>.npz
  <output-dir>/recovered/seed_<seed>/<sample_id>/predicted_refined.obj
  <output-dir>/recovered/ensemble_mean/<sample_id>/predicted_refined.obj
  <output-dir>/per_mesh_metrics.csv
  <output-dir>/aggregate_metrics.csv
  <output-dir>/summary.json
"""

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SEEDS = (7, 17, 27)
VISIBILITY_FIELD = "visibility_backface_and_occlusion"
VARIANTS = tuple(f"seed_{seed}" for seed in SEEDS) + ("ensemble_mean",)


def expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(value), indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def config_path(run_dir: Path) -> Path:
    for name in ("config.json", "launch_config.json", "run_config.json"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No config file found under {run_dir}")


def checkpoint_path(run_dir: Path) -> Path:
    for name in ("checkpoint_best.pt", "best.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No best checkpoint found under {run_dir}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_c2f2_root(repo_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        root = expand(requested)
        _require_three_seeds(root)
        return root
    runs = repo_root / "runs" / "learned_laplacian"
    candidates = [
        runs / "sofa50_c2_f2_50000step_3seed",
        runs / "sofa50_c2_f2_1920_20000step_3seed",
        runs / "sofa50_c2_f2_1920_50000step_3seed",
    ]
    valid = [path for path in candidates if _has_three_seeds(path)]
    if not valid:
        raise FileNotFoundError(
            "Could not find a complete C2F2 3-seed root. Pass --c2f2-root. "
            "Checked: " + ", ".join(str(path) for path in candidates)
        )
    # The repository comparison scripts define the 50k 960 root as the canonical
    # C2F2 3-seed location, so preserve that precedence when both exist.
    return valid[0]


def _has_three_seeds(root: Path) -> bool:
    try:
        _require_three_seeds(root)
    except (FileNotFoundError, ValueError):
        return False
    return True


def _require_three_seeds(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"C2F2 root not found: {root}")
    for seed in SEEDS:
        run_dir = root / f"seed_{seed}"
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing C2F2 seed directory: {run_dir}")
        config = read_json(config_path(run_dir))
        actual_seed = int(config.get("seed", -1))
        if actual_seed != seed:
            raise ValueError(f"{run_dir}: config seed={actual_seed}, expected {seed}")
        checkpoint_path(run_dir)


def model_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    image = config.get("image_encoder", {})
    model = config.get("model", {})
    return {
        "feature_dim": image.get("feature_dim"),
        "first_stride": image.get("first_stride"),
        "second_stride": image.get("second_stride"),
        "hidden_dim": model.get("hidden_dim"),
        "num_graph_layers": model.get("num_graph_layers"),
        "target_scaling": config.get("target_scaling"),
        "recovery": config.get("recovery"),
        "experiment_resolution": config.get("experiment_resolution"),
    }


def infer_source_manifest(c2f2_root: Path, config: Mapping[str, Any]) -> Path:
    resolution = config.get("experiment_resolution", {})
    image_size = resolution.get("input_image_size") if isinstance(resolution, Mapping) else None
    if isinstance(image_size, (int, float)):
        size = int(image_size)
    else:
        size = 1920 if "1920" in c2f2_root.name else 960
    return expand(
        Path.home()
        / "sofa_mesh"
        / "sofa50_refinement"
        / f"multiview_{size}"
        / "gt_query_manifest.json"
    )


def manifest_records(manifest: Path, split: str) -> list[dict[str, Any]]:
    payload = read_json(manifest)
    records = payload.get("samples")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest has no samples: {manifest}")
    selected = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if split != "all" and record.get("split") != split:
            continue
        selected.append(dict(record))
    if not selected:
        raise ValueError(f"No samples for split={split!r} in {manifest}")
    return selected


def resolve_source_sample_path(manifest: Path, record: Mapping[str, Any]) -> Path:
    value = Path(str(record["path"]))
    return value.resolve() if value.is_absolute() else (manifest.parent / value).resolve()


def query_mesh_path(coarse_models_root: Path, sample_id: str, mesh_name: str) -> Path:
    return coarse_models_root / sample_id / mesh_name


def _copy_source_for_current_graph(
    *,
    source: Mapping[str, Any],
    source_dataset_root: Path,
    current_mesh,
    visibility: np.ndarray,
    sample_id: str,
    coarse_path: Path,
    build_uniform_laplacian_data,
    apply_uniform_laplacian,
    faces_to_edge_index,
    incident_edge_length_and_valid_mask,
    normalize_laplacian_by_edge_scale,
    expected_views: int = 14,
    zero_initial_laplacian: bool = True,
) -> dict[str, Any]:
    sample = {
        key: (value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value))
        for key, value in source.items()
        if key
        not in {
            "_dataset_root",
            "_static_prepared",
            "edge_index",
            "vertex_degree",
            "query_positions",
            "query_offsets",
            "query_is_exact",
            "target_positions",
        }
    }

    if "image_paths" not in source:
        raise ValueError(
            f"Source {sample_id} is not lazy-image storage. Expected original 14-view image_paths."
        )
    sample["image_paths"] = [
        str((Path(value) if Path(value).is_absolute() else source_dataset_root / value).resolve())
        for value in source["image_paths"]
    ]
    if len(sample["image_paths"]) != expected_views:
        raise ValueError(
            f"{sample_id}: expected exactly {expected_views} prediction views, got {len(sample['image_paths'])}"
        )

    current_mesh.ensure_normals()
    vertices = torch.as_tensor(current_mesh.vertices, dtype=torch.float32)
    faces = torch.as_tensor(current_mesh.faces, dtype=torch.long)
    normals = torch.as_tensor(current_mesh.normals, dtype=torch.float32)
    edge_index = faces_to_edge_index(faces)
    local_h, valid = incident_edge_length_and_valid_mask(vertices, edge_index)
    lap_data = build_uniform_laplacian_data(current_mesh.faces, current_mesh.num_vertices)
    placeholder_raw = torch.as_tensor(
        apply_uniform_laplacian(current_mesh.vertices, lap_data), dtype=torch.float32
    )
    placeholder_normalized = normalize_laplacian_by_edge_scale(
        placeholder_raw,
        local_h,
        valid_scale_mask=valid,
    )
    visibility_t = torch.as_tensor(visibility, dtype=torch.bool)
    if tuple(visibility_t.shape) != (expected_views, current_mesh.num_vertices):
        raise ValueError(
            f"{sample_id}: visibility shape={tuple(visibility_t.shape)}, "
            f"expected={(expected_views, current_mesh.num_vertices)}"
        )

    sample.update(
        {
            "sample_id": sample_id,
            "vertices": vertices,
            "faces": faces,
            "vertex_normals": normals,
            # Canonical inference does not feed the current raw Laplacian as a
            # target/input correction.  Query-training evaluation uses zeros.
            "initial_laplacian": (
                torch.zeros_like(vertices) if zero_initial_laplacian else placeholder_raw
            ),
            # Schema-only identity placeholder.  It is never evaluated as a GT
            # differential target in this script.
            "laplacian_target": placeholder_normalized,
            "raw_laplacian_target": placeholder_raw,
            "normalized_laplacian_target": placeholder_normalized,
            "target_confidence": torch.ones(current_mesh.num_vertices, dtype=torch.float32),
            "local_edge_length": local_h,
            "local_edge_scale": local_h.square(),
            "valid_scale_mask": valid,
            "visibility": visibility_t,
            "visibility_backface_and_occlusion": visibility_t,
            "visibility_backface_only": None,
            "visibility_occlusion_only": None,
        }
    )

    # The GT mesh is retained only for final geometry metrics.  There is no
    # vertexwise current-to-GT target_positions field because topologies differ.
    if "gt_vertices" not in sample or "gt_faces" not in sample:
        raise ValueError(f"{sample_id}: source sample does not contain GT geometry for evaluation")

    metadata = dict(sample.get("metadata", {}))
    metadata.update(
        {
            "query_geometry_role": "openmvs48_coarse_current_graph",
            "coarse_mesh_path": str(coarse_path),
            "coarse_acquisition": "48_auxiliary_textured_views_openmvs",
            "prediction_image_source": f"nested_{expected_views}_view_rgb",
            "prediction_view_count": expected_views,
            "target_semantics": "identity_placeholder",
            "expanded_graph_oracle_available": False,
            "gt_differential_transfer_used": False,
            "renderer_visibility": {
                "definition": "depth_tested_face_id_incident_face_neighborhood",
                "mesh_identity": "computed_from_openmvs48_current_vertices_and_faces",
                "depth_image_used": False,
            },
        }
    )
    sample["metadata"] = metadata
    return sample


def compute_visibility(
    *,
    mesh,
    source: Mapping[str, Any],
    cache_path: Path,
    backend: str,
    visibility_size: int | None,
    Camera,
    SyntheticRenderConfig,
    compute_renderer_visibility,
    expected_views: int = 14,
) -> tuple[np.ndarray, dict[str, Any]]:
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            visibility = np.asarray(archive["visibility"], dtype=bool)
            counts = visibility.sum(axis=0)
        return visibility, {
            "cached": True,
            "zero_view_ratio": float(np.mean(counts == 0)),
            "one_two_view_ratio": float(np.mean((counts >= 1) & (counts <= 2))),
            "three_plus_view_ratio": float(np.mean(counts >= 3)),
            "mean_visible_views_per_vertex": float(counts.mean()),
        }

    image_size = int(source["prepared_image_size"])
    raster_size = image_size if visibility_size is None else int(visibility_size)
    if raster_size < 64:
        raise ValueError("visibility-size must be >= 64")
    intrinsics = source["intrinsics"].detach().cpu().numpy().copy()
    extrinsics = source["extrinsics"].detach().cpu().numpy()
    if intrinsics.shape[0] != expected_views:
        raise ValueError(f"Expected {expected_views} cameras, got {intrinsics.shape[0]}")
    if raster_size != image_size:
        scale = float(raster_size) / float(image_size)
        intrinsics[:, 0, :] *= scale
        intrinsics[:, 1, :] *= scale
    cameras = [
        Camera(
            intrinsics=intrinsics[index],
            rotation=extrinsics[index, :3, :3],
            translation=extrinsics[index, :3, 3],
            image_size=(raster_size, raster_size),
            name=f"prediction_{expected_views}_{index:02d}",
        )
        for index in range(expected_views)
    ]
    result = compute_renderer_visibility(
        mesh,
        cameras,
        SyntheticRenderConfig(
            num_views=expected_views,
            width=raster_size,
            height=raster_size,
            backend=backend,
            normalize_mesh=False,
            antialiasing="none",
            backface_culling=False,
            front_face_winding="ccw",
        ),
        neighborhood_radius=1,
    )
    visibility = np.asarray(result.backface_and_occlusion_visible, dtype=bool)
    counts = visibility.sum(axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        visibility=visibility,
        visibility_count=counts,
        source_image_size=np.asarray(image_size),
        raster_size=np.asarray(raster_size),
    )
    return visibility, {
        "cached": False,
        "source_image_size": image_size,
        "raster_size": raster_size,
        "zero_view_ratio": float(np.mean(counts == 0)),
        "one_two_view_ratio": float(np.mean((counts >= 1) & (counts <= 2))),
        "three_plus_view_ratio": float(np.mean(counts >= 3)),
        "mean_visible_views_per_vertex": float(counts.mean()),
    }


def topology_change(initial: np.ndarray, recovered: np.ndarray, faces: np.ndarray) -> dict[str, int]:
    before = np.cross(
        initial[faces[:, 1]] - initial[faces[:, 0]],
        initial[faces[:, 2]] - initial[faces[:, 0]],
    )
    after = np.cross(
        recovered[faces[:, 1]] - recovered[faces[:, 0]],
        recovered[faces[:, 2]] - recovered[faces[:, 0]],
    )
    before_degenerate = np.linalg.norm(before, axis=1) <= 1e-14
    after_degenerate = np.linalg.norm(after, axis=1) <= 1e-14
    return {
        "introduced_flips": int(np.sum(np.einsum("ij,ij->i", before, after) < 0)),
        "new_degeneracies": int(np.sum(after_degenerate & ~before_degenerate)),
    }


def recovery_row(
    *,
    sample_id: str,
    variant: str,
    seed: int | None,
    checkpoint: str | None,
    metrics: Mapping[str, Any],
    initial_vertices: np.ndarray,
    faces: np.ndarray,
    recovered_vertices: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    prediction: np.ndarray,
    output_mesh: Path,
) -> dict[str, Any]:
    coarse = metrics["geometry"]["coarse"]
    refined = metrics["geometry"]["predicted"]
    displacement = np.linalg.norm(recovered_vertices - initial_vertices, axis=1)
    counts = visibility.sum(axis=0)
    visible = counts > 0
    topology = topology_change(initial_vertices, recovered_vertices, faces)
    return {
        "sample_id": sample_id,
        "variant": variant,
        "seed": seed,
        "checkpoint": checkpoint,
        "initial_vertices": int(len(initial_vertices)),
        "initial_faces": int(len(faces)),
        "initial_chamfer": coarse.get("chamfer"),
        "refined_chamfer": refined.get("chamfer"),
        "chamfer_improvement": _difference(coarse.get("chamfer"), refined.get("chamfer")),
        "chamfer_ratio_to_initial": _ratio(refined.get("chamfer"), coarse.get("chamfer")),
        "better_than_initial_chamfer": _less(refined.get("chamfer"), coarse.get("chamfer")),
        "initial_point_to_surface": coarse.get("point_to_surface_bidirectional_mean"),
        "refined_point_to_surface": refined.get("point_to_surface_bidirectional_mean"),
        "initial_normal_consistency": coarse.get("normal_consistency"),
        "refined_normal_consistency": refined.get("normal_consistency"),
        "normal_change": _difference(refined.get("normal_consistency"), coarse.get("normal_consistency"), reverse=True),
        "mean_vertex_displacement": float(displacement.mean()),
        "max_vertex_displacement": float(displacement.max(initial=0.0)),
        "visible_mean_displacement": _masked_mean(displacement, visible),
        "invisible_mean_displacement": _masked_mean(displacement, ~visible),
        "one_two_view_mean_displacement": _masked_mean(displacement, (counts >= 1) & (counts <= 2)),
        "three_plus_view_mean_displacement": _masked_mean(displacement, counts >= 3),
        "visible_vertices": int(visible.sum()),
        "invisible_vertices": int((~visible).sum()),
        "mean_visible_views_per_vertex": float(counts.mean()),
        "mean_confidence": float(np.mean(confidence)),
        "mean_prediction_magnitude": float(np.linalg.norm(prediction, axis=1).mean()),
        "introduced_flips": topology["introduced_flips"],
        "new_degeneracies": topology["new_degeneracies"],
        "solver": metrics["reconstruction"].get("predicted_solver"),
        "solver_all_finite": metrics["reconstruction"].get("all_finite"),
        "output_mesh": str(output_mesh),
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        if not selected:
            continue
        result.append(
            {
                "variant": variant,
                "mesh_count": len(selected),
                "mean_initial_chamfer": _mean(selected, "initial_chamfer"),
                "mean_refined_chamfer": _mean(selected, "refined_chamfer"),
                "median_refined_chamfer": _median(selected, "refined_chamfer"),
                "mean_chamfer_improvement": _mean(selected, "chamfer_improvement"),
                "mean_chamfer_ratio_to_initial": _mean(selected, "chamfer_ratio_to_initial"),
                "better_than_initial_meshes": int(sum(bool(row.get("better_than_initial_chamfer")) for row in selected)),
                "mean_initial_point_to_surface": _mean(selected, "initial_point_to_surface"),
                "mean_refined_point_to_surface": _mean(selected, "refined_point_to_surface"),
                "mean_initial_normal_consistency": _mean(selected, "initial_normal_consistency"),
                "mean_refined_normal_consistency": _mean(selected, "refined_normal_consistency"),
                "introduced_flips": int(sum(int(row.get("introduced_flips") or 0) for row in selected)),
                "new_degeneracies": int(sum(int(row.get("new_degeneracies") or 0) for row in selected)),
                "mean_vertex_displacement": _mean(selected, "mean_vertex_displacement"),
                "mean_confidence": _mean(selected, "mean_confidence"),
            }
        )
    return result


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if _finite(row.get(key))]
    return float(np.mean(values)) if values else None


def _median(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if _finite(row.get(key))]
    return float(np.median(values)) if values else None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def _difference(a: Any, b: Any, reverse: bool = False) -> float | None:
    if not (_finite(a) and _finite(b)):
        return None
    return float(b) - float(a) if reverse else float(a) - float(b)


def _ratio(a: Any, b: Any) -> float | None:
    if not (_finite(a) and _finite(b)):
        return None
    return float(a) / max(abs(float(b)), 1e-12)


def _less(a: Any, b: Any) -> bool | None:
    if not (_finite(a) and _finite(b)):
        return None
    return bool(float(a) < float(b))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[np.asarray(mask, dtype=bool)]
    return float(selected.mean()) if len(selected) else None


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def prepare_query_samples(
    *,
    records: Sequence[Mapping[str, Any]],
    source_manifest: Path,
    coarse_models_root: Path,
    mesh_name: str,
    output_dir: Path,
    visibility_backend: str,
    visibility_size: int | None,
    require_all: bool,
    modules: Mapping[str, Any],
    expected_views: int = 14,
    zero_initial_laplacian: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    load_prepared_sample = modules["load_prepared_sample"]
    save_prepared_sample = modules["save_prepared_sample"]
    load_mesh = modules["load_mesh"]

    prepared_records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        sample_id = str(record.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"Manifest record lacks sample_id: {record}")
        coarse_path = query_mesh_path(coarse_models_root, sample_id, mesh_name)
        if not coarse_path.is_file():
            missing.append({"sample_id": sample_id, "coarse_mesh": str(coarse_path), "reason": "missing_coarse_mesh"})
            if require_all:
                raise FileNotFoundError(f"Missing OpenMVS query mesh: {coarse_path}")
            print(f"[skip] {sample_id}: coarse mesh not ready: {coarse_path}", flush=True)
            continue
        source_path = resolve_source_sample_path(source_manifest, record)
        source = load_prepared_sample(
            source_path,
            materialize_images=False,
            dataset_root=source_manifest.parent,
        )
        current_mesh = load_mesh(coarse_path).ensure_normals()
        visibility, visibility_diag = compute_visibility(
            mesh=current_mesh,
            source=source,
            cache_path=output_dir / "visibility" / f"{sample_id}.npz",
            backend=visibility_backend,
            visibility_size=visibility_size,
            Camera=modules["Camera"],
            SyntheticRenderConfig=modules["SyntheticRenderConfig"],
            compute_renderer_visibility=modules["compute_renderer_visibility"],
            expected_views=expected_views,
        )
        prepared = _copy_source_for_current_graph(
            source=source,
            source_dataset_root=source_manifest.parent,
            current_mesh=current_mesh,
            visibility=visibility,
            sample_id=sample_id,
            coarse_path=coarse_path,
            build_uniform_laplacian_data=modules["build_uniform_laplacian_data"],
            apply_uniform_laplacian=modules["apply_uniform_laplacian"],
            faces_to_edge_index=modules["faces_to_edge_index"],
            incident_edge_length_and_valid_mask=modules["incident_edge_length_and_valid_mask"],
            normalize_laplacian_by_edge_scale=modules["normalize_laplacian_by_edge_scale"],
            expected_views=expected_views,
            zero_initial_laplacian=zero_initial_laplacian,
        )
        destination = output_dir / "prepared_query" / f"{sample_id}.pt"
        save_prepared_sample(prepared, destination)
        prepared_records.append(
            {
                "sample_id": sample_id,
                "split": record.get("split"),
                "source_sample": str(source_path),
                "coarse_mesh": str(coarse_path),
                "prepared_sample": str(destination),
                "vertices": current_mesh.num_vertices,
                "faces": current_mesh.num_faces,
                "visibility": visibility_diag,
            }
        )
        print(
            f"[{index}/{len(records)}] prepared {sample_id}: "
            f"{current_mesh.num_vertices}v/{current_mesh.num_faces}f, "
            f"zero-visible={visibility_diag['zero_view_ratio']:.3f}",
            flush=True,
        )
    if not prepared_records:
        raise RuntimeError("No OpenMVS coarse meshes were available for evaluation.")
    write_json(output_dir / "prepared_query_summary.json", {"prepared": prepared_records, "missing": missing})
    return prepared_records, missing


def load_runtime_modules(repo_root: Path) -> dict[str, Any]:
    src = repo_root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"Repository src directory not found: {src}")
    sys.path.insert(0, str(src))

    from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
    from mlr.data import Camera
    from mlr.io import load_mesh
    from mlr.synthetic import SyntheticRenderConfig
    from mlr.learned_laplacian.canonical_pipeline import canonical_current_graph_recovery_inputs
    from mlr.learned_laplacian.dataset import load_prepared_sample, save_prepared_sample
    from mlr.learned_laplacian.diagnostics import _amp_settings
    from mlr.learned_laplacian.evaluation import reconstruct_and_evaluate
    from mlr.learned_laplacian.graph_layers import faces_to_edge_index
    from mlr.learned_laplacian.multi_trainer import _build_model, _prepare_item_for_use, _prepare_object_static
    from mlr.learned_laplacian.renderer_visibility import compute_renderer_visibility
    from mlr.learned_laplacian.target_scaling import incident_edge_length_and_valid_mask, normalize_laplacian_by_edge_scale
    from mlr.learned_laplacian.trainer import load_checkpoint

    return {
        "apply_uniform_laplacian": apply_uniform_laplacian,
        "build_uniform_laplacian_data": build_uniform_laplacian_data,
        "Camera": Camera,
        "load_mesh": load_mesh,
        "SyntheticRenderConfig": SyntheticRenderConfig,
        "canonical_current_graph_recovery_inputs": canonical_current_graph_recovery_inputs,
        "load_prepared_sample": load_prepared_sample,
        "save_prepared_sample": save_prepared_sample,
        "_amp_settings": _amp_settings,
        "reconstruct_and_evaluate": reconstruct_and_evaluate,
        "faces_to_edge_index": faces_to_edge_index,
        "_build_model": _build_model,
        "_prepare_item_for_use": _prepare_item_for_use,
        "_prepare_object_static": _prepare_object_static,
        "compute_renderer_visibility": compute_renderer_visibility,
        "incident_edge_length_and_valid_mask": incident_edge_length_and_valid_mask,
        "normalize_laplacian_by_edge_scale": normalize_laplacian_by_edge_scale,
        "load_checkpoint": load_checkpoint,
    }


def load_static_for_inference(prepared_path: Path, modules: Mapping[str, Any]) -> dict[str, Any]:
    return modules["load_prepared_sample"](
        prepared_path,
        materialize_images=False,
        dataset_root=prepared_path.parent.parent,
    )


def infer_seed(
    *,
    seed: int,
    seed_dir: Path,
    prepared_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    device: torch.device,
    modules: Mapping[str, Any],
    expected_views: int = 14,
) -> list[dict[str, Any]]:
    config = read_json(config_path(seed_dir))
    query_config = copy.deepcopy(config)
    query_config.setdefault("query_training", {})["enabled"] = False
    query_config["query_training"]["zero_initial_laplacian"] = True
    query_config.setdefault("local_query_jitter", {})["enabled"] = False
    checkpoint = checkpoint_path(seed_dir)
    model = modules["_build_model"](query_config, None, False).to(device)
    modules["load_checkpoint"](checkpoint, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = modules["_amp_settings"](query_config, device)
    epsilon = float(query_config.get("target_scaling", {}).get("epsilon", 1e-12))
    recovery_config = dict(query_config.get("recovery", {}))
    unseen_anchor_weight = float(recovery_config.get("unseen_anchor_weight", 0.0))
    recovery_config.update(
        {
            "dense_vertex_limit": int(recovery_config.get("dense_vertex_limit", 5000)),
            "chamfer_samples": int(recovery_config.get("chamfer_samples", 3000)),
            "metric_seed": 7,
            "evaluate_oracle": False,
        }
    )
    rows: list[dict[str, Any]] = []
    print(f"\n=== C2F2 seed {seed}: {checkpoint} ===", flush=True)

    with torch.no_grad():
        for index, record in enumerate(prepared_records, start=1):
            sample_id = str(record["sample_id"])
            static = load_static_for_inference(Path(record["prepared_sample"]), modules)
            visibility = torch.as_tensor(static[VISIBILITY_FIELD], dtype=torch.bool)
            prepared = modules["_prepare_item_for_use"](
                modules["_prepare_object_static"](static, query_config),
                query_config,
                device,
                cache_on_device=False,
                non_blocking=False,
                decode_images=True,
            )
            sample = dict(prepared.sample)
            if int(sample["images"].shape[0]) != expected_views:
                raise ValueError(
                    f"{sample_id}: model input has {sample['images'].shape[0]} views, "
                    f"expected {expected_views}"
                )
            sample["query_positions"] = sample["vertices"]
            sample["query_is_exact"] = torch.ones(
                sample["vertices"].shape[0], dtype=torch.bool, device=device
            )
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(sample)
            delta_hat = getattr(output, "delta_hat_prediction", None)
            if delta_hat is None:
                delta_hat = output.predicted_laplacian
            delta_hat = delta_hat.float().detach().cpu()
            confidence_t = getattr(output, "confidence_prediction", None)
            if confidence_t is None:
                raise RuntimeError(f"C2F2 seed {seed} has no confidence prediction head")
            confidence = confidence_t.float().detach().cpu()
            if not torch.isfinite(delta_hat).all() or not torch.isfinite(confidence).all():
                raise FloatingPointError(f"Non-finite prediction for seed={seed} sample={sample_id}")

            recovery_inputs = modules["canonical_current_graph_recovery_inputs"](
                static["vertices"],
                static["faces"],
                delta_hat,
                visibility,
                confidence,
                epsilon=epsilon,
            )
            prediction_dir = output_dir / "predictions" / f"seed_{seed}"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = prediction_dir / f"{sample_id}.npz"
            np.savez_compressed(
                prediction_path,
                delta_hat_prediction=recovery_inputs.delta_hat_prediction.numpy(),
                delta_pred_raw=recovery_inputs.delta_pred_raw.numpy(),
                confidence_prediction=recovery_inputs.confidence_prediction.numpy(),
                h_current=recovery_inputs.h_current.numpy(),
                weight=recovery_inputs.weight.numpy(),
                visible=recovery_inputs.visible.numpy(),
                checkpoint=np.asarray(str(checkpoint)),
                checkpoint_sha256=np.asarray(sha256(checkpoint)),
            )

            recover_dir = output_dir / "recovered" / f"seed_{seed}" / sample_id
            metrics = modules["reconstruct_and_evaluate"](
                static,
                recovery_inputs.delta_pred_raw,
                recover_dir,
                recovery_config,
                normalized_prediction=recovery_inputs.delta_hat_prediction,
                edge_scale_epsilon=epsilon,
                laplacian_weight=recovery_inputs.weight,
                unseen_anchor_weight=unseen_anchor_weight,
                evaluate_laplacian_prediction=False,
                evaluate_initial_geometry=True,
                solver_confidence=np.ones(int(static["vertices"].shape[0]), dtype=np.float64),
            )
            recovered = modules["load_mesh"](recover_dir / "predicted_refined.obj")
            initial_vertices = static["vertices"].detach().cpu().numpy()
            faces = static["faces"].detach().cpu().numpy()
            row = recovery_row(
                sample_id=sample_id,
                variant=f"seed_{seed}",
                seed=seed,
                checkpoint=str(checkpoint),
                metrics=metrics,
                initial_vertices=initial_vertices,
                faces=faces,
                recovered_vertices=recovered.vertices,
                visibility=visibility.numpy(),
                confidence=recovery_inputs.confidence_prediction.numpy(),
                prediction=recovery_inputs.delta_hat_prediction.numpy(),
                output_mesh=recover_dir / "predicted_refined.obj",
            )
            rows.append(row)
            print(
                f"[{index}/{len(prepared_records)}] {sample_id}: "
                f"Chamfer {row['initial_chamfer']:.6g} -> {row['refined_chamfer']:.6g} "
                f"better={row['better_than_initial_chamfer']}",
                flush=True,
            )
            del prepared, sample, output, delta_hat, confidence_t, confidence
            if device.type == "cuda":
                torch.cuda.empty_cache()

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def infer_ensemble(
    *,
    first_seed_config: Mapping[str, Any],
    prepared_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    modules: Mapping[str, Any],
) -> list[dict[str, Any]]:
    epsilon = float(first_seed_config.get("target_scaling", {}).get("epsilon", 1e-12))
    recovery_config = dict(first_seed_config.get("recovery", {}))
    unseen_anchor_weight = float(recovery_config.get("unseen_anchor_weight", 0.0))
    recovery_config.update(
        {
            "dense_vertex_limit": int(recovery_config.get("dense_vertex_limit", 5000)),
            "chamfer_samples": int(recovery_config.get("chamfer_samples", 3000)),
            "metric_seed": 7,
            "evaluate_oracle": False,
        }
    )
    rows: list[dict[str, Any]] = []
    print("\n=== 3-seed mean ensemble ===", flush=True)
    for index, record in enumerate(prepared_records, start=1):
        sample_id = str(record["sample_id"])
        static = load_static_for_inference(Path(record["prepared_sample"]), modules)
        visibility = torch.as_tensor(static[VISIBILITY_FIELD], dtype=torch.bool)
        delta_hats = []
        confidences = []
        checkpoints = []
        for seed in SEEDS:
            path = output_dir / "predictions" / f"seed_{seed}" / f"{sample_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing seed prediction required for ensemble: {path}")
            with np.load(path, allow_pickle=False) as archive:
                delta_hats.append(np.asarray(archive["delta_hat_prediction"], dtype=np.float32))
                confidences.append(np.asarray(archive["confidence_prediction"], dtype=np.float32))
                checkpoints.append(str(archive["checkpoint"].item()))
        delta_hat = torch.from_numpy(np.mean(np.stack(delta_hats), axis=0))
        confidence = torch.from_numpy(np.mean(np.stack(confidences), axis=0))
        recovery_inputs = modules["canonical_current_graph_recovery_inputs"](
            static["vertices"],
            static["faces"],
            delta_hat,
            visibility,
            confidence,
            epsilon=epsilon,
        )
        prediction_dir = output_dir / "predictions" / "ensemble_mean"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_dir / f"{sample_id}.npz",
            delta_hat_prediction=recovery_inputs.delta_hat_prediction.numpy(),
            delta_pred_raw=recovery_inputs.delta_pred_raw.numpy(),
            confidence_prediction=recovery_inputs.confidence_prediction.numpy(),
            h_current=recovery_inputs.h_current.numpy(),
            weight=recovery_inputs.weight.numpy(),
            visible=recovery_inputs.visible.numpy(),
            member_seeds=np.asarray(SEEDS),
        )
        recover_dir = output_dir / "recovered" / "ensemble_mean" / sample_id
        metrics = modules["reconstruct_and_evaluate"](
            static,
            recovery_inputs.delta_pred_raw,
            recover_dir,
            recovery_config,
            normalized_prediction=recovery_inputs.delta_hat_prediction,
            edge_scale_epsilon=epsilon,
            laplacian_weight=recovery_inputs.weight,
            unseen_anchor_weight=unseen_anchor_weight,
            evaluate_laplacian_prediction=False,
            evaluate_initial_geometry=True,
            solver_confidence=np.ones(int(static["vertices"].shape[0]), dtype=np.float64),
        )
        recovered = modules["load_mesh"](recover_dir / "predicted_refined.obj")
        initial_vertices = static["vertices"].detach().cpu().numpy()
        faces = static["faces"].detach().cpu().numpy()
        row = recovery_row(
            sample_id=sample_id,
            variant="ensemble_mean",
            seed=None,
            checkpoint=";".join(checkpoints),
            metrics=metrics,
            initial_vertices=initial_vertices,
            faces=faces,
            recovered_vertices=recovered.vertices,
            visibility=visibility.numpy(),
            confidence=recovery_inputs.confidence_prediction.numpy(),
            prediction=recovery_inputs.delta_hat_prediction.numpy(),
            output_mesh=recover_dir / "predicted_refined.obj",
        )
        rows.append(row)
        print(
            f"[{index}/{len(prepared_records)}] {sample_id}: "
            f"Chamfer {row['initial_chamfer']:.6g} -> {row['refined_chamfer']:.6g} "
            f"better={row['better_than_initial_chamfer']}",
            flush=True,
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Test 48-view OpenMVS Sofa50 coarse meshes using the original 14-view RGB "
            "with the C2F2 seed-7/17/27 checkpoints and a 3-seed mean ensemble."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("~/multiview-laplacian-refinement"),
    )
    parser.add_argument(
        "--c2f2-root",
        type=Path,
        help=(
            "C2F2 3-seed root containing seed_7/seed_17/seed_27. "
            "Default: auto-detect canonical 50k root, then 1920 root."
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help=(
            "Original 14-view GT-query manifest used only as the RGB/camera/GT source. "
            "Default is inferred as multiview_960 or multiview_1920 from the C2F2 run."
        ),
    )
    parser.add_argument(
        "--coarse-models-root",
        type=Path,
        default=Path(
            "~/sofa_mesh/sofa50_refinement/openmvs_texture_test_v6_48view/"
            "reconstruction/models"
        ),
    )
    parser.add_argument(
        "--mesh-name",
        default="coarse.obj",
        help="Query mesh filename inside each OpenMVS model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "~/multiview-laplacian-refinement/runs/learned_laplacian/"
            "sofa50_openmvs48_c2f2_3seed_test"
        ),
    )
    parser.add_argument(
        "--split",
        choices=("test", "validation", "train", "all"),
        default="test",
        help="Default is the held-out 5-mesh test split.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device. C2F2/F2 is normally evaluated on CUDA.",
    )
    parser.add_argument(
        "--visibility-backend",
        choices=("cpu", "opengl"),
        default="cpu",
        help="Renderer visibility backend; CPU avoids the known EGL issue on the current workstation.",
    )
    parser.add_argument(
        "--visibility-size",
        type=int,
        help=(
            "Optional lower raster size for visibility only; intrinsics are scaled consistently. "
            "Omit to use the original prediction image resolution."
        ),
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any selected split model has not finished OpenMVS coarse generation.",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Delete cached prepared-query and visibility files before preparation.",
    )
    args = parser.parse_args()

    repo_root = expand(args.repo_root)
    coarse_models_root = expand(args.coarse_models_root)
    output_dir = expand(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.force_prepare:
        for child in (output_dir / "prepared_query", output_dir / "visibility"):
            if child.exists():
                shutil.rmtree(child)

    modules = load_runtime_modules(repo_root)
    c2f2_root = discover_c2f2_root(repo_root, args.c2f2_root)
    seed_configs: dict[int, dict[str, Any]] = {}
    seed_info: dict[int, dict[str, Any]] = {}
    signatures = []
    for seed in SEEDS:
        seed_dir = c2f2_root / f"seed_{seed}"
        cfg_path = config_path(seed_dir)
        checkpoint = checkpoint_path(seed_dir)
        cfg = read_json(cfg_path)
        seed_configs[seed] = cfg
        signatures.append(model_signature(cfg))
        seed_info[seed] = {
            "seed_dir": str(seed_dir),
            "config": str(cfg_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        }
    if not all(signature == signatures[0] for signature in signatures[1:]):
        raise ValueError("The three C2F2 seed configs do not share the same inference signature.")

    source_manifest = (
        expand(args.source_manifest)
        if args.source_manifest is not None
        else infer_source_manifest(c2f2_root, seed_configs[SEEDS[0]])
    )
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Original 14-view source manifest not found: {source_manifest}")
    if not coarse_models_root.is_dir():
        raise FileNotFoundError(f"OpenMVS 48-view model root not found: {coarse_models_root}")

    records = manifest_records(source_manifest, args.split)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for C2F2 inference but torch.cuda.is_available() is false")

    experiment = {
        "purpose": "openmvs48_coarse_original14_c2f2_3seed_test",
        "repo_root": str(repo_root),
        "c2f2_root": str(c2f2_root),
        "seeds": list(SEEDS),
        "seed_runs": seed_info,
        "source_manifest": str(source_manifest),
        "prediction_views": 14,
        "prediction_images": "original_sofa50_rgb",
        "coarse_models_root": str(coarse_models_root),
        "coarse_acquisition": "48 auxiliary textured views -> COLMAP/OpenMVS",
        "query_mesh_name": args.mesh_name,
        "split": args.split,
        "selected_manifest_records": len(records),
        "target_semantics": "identity_placeholder",
        "gt_differential_transfer_used": False,
        "gt_usage": "geometry_metrics_only",
        "visibility_backend": args.visibility_backend,
        "visibility_size": args.visibility_size,
        "device": str(device),
        "ensemble": "arithmetic mean of delta_hat and confidence across seeds 7/17/27",
        "model_signature": signatures[0],
    }
    write_json(output_dir / "experiment_config.json", experiment)

    prepared_records, missing = prepare_query_samples(
        records=records,
        source_manifest=source_manifest,
        coarse_models_root=coarse_models_root,
        mesh_name=args.mesh_name,
        output_dir=output_dir,
        visibility_backend=args.visibility_backend,
        visibility_size=args.visibility_size,
        require_all=args.require_all,
        modules=modules,
    )

    all_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        all_rows.extend(
            infer_seed(
                seed=seed,
                seed_dir=c2f2_root / f"seed_{seed}",
                prepared_records=prepared_records,
                output_dir=output_dir,
                device=device,
                modules=modules,
            )
        )
    all_rows.extend(
        infer_ensemble(
            first_seed_config=seed_configs[SEEDS[0]],
            prepared_records=prepared_records,
            output_dir=output_dir,
            modules=modules,
        )
    )

    aggregates = aggregate_rows(all_rows)
    write_csv(output_dir / "per_mesh_metrics.csv", all_rows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregates)
    summary = {
        **experiment,
        "evaluated_mesh_count": len(prepared_records),
        "missing_meshes": missing,
        "per_mesh_metrics": all_rows,
        "aggregate_metrics": aggregates,
    }
    write_json(output_dir / "summary.json", summary)

    print("\n=== Aggregate ===", flush=True)
    for row in aggregates:
        print(
            f"{row['variant']:>14}: initial={row['mean_initial_chamfer']:.6g} "
            f"refined={row['mean_refined_chamfer']:.6g} "
            f"better={row['better_than_initial_meshes']}/{row['mesh_count']}",
            flush=True,
        )
    print(f"\nWrote: {output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
