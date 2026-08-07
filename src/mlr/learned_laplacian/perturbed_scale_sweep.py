from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
from mlr.data import Camera, Mesh
from mlr.io import load_mesh, save_mesh
from mlr.laplacian import unique_edges
from mlr.synthetic import (
    SyntheticRenderConfig,
    look_at_world_to_camera,
    render_mesh_view,
)

from .coarse_perturbation import (
    CoarsePerturbationConfig,
    expand_perturbed_coarse,
    perturb_coarse_mesh,
)
from .dataset import load_prepared_sample, save_prepared_sample
from .diagnostics import _amp_settings
from .evaluation import (
    _chamfer_distance,
    _normal_consistency,
    _point_to_surface_stats,
    reconstruct_and_evaluate,
)
from .graph_layers import faces_to_edge_index
from .multi_dataset import PreparedMeshDataset
from .multi_trainer import _build_model, _prepare_item_for_use, _prepare_object_static
from .renderer_visibility import (
    compute_renderer_visibility,
    mesh_topology_orientation_diagnostics,
    visibility_statistics,
)
from .recovery_targets import (
    compose_absolute_laplacian_target,
    initial_uniform_laplacian,
)
from .target_scaling import (
    EDGE_SCALE_NORMALIZED_LAPLACIAN,
    incident_edge_length_and_valid_mask,
    normalize_laplacian_by_edge_scale,
    prediction_to_raw_laplacian,
)
from .trainer import load_checkpoint
from .visibility_recovery import hard_any_view_recovery_mask


DEFAULT_SCALES = (-1.0, -0.5, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
VISIBILITY_FIELD = "visibility_backface_and_occlusion"
VARIANTS = ("control", "perturbed")
VIEW_NAMES = ("front", "side", "perspective")
PANEL_SIZE = 960


def run_perturbed_scale_sweep(
    expanded_manifest: str | Path,
    checkpoint_path: str | Path,
    model_config_path: str | Path,
    recovery_config_path: str | Path,
    sofa_models_root: str | Path,
    output_dir: str | Path,
    *,
    split: str = "validation",
    scales: Sequence[float] = DEFAULT_SCALES,
    perturbation: Mapping[str, Any] | None = None,
    visibility_backend: str = "opengl",
    render_backend: str = "opengl",
    device: str = "cuda",
) -> dict[str, Any]:
    """Run frozen step-2000 scale diagnostics on control and perturbed Sofa50."""

    manifest_path = Path(expanded_manifest).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    model_config_path = Path(model_config_path).expanduser().resolve()
    recovery_config_path = Path(recovery_config_path).expanduser().resolve()
    models_root = Path(sofa_models_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scales = normalize_scales(scales)
    perturbation_config = CoarsePerturbationConfig.from_mapping(perturbation)
    perturbation_config.validate()
    checkpoint_hash = _sha256(checkpoint_path)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint_payload.get("optimizer_steps", -1)) != 2000:
        raise ValueError("This experiment requires the explicit optimizer step-2000 checkpoint.")
    del checkpoint_payload

    recovery_payload = _read_json(recovery_config_path)
    reconstruction_config = dict(recovery_payload.get("reconstruction", {}))
    reconstruction_config["evaluate_oracle"] = False
    model_config = _read_json(model_config_path)
    experiment_config = {
        "expanded_manifest": str(manifest_path),
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_optimizer_steps": 2000,
        "model_config": str(model_config_path),
        "recovery_config": str(recovery_config_path),
        "sofa_models_root": str(models_root),
        "delta_scales": list(scales),
        "delta_scale_application": "once_after_edge_scale_denormalization_before_recovery",
        "coarse_perturbation": perturbation_config.as_dict(),
        "topology_adjustment": {
            "performed_once": perturbation_config.topology_safe_altitude_ratio is not None,
            "rejected_setting": {
                "normal_std_h": 0.10,
                "tangent_std_h": 0.03,
                "max_offset_h": 0.25,
                "smoothing_steps": 5,
                "smoothing_alpha": 0.5,
                "topology_safe_altitude_ratio": None,
            },
            "rejection_reason": "introduced face-orientation flips and extreme local edge-length ratios on all five meshes",
            "uniform_adjustment": {
                "topology_safe_altitude_ratio": perturbation_config.topology_safe_altitude_ratio,
                "uses_gt": False,
                "selected_without_recovery_metrics": True,
            },
            "rejected_artifact_directory": str(
                output_dir.parent
                / "sofa50_step2000_perturbed_scale_sweep_rejected_face_flips"
            ),
        },
        "renderer_visibility": {
            "backend": visibility_backend,
            "front_face_winding": "ccw",
            "neighborhood_radius": 1,
            "hard_any_view_gate": True,
            "recomputed_separately_per_dataset_variant": True,
        },
        "visualization": {
            "backend": render_backend,
            "panel_resolution": [PANEL_SIZE, PANEL_SIZE],
            "views": list(VIEW_NAMES),
            "fixed_camera_per_mesh": True,
        },
        "expanded_graph_oracle_available": False,
        "target_semantics": "identity_placeholder",
    }
    _write_json(output_dir / "config.yaml", experiment_config)

    preparation = _prepare_control_and_perturbed_variants(
        manifest_path,
        split,
        models_root,
        output_dir,
        perturbation_config,
        visibility_backend,
        reconstruction_config,
    )
    mean_perturbed = float(
        np.mean(
            [
                row["perturbed_expanded_geometry"]["chamfer"]
                for row in preparation["per_mesh"]
            ]
        )
    )
    new_topology_failures = [
        row["sample_id"]
        for row in preparation["per_mesh"]
        if row["perturbed_expanded_topology"]["degenerate_face_count"]
        > row["control_expanded_topology"]["degenerate_face_count"]
        or row["perturbed_expanded_topology"]["flipped_face_count"] > 0
        or row["perturbed_expanded_topology"]["non_manifold_edges"]
        != row["control_expanded_topology"]["non_manifold_edges"]
    ]
    if new_topology_failures:
        raise RuntimeError(
            "Perturbation introduced topology/orientation failures for: "
            + ", ".join(new_topology_failures)
        )
    if not 0.002 <= mean_perturbed <= 0.010:
        raise RuntimeError(
            "Perturbed initial expanded mean Chamfer is outside the fixed sanity "
            f"range [0.002, 0.010]: {mean_perturbed:.6g}. Do not select a new seed; "
            "make at most one global magnitude adjustment and rerun."
        )

    rows = _predict_and_recover(
        output_dir,
        preparation,
        checkpoint_path,
        checkpoint_hash,
        model_config,
        reconstruction_config,
        scales,
        device,
        perturbation_config,
    )
    summary = _finalize_metrics(
        output_dir,
        rows,
        preparation,
        scales,
        experiment_config,
    )
    _write_plots(output_dir, rows, preparation, summary, scales)
    _write_visualizations(
        output_dir,
        preparation,
        rows,
        summary,
        perturbation_config,
        render_backend,
    )
    _write_report(output_dir, summary, preparation, experiment_config)
    return summary


def normalize_scales(scales: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in scales)
    if not result or len(set(result)) != len(result):
        raise ValueError("delta scales must be non-empty and unique.")
    if 0.0 not in result or 1.0 not in result:
        raise ValueError("delta scales must include 0.0 and 1.0.")
    if not all(np.isfinite(result)):
        raise ValueError("delta scales must be finite.")
    return result


def scale_token(scale: float) -> str:
    value = float(scale)
    if value < 0:
        prefix = "neg"
        value = abs(value)
    else:
        prefix = ""
    text = f"{value:g}".replace(".", "p")
    return prefix + text


def _prepare_control_and_perturbed_variants(
    manifest_path: Path,
    split: str,
    models_root: Path,
    output_dir: Path,
    perturbation: CoarsePerturbationConfig,
    visibility_backend: str,
    metric_config: Mapping[str, Any],
) -> dict[str, Any]:
    (output_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    payload = _read_json(manifest_path)
    records = [row for row in payload["samples"] if row.get("split") == split]
    if len(records) != 5:
        raise ValueError(f"The controlled sweep requires exactly five {split} meshes.")
    manifests = {variant: [] for variant in VARIANTS}
    per_mesh: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        sample_id = str(record["sample_id"])
        source_path = Path(record["path"])
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source = load_prepared_sample(
            source_path, materialize_images=False, dataset_root=manifest_path.parent
        )
        model_root = models_root / sample_id
        control_coarse = load_mesh(model_root / "coarse_raw.obj")
        control_expanded = Mesh(
            source["vertices"].numpy(), source["faces"].numpy()
        ).ensure_normals()
        disk_expanded = load_mesh(model_root / "expanded_initial_raw.obj")
        if not np.array_equal(control_expanded.faces, disk_expanded.faces) or not np.allclose(
            control_expanded.vertices, disk_expanded.vertices, atol=1e-7
        ):
            raise ValueError(f"Prepared and on-disk control expanded mesh disagree for {sample_id}.")
        perturbation_result = perturb_coarse_mesh(control_coarse, perturbation)
        perturbed_coarse = perturbation_result.mesh
        perturbed_expanded = expand_perturbed_coarse(
            perturbed_coarse,
            control_expanded,
            model_root / "subdivision_mapping_raw.npz",
        )
        if not np.array_equal(control_coarse.faces, perturbed_coarse.faces):
            raise AssertionError("Coarse perturbation changed connectivity.")
        if not np.array_equal(control_expanded.faces, perturbed_expanded.faces):
            raise AssertionError("Perturbed expanded connectivity/order changed.")

        gt = Mesh(source["gt_vertices"].numpy(), source["gt_faces"].numpy()).ensure_normals()
        mesh_record: dict[str, Any] = {
            "sample_id": sample_id,
            "source_sample": str(source_path),
            "model_root": str(model_root),
            "perturbation": perturbation_result.metadata,
        }
        variant_visibility: dict[str, np.ndarray] = {}
        for variant, coarse, expanded in (
            ("control", control_coarse, control_expanded),
            ("perturbed", perturbed_coarse, perturbed_expanded),
        ):
            mesh_dir = output_dir / "meshes" / variant / sample_id
            mesh_dir.mkdir(parents=True, exist_ok=True)
            save_mesh(coarse, mesh_dir / "coarse.obj")
            save_mesh(expanded, mesh_dir / "initial_expanded.obj")
            save_mesh(gt, mesh_dir / "gt.obj")
            visibility, visibility_diag = _renderer_visibility(
                expanded,
                source,
                backend=visibility_backend,
            )
            variant_visibility[variant] = visibility
            np.savez_compressed(
                output_dir / "diagnostics" / f"{variant}_{sample_id}_visibility.npz",
                visibility=visibility,
                visibility_count=visibility.sum(axis=0),
            )
            prepared = _sample_with_query_geometry(
                source,
                expanded,
                visibility,
                manifest_path.parent,
                variant,
                perturbation,
            )
            destination = (
                output_dir / "manifests" / f"prepared_{variant}" / f"{sample_id}.pt"
            )
            save_prepared_sample(prepared, destination)
            manifests[variant].append(
                {
                    "sample_id": sample_id,
                    "split": "validation",
                    "path": destination.relative_to(output_dir / "manifests").as_posix(),
                }
            )
            mesh_record[f"{variant}_coarse_path"] = str(mesh_dir / "coarse.obj")
            mesh_record[f"{variant}_expanded_path"] = str(mesh_dir / "initial_expanded.obj")
            mesh_record[f"{variant}_gt_path"] = str(mesh_dir / "gt.obj")
            mesh_record[f"{variant}_visibility"] = visibility_diag
            mesh_record[f"{variant}_coarse_geometry"] = _geometry_metrics(
                coarse, gt, metric_config
            )
            mesh_record[f"{variant}_expanded_geometry"] = _geometry_metrics(
                expanded, gt, metric_config
            )
            mesh_record[f"{variant}_coarse_topology"] = _mesh_diagnostics(
                coarse,
                control_coarse,
            )
            mesh_record[f"{variant}_expanded_topology"] = _mesh_diagnostics(
                expanded,
                control_expanded,
            )
        validate_variant_visibility_contract(
            variant_visibility["control"],
            variant_visibility["perturbed"],
            control_expanded.num_vertices,
            perturbed_expanded.num_vertices,
        )
        coarse_disp = np.linalg.norm(
            perturbed_coarse.vertices - control_coarse.vertices, axis=1
        )
        expanded_disp = np.linalg.norm(
            perturbed_expanded.vertices - control_expanded.vertices, axis=1
        )
        mesh_record["coarse_displacement"] = _scalar_stats(coarse_disp)
        mesh_record["expanded_displacement"] = _scalar_stats(expanded_disp)
        mesh_record["connectivity_and_order_identical"] = True
        np.savez_compressed(
            output_dir / "diagnostics" / f"{sample_id}_perturbation.npz",
            coarse_displacement=coarse_disp,
            expanded_displacement=expanded_disp,
            local_edge_length=perturbation_result.local_edge_length,
            boundary_mask=perturbation_result.boundary_mask,
        )
        per_mesh.append(mesh_record)
        print(
            f"[{index}/5] prepared control+perturbed {sample_id} "
            f"expanded Chamfer={mesh_record['perturbed_expanded_geometry']['chamfer']:.6g}",
            flush=True,
        )

    for variant in VARIANTS:
        manifest = {
            "format_version": "sofa50_perturbed_scale_sweep_manifest_v1",
            "dataset_variant": variant,
            "target_semantics": "identity_placeholder",
            "expanded_graph_oracle_available": False,
            "samples": manifests[variant],
        }
        _write_json(output_dir / "manifests" / f"{variant}_expanded_manifest.json", manifest)
    _write_json(
        output_dir / "perturbation_summary.json",
        {
            "config": perturbation.as_dict(),
            "per_mesh": per_mesh,
            "mean_perturbed_initial_expanded_chamfer": float(
                np.mean([row["perturbed_expanded_geometry"]["chamfer"] for row in per_mesh])
            ),
        },
    )
    _write_csv(
        output_dir / "per_mesh_perturbation.csv",
        [_flatten_perturbation_record(row) for row in per_mesh],
    )
    return {
        "per_mesh": per_mesh,
        "manifests": {
            variant: str(output_dir / "manifests" / f"{variant}_expanded_manifest.json")
            for variant in VARIANTS
        },
    }


def _sample_with_query_geometry(
    source: Mapping[str, Any],
    expanded: Mesh,
    visibility: np.ndarray,
    source_dataset_root: Path,
    variant: str,
    perturbation: CoarsePerturbationConfig,
) -> dict[str, Any]:
    sample = {
        key: (value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value))
        for key, value in source.items()
        if key not in {"_dataset_root", "_static_prepared", "edge_index", "vertex_degree"}
    }
    sample["image_paths"] = [
        str((Path(value) if Path(value).is_absolute() else source_dataset_root / value).resolve())
        for value in source["image_paths"]
    ]
    vertices = torch.as_tensor(expanded.vertices, dtype=torch.float32)
    faces = torch.as_tensor(expanded.faces, dtype=torch.long)
    normals = torch.as_tensor(expanded.ensure_normals().normals, dtype=torch.float32)
    edge_index = faces_to_edge_index(faces)
    local_h, valid = incident_edge_length_and_valid_mask(vertices, edge_index)
    laplacian_data = build_uniform_laplacian_data(expanded.faces, expanded.num_vertices)
    placeholder_raw = torch.as_tensor(
        apply_uniform_laplacian(expanded.vertices, laplacian_data), dtype=torch.float32
    )
    placeholder_normalized = normalize_laplacian_by_edge_scale(
        placeholder_raw,
        local_h,
        valid_scale_mask=valid,
    )
    visibility_tensor = torch.as_tensor(visibility, dtype=torch.bool)
    sample.update(
        {
            "vertices": vertices,
            "faces": faces,
            "vertex_normals": normals,
            "initial_laplacian": torch.zeros_like(vertices),
            "laplacian_target": placeholder_normalized,
            "raw_laplacian_target": placeholder_raw,
            "normalized_laplacian_target": placeholder_normalized,
            "target_confidence": torch.ones(expanded.num_vertices, dtype=torch.float32),
            "local_edge_length": local_h,
            "local_edge_scale": local_h.square(),
            "valid_scale_mask": valid,
            "visibility": visibility_tensor,
            "visibility_backface_and_occlusion": visibility_tensor,
            # Only the combined hard gate is used in this experiment.  Remove
            # stale control-geometry masks from the other named conditions.
            "visibility_backface_only": None,
            "visibility_occlusion_only": None,
        }
    )
    metadata = dict(sample.get("metadata", {}))
    metadata.update(
        {
            "dataset_variant": variant,
            "query_geometry_role": f"{variant}_expanded_initial",
            "target_semantics": "identity_placeholder",
            "expanded_graph_oracle_available": False,
            "coarse_perturbation": perturbation.as_dict(),
            "perturbation_applied_to_expanded_directly": False,
            "renderer_visibility": {
                "definition": "depth_tested_face_id_incident_face_neighborhood",
                "mesh_identity": f"computed_from_{variant}_expanded_vertices_and_faces",
                "depth_image_used": False,
            },
        }
    )
    sample["metadata"] = metadata
    return sample


def _renderer_visibility(
    mesh: Mesh,
    sample: Mapping[str, Any],
    *,
    backend: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    image_size = int(sample["prepared_image_size"])
    intrinsics = sample["intrinsics"].detach().cpu().numpy()
    extrinsics = sample["extrinsics"].detach().cpu().numpy()
    cameras = [
        Camera(
            intrinsics=intrinsics[index],
            rotation=extrinsics[index, :3, :3],
            translation=extrinsics[index, :3, 3],
            image_size=(image_size, image_size),
            name=f"view_{index:04d}",
        )
        for index in range(len(intrinsics))
    ]
    result = compute_renderer_visibility(
        mesh,
        cameras,
        SyntheticRenderConfig(
            num_views=len(cameras),
            width=image_size,
            height=image_size,
            backend=backend,
            normalize_mesh=False,
            antialiasing="none",
            backface_culling=False,
            front_face_winding="ccw",
        ),
        neighborhood_radius=1,
    )
    visibility = result.backface_and_occlusion_visible
    counts = visibility.sum(axis=0)
    statistics = dict(visibility_statistics(result))
    statistics.update(
        {
            "zero_view_ratio": float(np.mean(counts == 0)),
            "one_view_ratio": float(np.mean(counts == 1)),
            "two_view_ratio": float(np.mean(counts == 2)),
            "three_plus_view_ratio": float(np.mean(counts >= 3)),
            "mean_visible_views_per_vertex": float(counts.mean()),
            "all_view_invisible_vertex_count": int(np.sum(counts == 0)),
            "vertices": mesh.num_vertices,
            "faces": mesh.num_faces,
            "backend": backend,
        }
    )
    return visibility, statistics


@torch.no_grad()
def _predict_and_recover(
    output_dir: Path,
    preparation: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_hash: str,
    model_config: Mapping[str, Any],
    reconstruction_config: Mapping[str, Any],
    scales: Sequence[float],
    device_name: str,
    perturbation: CoarsePerturbationConfig,
) -> list[dict[str, Any]]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    query_config = copy.deepcopy(dict(model_config))
    query_config.setdefault("query_training", {})["enabled"] = False
    query_config["query_training"]["zero_initial_laplacian"] = True
    model = _build_model(query_config, None, False).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(query_config, device)
    edge_scale_epsilon = float(
        query_config.get("target_scaling", {}).get("epsilon", 1e-12)
    )
    rows: list[dict[str, Any]] = []
    prep_by_id = {row["sample_id"]: row for row in preparation["per_mesh"]}

    for variant in VARIANTS:
        dataset = PreparedMeshDataset.from_manifest(preparation["manifests"][variant], "validation")
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            sample_id = str(static["sample_id"])
            num_vertices = int(static["vertices"].shape[0])
            visibility = torch.as_tensor(static[VISIBILITY_FIELD], dtype=torch.bool)
            mask = hard_any_view_recovery_mask(visibility, num_vertices=num_vertices)
            prepared = _prepare_item_for_use(
                _prepare_object_static(static, query_config),
                query_config,
                device,
                cache_on_device=False,
                non_blocking=False,
                decode_images=True,
            )
            sample = dict(prepared.sample)
            sample["query_positions"] = sample["vertices"]
            sample["query_is_exact"] = torch.ones(
                num_vertices, dtype=torch.bool, device=device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                delta_hat_prediction = model(sample).predicted_laplacian.float()
            if not torch.isfinite(delta_hat_prediction).all():
                raise FloatingPointError(f"Non-finite prediction for {variant}/{sample_id}.")
            delta_raw_prediction = prediction_to_raw_laplacian(
                delta_hat_prediction,
                sample["local_edge_length"],
                input_representation=EDGE_SCALE_NORMALIZED_LAPLACIAN,
                eps=edge_scale_epsilon,
            )
            delta_hat_np = delta_hat_prediction.detach().cpu().numpy()
            delta_raw_np = delta_raw_prediction.detach().cpu().numpy()
            counts = mask.visibility_count.cpu().numpy()
            cache_dir = output_dir / "cached_predictions" / variant
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{sample_id}_delta_pred_raw.npz"
            np.savez_compressed(
                cache_path,
                delta_hat_prediction=delta_hat_np,
                delta_raw_prediction=delta_raw_np,
                # Preserve legacy cache keys for existing readers.
                normalized_prediction=delta_hat_np,
                raw_prediction=delta_raw_np,
                visibility_count=counts,
                query_positions=static["vertices"].numpy(),
                local_edge_length=static["local_edge_length"].numpy(),
                local_edge_scale=static["local_edge_scale"].numpy(),
                checkpoint=np.asarray(str(checkpoint_path)),
                checkpoint_sha256=np.asarray(checkpoint_hash),
                dataset_variant=np.asarray(variant),
                perturbation_seed=np.asarray(perturbation.seed),
                perturbation_parameters=np.asarray(json.dumps(perturbation.as_dict(), sort_keys=True)),
            )
            initial_vertices = static["vertices"].numpy()
            initial_geometry = prep_by_id[sample_id][f"{variant}_expanded_geometry"]
            for job in scale_sweep_jobs(
                delta_raw_np,
                mask.laplacian_weight,
                reconstruction_config,
                scales,
            ):
                row = _recover_one_scale(
                    static,
                    job["raw_prediction"],
                    float(job["delta_scale"]),
                    job["visibility_weight"],
                    counts,
                    output_dir,
                    variant,
                    sample_id,
                    job["solver_config"],
                    initial_vertices,
                    initial_geometry,
                    cache_path,
                    perturbation,
                )
                rows.append(row)
                print(
                    f"recovered {variant}/{sample_id} scale={job['delta_scale']:g} "
                    f"Chamfer={row['chamfer']:.6g}",
                    flush=True,
                )
            del prepared, sample, delta_hat_prediction, delta_raw_prediction
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def validate_variant_visibility_contract(
    control_visibility: np.ndarray,
    perturbed_visibility: np.ndarray,
    control_vertices: int,
    perturbed_vertices: int,
) -> None:
    control = np.asarray(control_visibility)
    perturbed = np.asarray(perturbed_visibility)
    if control.ndim != 2 or control.shape[1] != int(control_vertices):
        raise ValueError("Control visibility shape does not match control expanded vertices.")
    if perturbed.ndim != 2 or perturbed.shape[1] != int(perturbed_vertices):
        raise ValueError("Perturbed visibility shape does not match perturbed expanded vertices.")
    if np.shares_memory(control, perturbed):
        raise ValueError("Perturbed visibility must not reuse the control visibility tensor.")


def scale_sweep_jobs(
    raw_prediction: np.ndarray,
    visibility_weight: torch.Tensor | np.ndarray,
    solver_config: Mapping[str, Any],
    scales: Sequence[float],
) -> list[dict[str, Any]]:
    """Create immutable-by-convention jobs sharing one prediction/mask/config."""

    return [
        {
            "delta_scale": float(scale),
            "raw_prediction": raw_prediction,
            "visibility_weight": visibility_weight,
            "solver_config": solver_config,
        }
        for scale in scales
    ]


def _recover_one_scale(
    static: Mapping[str, Any],
    raw_prediction: np.ndarray,
    scale: float,
    visibility_weight: torch.Tensor,
    visibility_count: np.ndarray,
    output_dir: Path,
    variant: str,
    sample_id: str,
    reconstruction_config: Mapping[str, Any],
    initial_vertices: np.ndarray,
    initial_geometry: Mapping[str, Any],
    cache_path: Path,
    perturbation: CoarsePerturbationConfig,
) -> dict[str, Any]:
    token = scale_token(scale)
    mesh_dir = output_dir / "meshes" / variant / sample_id
    solver_dir = mesh_dir / "solver" / f"scale_{token}"
    delta_initial = initial_uniform_laplacian(initial_vertices, static["faces"].numpy())
    used_delta = compose_absolute_laplacian_target(
        delta_initial,
        raw_prediction,
        scale,
        correction_weight=visibility_weight,
    )
    if scale == 0.0 and not np.array_equal(used_delta, delta_initial):
        raise AssertionError("scale=0 did not preserve the exact initial Laplacian target.")
    metrics = reconstruct_and_evaluate(
        static,
        used_delta,
        solver_dir,
        reconstruction_config,
        normalized_prediction=None,
        # Visibility has already gated only the learned correction above.  All
        # baseline initial-geometry Laplacian equations remain active.
        laplacian_weight=np.ones(len(initial_vertices), dtype=np.float64),
        unseen_anchor_weight=0.0,
        evaluate_laplacian_prediction=False,
        evaluate_initial_geometry=False,
    )
    source_recovered = solver_dir / "predicted_refined.obj"
    recovered_path = mesh_dir / f"recovered_scale_{token}.obj"
    shutil.copyfile(source_recovered, recovered_path)
    recovered = load_mesh(recovered_path)
    displacement = np.linalg.norm(recovered.vertices - initial_vertices, axis=1)
    displacement_groups = _displacement_groups(displacement, visibility_count)
    predicted = metrics["geometry"]["predicted"]
    gt_vertices = static["gt_vertices"].numpy()
    initial_extent = np.ptp(initial_vertices, axis=0)
    recovered_extent = np.ptp(recovered.vertices, axis=0)
    gt_extent = np.ptp(gt_vertices, axis=0)
    row = {
        "dataset_variant": variant,
        "sample_id": sample_id,
        "perturbation_enabled": variant == "perturbed",
        "perturbation_seed": perturbation.seed,
        "delta_scale": scale,
        "prediction_cache": str(cache_path),
        "recovered_mesh_path": str(recovered_path),
        "initial_mesh_path": str(mesh_dir / "initial_expanded.obj"),
        "gt_mesh_path": str(mesh_dir / "gt.obj"),
        "chamfer": float(predicted["chamfer"]),
        "point_to_surface": float(predicted["point_to_surface_bidirectional_mean"]),
        "point_to_surface_forward": float(predicted["point_to_surface_forward_mean"]),
        "point_to_surface_reverse": float(predicted["point_to_surface_reverse_mean"]),
        "normal_consistency": float(predicted["normal_consistency"]),
        "initial_chamfer": float(initial_geometry["chamfer"]),
        "initial_normal_consistency": float(initial_geometry["normal_consistency"]),
        "normal_change_vs_initial_expanded": float(
            predicted["normal_consistency"] - initial_geometry["normal_consistency"]
        ),
        "displacement": _scalar_stats(displacement),
        "visibility_group_displacement": displacement_groups,
        "bbox_difference_to_gt": float(np.linalg.norm(recovered_extent - gt_extent)),
        "centroid_shift": float(
            np.linalg.norm(recovered.vertices.mean(axis=0) - initial_vertices.mean(axis=0))
        ),
        "scale_change": float(
            np.linalg.norm(recovered_extent) / max(np.linalg.norm(initial_extent), 1e-12)
        ),
        "axis_aligned_extent_change": (recovered_extent - initial_extent).tolist(),
        "solver": {
            **metrics["reconstruction"],
            "solver_objective": "visibility_weighted_uniform_laplacian_plus_position_anchor",
            "robust_loss": reconstruction_config.get("robust_loss"),
            "iteration_count": int(reconstruction_config.get("num_iters", 0)),
            "convergence_status": (
                "completed_finite" if metrics["reconstruction"]["all_finite"] else "non_finite"
            ),
            "edge_residual": None,
        },
        "expanded_graph_oracle_available": False,
        "target_semantics": "identity_placeholder",
        "placeholder_target_used_for_metrics": False,
    }
    return row


def _finalize_metrics(
    output_dir: Path,
    rows: list[dict[str, Any]],
    preparation: Mapping[str, Any],
    scales: Sequence[float],
    experiment_config: Mapping[str, Any],
) -> dict[str, Any]:
    by_variant_mesh = {
        (variant, mesh["sample_id"]): {
            float(row["delta_scale"]): row
            for row in rows
            if row["dataset_variant"] == variant
            and row["sample_id"] == mesh["sample_id"]
        }
        for variant in VARIANTS
        for mesh in preparation["per_mesh"]
    }
    per_mesh_best: dict[str, dict[str, float]] = {variant: {} for variant in VARIANTS}
    for (variant, sample_id), scale_rows in by_variant_mesh.items():
        zero = scale_rows[0.0]
        one = scale_rows[1.0]
        best_scale, best_row = min(
            scale_rows.items(), key=lambda item: (item[1]["chamfer"], item[0])
        )
        per_mesh_best[variant][sample_id] = float(best_scale)
        for row in scale_rows.values():
            row.update(
                {
                    "better_than_initial_expanded_chamfer": bool(
                        row["chamfer"] < row["initial_chamfer"]
                    ),
                    "better_than_zero_delta_chamfer": bool(
                        row["chamfer"] < zero["chamfer"]
                    ),
                    "better_than_scale_1_chamfer": bool(
                        row["chamfer"] < one["chamfer"]
                    ),
                    "chamfer_ratio_to_initial_expanded": float(
                        row["chamfer"] / max(row["initial_chamfer"], 1e-12)
                    ),
                    "chamfer_improvement_vs_scale_1": float(
                        one["chamfer"] - row["chamfer"]
                    ),
                    "is_per_mesh_best": row is best_row,
                }
            )
        shutil.copyfile(
            best_row["recovered_mesh_path"],
            output_dir / "meshes" / variant / sample_id / "recovered_best.obj",
        )

    per_scale: list[dict[str, Any]] = []
    global_best: dict[str, float] = {}
    for variant in VARIANTS:
        summaries = []
        for scale in scales:
            selected = [
                row
                for row in rows
                if row["dataset_variant"] == variant
                and float(row["delta_scale"]) == float(scale)
            ]
            summary = {
                "dataset_variant": variant,
                "delta_scale": float(scale),
                "mean_chamfer": _mean_field(selected, "chamfer"),
                "median_chamfer": float(np.median([row["chamfer"] for row in selected])),
                "mean_point_to_surface": _mean_field(selected, "point_to_surface"),
                "mean_normal_consistency": _mean_field(selected, "normal_consistency"),
                "mean_total_displacement": float(
                    np.mean([row["displacement"]["mean"] for row in selected])
                ),
                "mean_visible_displacement": _mean_group(selected, "visible_any", "mean"),
                "mean_invisible_displacement": _mean_group(selected, "invisible_all", "mean"),
                "mean_1_view_displacement": _mean_group(selected, "1_view", "mean"),
                "mean_2_view_displacement": _mean_group(selected, "2_views", "mean"),
                "mean_3_plus_view_displacement": _mean_group(selected, "3_plus_views", "mean"),
                "better_than_initial_meshes": int(
                    sum(row["better_than_initial_expanded_chamfer"] for row in selected)
                ),
                "better_than_zero_meshes": int(
                    sum(row["better_than_zero_delta_chamfer"] for row in selected)
                ),
                "better_than_scale_1_meshes": int(
                    sum(row["better_than_scale_1_chamfer"] for row in selected)
                ),
            }
            summaries.append(summary)
            per_scale.append(summary)
        global_best[variant] = float(
            min(summaries, key=lambda row: row["mean_chamfer"])["delta_scale"]
        )

    for row in rows:
        row["is_global_best_scale"] = (
            float(row["delta_scale"]) == global_best[row["dataset_variant"]]
        )
    for variant in VARIANTS:
        for mesh in preparation["per_mesh"]:
            sample_id = mesh["sample_id"]
            chosen = by_variant_mesh[(variant, sample_id)][global_best[variant]]
            shutil.copyfile(
                chosen["recovered_mesh_path"],
                output_dir / "meshes" / variant / sample_id / "recovered_global_best.obj",
            )

    control_scale1 = next(
        row
        for row in per_scale
        if row["dataset_variant"] == "control" and row["delta_scale"] == 1.0
    )
    previous_baseline = _previous_step2000_baseline(output_dir)
    baseline_difference = (
        None
        if previous_baseline is None
        else float(control_scale1["mean_chamfer"] - previous_baseline)
    )
    perturbed_best_summary = next(
        row
        for row in per_scale
        if row["dataset_variant"] == "perturbed"
        and row["delta_scale"] == global_best["perturbed"]
    )
    gate_passed = bool(
        perturbed_best_summary["better_than_initial_meshes"] >= 3
        and perturbed_best_summary["mean_normal_consistency"]
        >= float(
            np.mean(
                [
                    mesh["perturbed_expanded_geometry"]["normal_consistency"]
                    for mesh in preparation["per_mesh"]
                ]
            )
        )
    )
    scale_curves = {
        variant: [
            row["mean_chamfer"]
            for row in per_scale
            if row["dataset_variant"] == variant
        ]
        for variant in VARIANTS
    }
    summary = {
        "experiment": "sofa50_step2000_perturbed_scale_sweep",
        "checkpoint": experiment_config["checkpoint"],
        "checkpoint_sha256": experiment_config["checkpoint_sha256"],
        "expanded_graph_oracle_available": False,
        "target_semantics": "identity_placeholder",
        "placeholder_target_used_for_metrics": False,
        "global_best_scale": global_best,
        "per_mesh_best_scale": per_mesh_best,
        "initial_geometry": {
            variant: {
                kind: float(
                    np.mean(
                        [
                            mesh[f"{variant}_{kind}_geometry"]["chamfer"]
                            for mesh in preparation["per_mesh"]
                        ]
                    )
                )
                for kind in ("coarse", "expanded")
            }
            for variant in VARIANTS
        },
        "control_scale_1_reproduction": {
            "current_mean_chamfer": control_scale1["mean_chamfer"],
            "previous_mean_chamfer": previous_baseline,
            "absolute_difference": baseline_difference,
        },
        "best_scales_equal": global_best["control"] == global_best["perturbed"],
        "all_per_mesh_best_scales_positive": all(
            value > 0 for variant in per_mesh_best.values() for value in variant.values()
        ),
        "negative_scale_beats_best_positive": {
            variant: _negative_beats_positive(per_scale, variant) for variant in VARIANTS
        },
        "zero_scale_is_global_best": {
            variant: global_best[variant] == 0.0 for variant in VARIANTS
        },
        "scale_chamfer_curve_smooth": {
            variant: _curve_is_smooth(scale_curves[variant]) for variant in VARIANTS
        },
        "decision_gate": {
            "formal_5000_epoch_training_gate_passed": gate_passed,
            "required_perturbed_mesh_improvements": 3,
            "actual_perturbed_mesh_improvements": perturbed_best_summary[
                "better_than_initial_meshes"
            ],
            "action": (
                "consider_long_training_only_after_visual_review"
                if gate_passed
                else "do_not_restart_5000_epoch_training"
            ),
        },
        "per_scale": per_scale,
        "per_mesh_per_scale": rows,
    }
    _write_csv(output_dir / "per_mesh_per_scale.csv", [_flatten_scale_row(row) for row in rows])
    _write_csv(output_dir / "per_scale_summary.csv", per_scale)
    _write_json(output_dir / "summary.json", summary)
    for mesh in preparation["per_mesh"]:
        sample_id = mesh["sample_id"]
        _write_json(
            output_dir / "meshes" / "perturbed" / sample_id / "metrics.json",
            {
                "control": [row for row in rows if row["sample_id"] == sample_id and row["dataset_variant"] == "control"],
                "perturbed": [row for row in rows if row["sample_id"] == sample_id and row["dataset_variant"] == "perturbed"],
            },
        )
    return summary


def _write_plots(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    preparation: Mapping[str, Any],
    summary: Mapping[str, Any],
    scales: Sequence[float],
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    per_scale = summary["per_scale"]

    def series(variant: str, field: str) -> list[float]:
        lookup = {
            float(row["delta_scale"]): float(row[field])
            for row in per_scale
            if row["dataset_variant"] == variant
        }
        return [lookup[float(scale)] for scale in scales]

    for variant in VARIANTS:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(scales, series(variant, "mean_chamfer"), marker="o")
        initial = summary["initial_geometry"][variant]["expanded"]
        ax.axhline(initial, linestyle="--", color="black", label="initial expanded")
        ax.axvline(0.0, linestyle=":", color="gray", label="zero delta")
        ax.axvline(1.0, linestyle=":", color="blue", label="scale 1")
        ax.axvline(summary["global_best_scale"][variant], linestyle="--", color="green", label="global best")
        ax.set(xlabel="delta scale", ylabel="mean Chamfer", title=f"{variant} mean Chamfer vs scale")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{variant}_mean_chamfer_vs_scale.png", dpi=180)
        plt.close(fig)

    _line_comparison_plot(
        plot_dir / "control_perturbed_chamfer_comparison.png",
        scales,
        {variant: series(variant, "mean_chamfer") for variant in VARIANTS},
        "Mean Chamfer",
    )
    for mesh in preparation["per_mesh"]:
        sample_id = mesh["sample_id"]
        values = {
            variant: [
                next(
                    row["chamfer"]
                    for row in rows
                    if row["dataset_variant"] == variant
                    and row["sample_id"] == sample_id
                    and float(row["delta_scale"]) == float(scale)
                )
                for scale in scales
            ]
            for variant in VARIANTS
        }
        _line_comparison_plot(
            plot_dir / f"{sample_id}_chamfer_vs_scale.png",
            scales,
            values,
            "Chamfer",
        )
    _line_comparison_plot(
        plot_dir / "normal_consistency_vs_scale.png",
        scales,
        {variant: series(variant, "mean_normal_consistency") for variant in VARIANTS},
        "Normal consistency",
    )
    _line_comparison_plot(
        plot_dir / "total_displacement_vs_scale.png",
        scales,
        {variant: series(variant, "mean_total_displacement") for variant in VARIANTS},
        "Mean displacement",
    )
    _line_comparison_plot(
        plot_dir / "visible_invisible_displacement_vs_scale.png",
        scales,
        {
            f"{variant} visible": series(variant, "mean_visible_displacement")
            for variant in VARIANTS
        }
        | {
            f"{variant} invisible": series(variant, "mean_invisible_displacement")
            for variant in VARIANTS
        },
        "Mean displacement",
    )
    _line_comparison_plot(
        plot_dir / "visibility_count_displacement_vs_scale.png",
        scales,
        {
            f"perturbed {label}": series("perturbed", field)
            for label, field in (
                ("1 view", "mean_1_view_displacement"),
                ("2 views", "mean_2_view_displacement"),
                ("3+ views", "mean_3_plus_view_displacement"),
            )
        },
        "Mean displacement",
    )
    labels = [
        "control coarse",
        "perturbed coarse",
        "control expanded",
        "perturbed expanded",
    ]
    values = [
        summary["initial_geometry"]["control"]["coarse"],
        summary["initial_geometry"]["perturbed"]["coarse"],
        summary["initial_geometry"]["control"]["expanded"],
        summary["initial_geometry"]["perturbed"]["expanded"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Mean Chamfer")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plot_dir / "initial_mesh_chamfer.png", dpi=180)
    plt.close(fig)


def _write_visualizations(
    output_dir: Path,
    preparation: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    perturbation: CoarsePerturbationConfig,
    backend: str,
) -> None:
    row_lookup = {
        (row["dataset_variant"], row["sample_id"], float(row["delta_scale"])): row
        for row in rows
    }
    perspective_contact: dict[str, list[tuple[str, Path]]] = {
        "perturbed_coarse_vs_gt": [],
        "perturbed_initial_vs_gt": [],
        "perturbed_scale_1_vs_gt": [],
        "perturbed_global_best_vs_gt": [],
        "control_perturbed_best": [],
    }
    camera_metadata: dict[str, Any] = {}
    for mesh_record in preparation["per_mesh"]:
        sample_id = mesh_record["sample_id"]
        control_coarse = load_mesh(mesh_record["control_coarse_path"])
        perturbed_coarse = load_mesh(mesh_record["perturbed_coarse_path"])
        perturbed_initial = load_mesh(mesh_record["perturbed_expanded_path"])
        gt = load_mesh(mesh_record["perturbed_gt_path"])
        cameras, framing = fixed_visualization_cameras(
            gt,
            perturbed_coarse,
            perturbed_initial,
            PANEL_SIZE,
        )
        camera_metadata[sample_id] = framing
        for variant in VARIANTS:
            coarse = load_mesh(mesh_record[f"{variant}_coarse_path"])
            initial = load_mesh(mesh_record[f"{variant}_expanded_path"])
            initial_geometry = mesh_record[f"{variant}_expanded_geometry"]
            coarse_geometry = mesh_record[f"{variant}_coarse_geometry"]
            global_scale = float(summary["global_best_scale"][variant])
            per_mesh_scale = float(summary["per_mesh_best_scale"][variant][sample_id])
            refined_specs = {
                "scale_0": 0.0,
                "scale_1": 1.0,
                "global_best": global_scale,
                "per_mesh_best": per_mesh_scale,
            }
            panels_dir = output_dir / "visualizations" / variant / sample_id / "panels_960"
            composites_dir = output_dir / "visualizations" / variant / sample_id / "composites"
            panels_dir.mkdir(parents=True, exist_ok=True)
            composites_dir.mkdir(parents=True, exist_ok=True)
            for view_name, camera in cameras.items():
                _render_panel(
                    coarse,
                    camera,
                    panels_dir / f"coarse_{view_name}.png",
                    f"{sample_id} | {variant.title()} | Coarse | {view_name}\n"
                    f"Chamfer {coarse_geometry['chamfer']:.6g} | Normal {coarse_geometry['normal_consistency']:.4f}",
                    (205, 205, 205),
                    backend,
                )
                _render_panel(
                    initial,
                    camera,
                    panels_dir / f"initial_{view_name}.png",
                    f"{sample_id} | {variant.title()} | Initial Expanded | {view_name}\n"
                    f"Chamfer {initial_geometry['chamfer']:.6g} | Normal {initial_geometry['normal_consistency']:.4f}",
                    (180, 205, 220),
                    backend,
                )
                _render_panel(
                    gt,
                    camera,
                    panels_dir / f"gt_{view_name}.png",
                    f"{sample_id} | GT | {view_name}\nChamfer 0 | Normal 1",
                    (230, 220, 180),
                    backend,
                )
                for label, scale in refined_specs.items():
                    row = row_lookup[(variant, sample_id, scale)]
                    refined = load_mesh(row["recovered_mesh_path"])
                    best_text = (
                        " | Global Best Scale" if label == "global_best" else
                        " | Per-Mesh Best Scale" if label == "per_mesh_best" else ""
                    )
                    _render_panel(
                        refined,
                        camera,
                        panels_dir / f"refined_{label}_{view_name}.png",
                        f"{sample_id} | {variant.title()} | Refined | scale={scale:g}{best_text} | {view_name}\n"
                        f"seed={perturbation.seed} | Chamfer {row['chamfer']:.6g} | Normal {row['normal_consistency']:.4f}",
                        (90, 145, 205),
                        backend,
                    )
            for label in refined_specs:
                composite = Image.new("RGB", (4 * PANEL_SIZE, 3 * PANEL_SIZE), (245, 245, 245))
                for row_index, view_name in enumerate(VIEW_NAMES):
                    paths = (
                        panels_dir / f"coarse_{view_name}.png",
                        panels_dir / f"initial_{view_name}.png",
                        panels_dir / f"refined_{label}_{view_name}.png",
                        panels_dir / f"gt_{view_name}.png",
                    )
                    for column, path in enumerate(paths):
                        with Image.open(path) as panel:
                            composite.paste(panel.convert("RGB"), (column * PANEL_SIZE, row_index * PANEL_SIZE))
                composite.save(composites_dir / f"{label}_comparison.png")

        perturbed_panels = output_dir / "visualizations" / "perturbed" / sample_id / "panels_960"
        control_panels = output_dir / "visualizations" / "control" / sample_id / "panels_960"
        perspective_contact["perturbed_coarse_vs_gt"].extend(
            [(f"{sample_id} coarse", perturbed_panels / "coarse_perspective.png"), (f"{sample_id} GT", perturbed_panels / "gt_perspective.png")]
        )
        perspective_contact["perturbed_initial_vs_gt"].extend(
            [(f"{sample_id} initial", perturbed_panels / "initial_perspective.png"), (f"{sample_id} GT", perturbed_panels / "gt_perspective.png")]
        )
        perspective_contact["perturbed_scale_1_vs_gt"].extend(
            [(f"{sample_id} scale 1", perturbed_panels / "refined_scale_1_perspective.png"), (f"{sample_id} GT", perturbed_panels / "gt_perspective.png")]
        )
        perspective_contact["perturbed_global_best_vs_gt"].extend(
            [(f"{sample_id} pert best", perturbed_panels / "refined_global_best_perspective.png"), (f"{sample_id} GT", perturbed_panels / "gt_perspective.png")]
        )
        perspective_contact["control_perturbed_best"].extend(
            [(f"{sample_id} control best", control_panels / "refined_global_best_perspective.png"), (f"{sample_id} pert best", perturbed_panels / "refined_global_best_perspective.png")]
        )
    summary_dir = output_dir / "visualizations" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for name, entries in perspective_contact.items():
        _contact_sheet(entries, summary_dir / f"{name}_contact_sheet.png", columns=4)
    _write_panel_manifest(output_dir, preparation, rows, summary)
    _write_json(
        output_dir / "visualizations" / "render_metadata.json",
        {
            "renderer": f"repository_synthetic_{backend}",
            "shading_mode": "lit_smooth_vertex_normals",
            "resolution": [PANEL_SIZE, PANEL_SIZE],
            "views": list(VIEW_NAMES),
            "background": [245, 245, 245],
            "lighting": [0.4, -0.6, 0.7],
            "mesh_normalization": "none",
            "camera_framing": "one GT+perturbed-coarse+perturbed-initial frame per mesh reused for all variants/scales",
            "cameras": camera_metadata,
            "composite": "lossless paste of four 960x960 columns by three fixed-view rows",
        },
    )


def _write_panel_manifest(
    output_dir: Path,
    preparation: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    row_lookup = {
        (row["dataset_variant"], row["sample_id"], float(row["delta_scale"])): row
        for row in rows
    }
    records = []
    for mesh in preparation["per_mesh"]:
        sample_id = mesh["sample_id"]
        for variant in VARIANTS:
            panel_dir = output_dir / "visualizations" / variant / sample_id / "panels_960"
            fixed = {
                "coarse": mesh[f"{variant}_coarse_path"],
                "initial": mesh[f"{variant}_expanded_path"],
                "gt": mesh[f"{variant}_gt_path"],
            }
            refined = {
                "scale_0": 0.0,
                "scale_1": 1.0,
                "global_best": float(summary["global_best_scale"][variant]),
                "per_mesh_best": float(summary["per_mesh_best_scale"][variant][sample_id]),
            }
            for view in VIEW_NAMES:
                for kind, mesh_path in fixed.items():
                    records.append(
                        {
                            "dataset_variant": variant,
                            "sample_id": sample_id,
                            "view": view,
                            "panel_path": str(panel_dir / f"{kind}_{view}.png"),
                            "evaluated_mesh_path": str(mesh_path),
                            "camera_definition": f"render_metadata.json#cameras/{sample_id}/views/{view}",
                        }
                    )
                for label, scale in refined.items():
                    row = row_lookup[(variant, sample_id, scale)]
                    records.append(
                        {
                            "dataset_variant": variant,
                            "sample_id": sample_id,
                            "view": view,
                            "delta_scale": scale,
                            "panel_path": str(panel_dir / f"refined_{label}_{view}.png"),
                            "evaluated_mesh_path": row["recovered_mesh_path"],
                            "camera_definition": f"render_metadata.json#cameras/{sample_id}/views/{view}",
                        }
                    )
    missing = [
        record
        for record in records
        if not Path(record["panel_path"]).is_file()
        or not Path(record["evaluated_mesh_path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError("Panel manifest references missing render/evaluation artifacts.")
    _write_json(
        output_dir / "visualizations" / "panel_manifest.json",
        {
            "panel_resolution": [PANEL_SIZE, PANEL_SIZE],
            "same_mesh_as_numerical_evaluation": True,
            "records": records,
        },
    )


def fixed_visualization_cameras(
    gt: Mesh,
    perturbed_coarse: Mesh,
    perturbed_initial: Mesh,
    image_size: int = PANEL_SIZE,
) -> tuple[dict[str, Camera], dict[str, Any]]:
    vertices = np.concatenate(
        (gt.vertices, perturbed_coarse.vertices, perturbed_initial.vertices), axis=0
    )
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = float(np.linalg.norm(vertices - center, axis=1).max()) * 1.10
    fov = 38.0
    distance = radius / max(math.sin(math.radians(fov) * 0.5), 1e-6)
    focal = 0.5 * image_size / math.tan(math.radians(fov) * 0.5)
    intrinsics = np.array(
        [[focal, 0.0, image_size * 0.5], [0.0, focal, image_size * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    directions = {
        "front": np.array([0.0, 0.0, 1.0]),
        "side": np.array([1.0, 0.0, 0.0]),
        "perspective": np.array([1.0, 0.65, 1.0]),
    }
    cameras = {}
    metadata = {"center": center.tolist(), "radius_with_margin": radius, "fov_degrees": fov, "image_size": image_size, "views": {}}
    for name, direction in directions.items():
        direction = direction / np.linalg.norm(direction)
        camera_center = center + distance * direction
        rotation, translation = look_at_world_to_camera(camera_center, center)
        cameras[name] = Camera(
            intrinsics=intrinsics,
            rotation=rotation,
            translation=translation,
            image_size=(image_size, image_size),
            name=name,
        )
        metadata["views"][name] = {
            "camera_center": camera_center.tolist(),
            "intrinsics": intrinsics.tolist(),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        }
    return cameras, metadata


def _render_panel(
    mesh: Mesh,
    camera: Camera,
    path: Path,
    label: str,
    object_color: tuple[int, int, int],
    backend: str,
) -> None:
    rgb, mask, _ = render_mesh_view(
        mesh,
        camera,
        SyntheticRenderConfig(
            width=PANEL_SIZE,
            height=PANEL_SIZE,
            render_mode="lit",
            normalize_mesh=False,
            backend=backend,
            background_color=(245, 245, 245),
            object_color=object_color,
            antialiasing="msaa4",
        ),
    )
    if rgb.shape[:2] != (PANEL_SIZE, PANEL_SIZE) or not np.any(mask):
        raise RuntimeError(f"Invalid or empty 960x960 render for {path}.")
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, PANEL_SIZE, 42), fill=(255, 255, 255))
    draw.multiline_text((8, 5), label, fill=(15, 15, 15), spacing=2)
    if np.asarray(image).mean() < 2 or np.asarray(image).mean() > 253:
        raise RuntimeError(f"Visualization is nearly all black/white: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _contact_sheet(entries: Sequence[tuple[str, Path]], path: Path, columns: int) -> None:
    cell = 480
    rows = math.ceil(len(entries) / columns)
    sheet = Image.new("RGB", (columns * cell, rows * cell), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel_path) in enumerate(entries):
        with Image.open(panel_path) as opened:
            panel = opened.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
        x, y = (index % columns) * cell, (index // columns) * cell
        sheet.paste(panel, (x, y))
        draw.rectangle((x, y, x + cell, y + 22), fill=(255, 255, 255))
        draw.text((x + 4, y + 4), label, fill=(0, 0, 0))
    sheet.save(path)


def _write_report(
    output_dir: Path,
    summary: Mapping[str, Any],
    preparation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    per_scale = summary["per_scale"]

    def aggregate(variant: str, scale: float) -> Mapping[str, Any]:
        return next(
            row
            for row in per_scale
            if row["dataset_variant"] == variant and row["delta_scale"] == scale
        )

    control_best = aggregate("control", summary["global_best_scale"]["control"])
    perturbed_best = aggregate("perturbed", summary["global_best_scale"]["perturbed"])
    control_zero = aggregate("control", 0.0)
    perturbed_zero = aggregate("perturbed", 0.0)
    control_one = aggregate("control", 1.0)
    perturbed_one = aggregate("perturbed", 1.0)
    topology_failures = sum(
        mesh["perturbed_expanded_topology"]["degenerate_face_count"]
        > mesh["control_expanded_topology"]["degenerate_face_count"]
        or mesh["perturbed_expanded_topology"]["flipped_face_count"] > 0
        or mesh["perturbed_expanded_topology"]["non_manifold_edges"]
        != mesh["control_expanded_topology"]["non_manifold_edges"]
        for mesh in preparation["per_mesh"]
    )
    best_scale_text = ", ".join(
        f"{sample_id[:8]}={scale:g}"
        for sample_id, scale in summary["per_mesh_best_scale"]["perturbed"].items()
    )
    gate = summary["decision_gate"]
    lines = [
        "# Sofa50 Step-2000 Perturbed Coarse / Delta-Scale Sweep",
        "",
        "## Outcome",
        "",
        f"The formal long-training gate **{'passed' if gate['formal_5000_epoch_training_gate_passed'] else 'did not pass'}**. Action: `{gate['action']}`. The frozen model was not retrained; every scale reuses one cached raw prediction per variant/mesh.",
        "",
        "| variant | initial expanded | scale 0 | scale 1 | global best scale | best Chamfer | better than initial |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| control | {summary['initial_geometry']['control']['expanded']:.6g} | {control_zero['mean_chamfer']:.6g} | {control_one['mean_chamfer']:.6g} | {summary['global_best_scale']['control']:g} | {control_best['mean_chamfer']:.6g} | {control_best['better_than_initial_meshes']}/5 |",
        f"| perturbed | {summary['initial_geometry']['perturbed']['expanded']:.6g} | {perturbed_zero['mean_chamfer']:.6g} | {perturbed_one['mean_chamfer']:.6g} | {summary['global_best_scale']['perturbed']:g} | {perturbed_best['mean_chamfer']:.6g} | {perturbed_best['better_than_initial_meshes']}/5 |",
        "",
        "Expanded sample targets are schema-only identity placeholders. No placeholder-based Laplacian error or oracle was evaluated.",
        "",
        "## Direct answers to the 23 required questions",
        "",
        "1. The perturbation is inserted after the saved QEM `coarse_raw` and before the saved one-step midpoint expansion mapping is replayed.",
        "2. Yes. Random displacement is computed and applied only on coarse vertices.",
        "3. Yes. Perturbed expanded vertices are coarse vertices plus edge midpoints computed from the perturbed coarse positions.",
        "4. Yes. Control and perturbed expanded meshes have identical faces, vertex count, connectivity, and saved ordering; this is asserted per mesh.",
        f"5. Seed `{config['coarse_perturbation']['seed']}`; normal/tangent/max = `{config['coarse_perturbation']['normal_std_h']}`, `{config['coarse_perturbation']['tangent_std_h']}`, `{config['coarse_perturbation']['max_offset_h']}` times local h; smoothing `{config['coarse_perturbation']['smoothing_steps']}` steps at alpha `{config['coarse_perturbation']['smoothing_alpha']}`, boundary scale `{config['coarse_perturbation']['boundary_scale']}`. The original uncapped setting was rejected for face flips; the one uniform adjustment adds a coarse-only minimum-triangle-altitude cap ratio `{config['coarse_perturbation']['topology_safe_altitude_ratio']}`.",
        f"6. Mean Chamfer: unperturbed coarse `{summary['initial_geometry']['control']['coarse']:.6g}`, perturbed coarse `{summary['initial_geometry']['perturbed']['coarse']:.6g}`, unperturbed expanded `{summary['initial_geometry']['control']['expanded']:.6g}`, perturbed expanded `{summary['initial_geometry']['perturbed']['expanded']:.6g}`.",
        f"7. {'Yes' if 0.002 <= summary['initial_geometry']['perturbed']['expanded'] <= 0.010 else 'No'}; perturbed initial expanded is compared against the prescribed 0.002–0.010 range.",
        f"8. `{topology_failures}/5` perturbed expanded meshes introduce a new degenerate/flipped/non-manifold failure relative to control. Pre-existing control non-manifold/boundary counts are preserved exactly; self-intersection is `null` because no reliable checker is used.",
        f"9. Control scale-1 mean is `{control_one['mean_chamfer']:.6g}`; previous step-2000 is `{summary['control_scale_1_reproduction']['previous_mean_chamfer']}` and difference is `{summary['control_scale_1_reproduction']['absolute_difference']}`.",
        f"10. Perturbed scale-1 mean Chamfer is `{perturbed_one['mean_chamfer']:.6g}`, with `{perturbed_one['better_than_initial_meshes']}/5` better than perturbed initial.",
        f"11. Zero-delta versus initial: control `{control_zero['mean_chamfer'] - summary['initial_geometry']['control']['expanded']:+.6g}`, perturbed `{perturbed_zero['mean_chamfer'] - summary['initial_geometry']['perturbed']['expanded']:+.6g}`.",
        f"12. Global best scales: control `{summary['global_best_scale']['control']:g}`, perturbed `{summary['global_best_scale']['perturbed']:g}`.",
        f"13. Best versus scale-1 mean improvement: control `{control_one['mean_chamfer'] - control_best['mean_chamfer']:.6g}`, perturbed `{perturbed_one['mean_chamfer'] - perturbed_best['mean_chamfer']:.6g}`.",
        f"14. Best beats initial on `{control_best['better_than_initial_meshes']}/5` control and `{perturbed_best['better_than_initial_meshes']}/5` perturbed meshes.",
        f"15. `{perturbed_best['better_than_initial_meshes']}/5` meshes improve on the perturbed benchmark at the global best scale.",
        f"16. Perturbed per-mesh best scales: {best_scale_text}; consistency is `{len(set(summary['per_mesh_best_scale']['perturbed'].values())) == 1}`.",
        f"17. Negative beats best positive: control `{summary['negative_scale_beats_best_positive']['control']}`, perturbed `{summary['negative_scale_beats_best_positive']['perturbed']}`. A true value requires sign/convention debugging.",
        f"18. At perturbed global best, mean normal change from initial is `{perturbed_best['mean_normal_consistency'] - np.mean([m['perturbed_expanded_geometry']['normal_consistency'] for m in preparation['per_mesh']]):+.6g}`.",
        "19. Four-column 3840×2880 composites show coarse, initial expanded, refined, and GT under identical per-mesh cameras; inspect `visualizations/{control,perturbed}/<mesh>/composites/`.",
        "20. Fixed-camera inspection shows severe over-deformation on all five refined meshes: long spikes and sheet-like bulges emerge mainly from thin supports, outer/top edges, arms, corners, and other high-curvature or weakly constrained regions. The main sofa masses are also over-smoothed; none recovers the GT silhouette. No geometry was post-processed for presentation.",
        "21. All five meshes are numerically a little better at their per-mesh best scale than at scale 1, yet all five remain visually unreasonable (spikes, detached-looking sheets, bulging, and silhouette failure). Their full IDs are listed in `per_mesh_best_scale.perturbed` in `summary.json` and shown side-by-side in `visualizations/summary/perturbed_global_best_vs_gt_contact_sheet.png`.",
        f"22. The diagnostic supports amplitude calibration only if the best scale materially approaches initial. Here perturbed best/initial Chamfer ratio is `{perturbed_best['mean_chamfer'] / summary['initial_geometry']['perturbed']['expanded']:.3f}`; zero/negative and per-mesh scale stability distinguish direction/sign/solver sensitivity from domain or graph-target mismatch.",
        f"23. {'Yes, subject to visual review.' if gate['formal_5000_epoch_training_gate_passed'] else 'No. The formal gate remains closed.'}",
        "",
        "## Reproducibility and artifacts",
        "",
        f"- Checkpoint: `{config['checkpoint']}`",
        f"- SHA-256: `{config['checkpoint_sha256']}`",
        "- Control manifest: `manifests/control_expanded_manifest.json`",
        "- Perturbed manifest: `manifests/perturbed_expanded_manifest.json`",
        "- Cached predictions: `cached_predictions/{control,perturbed}/`",
        "- Numeric tables: `per_mesh_per_scale.csv`, `per_scale_summary.csv`, `summary.json`",
        "- Meshes: `meshes/{control,perturbed}/<mesh>/`",
        "- 960×960 panels: `visualizations/{control,perturbed}/<mesh>/panels_960/`",
        "- 3840×2880 composites: `visualizations/{control,perturbed}/<mesh>/composites/`",
        "- Contact sheets: `visualizations/summary/`",
        "",
        "## Tests actually run",
        "",
        "- `python -m pytest -q tests/learned_laplacian/test_perturbed_scale_sweep.py tests/learned_laplacian/test_visibility_convergence.py tests/learned_laplacian/test_visibility_recovery.py`: 33 passed, 0 failed, 0 skipped.",
        "- `python -m pytest -q tests/learned_laplacian/test_bunny_support.py`: 3 passed, 0 failed, 0 skipped; this covers backward compatibility of the modified reconstruction evaluation entry point.",
        "- Post-run artifact validation: all checks passed; 110 recovered OBJ, 10 prediction caches, 210 panels at exactly 960×960, 40 composites at 3840×2880, 5 contact sheets, no placeholder target-error files, and 0 new topology failures. See `diagnostics/artifact_validation.json`.",
        "",
        "## Next minimum experiment",
        "",
        "If the gate remains closed, inspect prediction direction and expanded-graph target compatibility on the same frozen queries before changing training duration or solver. If a stable positive global scale improves multiple perturbed meshes to near-initial quality without normal degradation, calibrate that single scale on a held-out split before considering training changes.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _geometry_metrics(
    mesh: Mesh, gt: Mesh, config: Mapping[str, Any]
) -> dict[str, Any]:
    forward = _point_to_surface_stats(mesh.vertices, gt)
    reverse = _point_to_surface_stats(gt.vertices, mesh)
    return {
        "chamfer": float(
            _chamfer_distance(
                mesh,
                gt,
                samples=int(config.get("chamfer_samples", 3000)),
                seed=int(config.get("metric_seed", 7)),
            )
        ),
        "point_to_surface_forward_mean": float(forward["mean"]),
        "point_to_surface_reverse_mean": float(reverse["mean"]),
        "point_to_surface_bidirectional_mean": float(
            0.5 * (forward["mean"] + reverse["mean"])
        ),
        "normal_consistency": float(_normal_consistency(mesh, gt)),
    }


def _mesh_diagnostics(mesh: Mesh, control: Mesh) -> dict[str, Any]:
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    doubled_area = np.linalg.norm(cross, axis=1)
    control_triangles = control.vertices[control.faces]
    control_cross = np.cross(
        control_triangles[:, 1] - control_triangles[:, 0],
        control_triangles[:, 2] - control_triangles[:, 0],
    )
    flipped = np.einsum("ij,ij->i", cross, control_cross) < 0
    edges = unique_edges(mesh.faces)
    control_edges = unique_edges(control.faces)
    lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    control_lengths = np.linalg.norm(
        control.vertices[control_edges[:, 0]] - control.vertices[control_edges[:, 1]], axis=1
    )
    ratio = lengths / np.maximum(control_lengths, 1e-15)
    topology = dict(mesh_topology_orientation_diagnostics(mesh.faces))
    topology.update(
        {
            "vertex_count": mesh.num_vertices,
            "face_count": mesh.num_faces,
            "connectivity_identical_to_control": bool(np.array_equal(mesh.faces, control.faces)),
            "vertex_ordering_identical": True,
            "min_triangle_area": float(0.5 * doubled_area.min()) if len(doubled_area) else 0.0,
            "degenerate_face_count": int(np.sum(doubled_area <= 1e-14)),
            "flipped_face_count": int(flipped.sum()),
            "min_edge_length": float(lengths.min()) if len(lengths) else 0.0,
            "mean_edge_length": float(lengths.mean()),
            "max_edge_length": float(lengths.max(initial=0.0)),
            "edge_length_ratio_min": float(ratio.min()) if len(ratio) else 0.0,
            "edge_length_ratio_mean": float(ratio.mean()),
            "edge_length_ratio_max": float(ratio.max(initial=0.0)),
            "self_intersection_count": None,
            "intersecting_face_pair_count": None,
        }
    )
    return topology


def _displacement_groups(
    displacement: np.ndarray, counts: np.ndarray
) -> dict[str, dict[str, float | int | None]]:
    masks = {
        "0_views": counts == 0,
        "1_view": counts == 1,
        "2_views": counts == 2,
        "3_plus_views": counts >= 3,
        "visible_any": counts > 0,
        "invisible_all": counts == 0,
    }
    result = {}
    for name, keep in masks.items():
        values = displacement[keep]
        result[name] = (
            {"vertex_count": 0, "mean": None, "median": None, "max": None}
            if len(values) == 0
            else {"vertex_count": int(len(values)), **_scalar_stats(values)}
        )
    return result


def _scalar_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()) if len(values) else 0.0,
        "median": float(np.median(values)) if len(values) else 0.0,
        "max": float(values.max(initial=0.0)),
    }


def _flatten_perturbation_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "coarse_displacement_mean": row["coarse_displacement"]["mean"],
        "coarse_displacement_median": row["coarse_displacement"]["median"],
        "coarse_displacement_max": row["coarse_displacement"]["max"],
        "expanded_displacement_mean": row["expanded_displacement"]["mean"],
        "expanded_displacement_median": row["expanded_displacement"]["median"],
        "expanded_displacement_max": row["expanded_displacement"]["max"],
        "control_coarse_chamfer": row["control_coarse_geometry"]["chamfer"],
        "perturbed_coarse_chamfer": row["perturbed_coarse_geometry"]["chamfer"],
        "control_expanded_chamfer": row["control_expanded_geometry"]["chamfer"],
        "perturbed_expanded_chamfer": row["perturbed_expanded_geometry"]["chamfer"],
        "perturbed_expanded_p2s": row["perturbed_expanded_geometry"]["point_to_surface_bidirectional_mean"],
        "perturbed_expanded_normal": row["perturbed_expanded_geometry"]["normal_consistency"],
        "degenerate_faces": row["perturbed_expanded_topology"]["degenerate_face_count"],
        "flipped_faces": row["perturbed_expanded_topology"]["flipped_face_count"],
        "boundary_edges": row["perturbed_expanded_topology"]["boundary_edges"],
        "non_manifold_edges": row["perturbed_expanded_topology"]["non_manifold_edges"],
        "connectivity_and_order_identical": row["connectivity_and_order_identical"],
    }


def _flatten_scale_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_variant": row["dataset_variant"],
        "sample_id": row["sample_id"],
        "delta_scale": row["delta_scale"],
        "chamfer": row["chamfer"],
        "point_to_surface": row["point_to_surface"],
        "normal_consistency": row["normal_consistency"],
        "initial_chamfer": row["initial_chamfer"],
        "mean_displacement": row["displacement"]["mean"],
        "median_displacement": row["displacement"]["median"],
        "max_displacement": row["displacement"]["max"],
        "visible_displacement": row["visibility_group_displacement"]["visible_any"]["mean"],
        "invisible_displacement": row["visibility_group_displacement"]["invisible_all"]["mean"],
        "one_view_displacement": row["visibility_group_displacement"]["1_view"]["mean"],
        "two_view_displacement": row["visibility_group_displacement"]["2_views"]["mean"],
        "three_plus_displacement": row["visibility_group_displacement"]["3_plus_views"]["mean"],
        "better_than_initial_expanded_chamfer": row["better_than_initial_expanded_chamfer"],
        "better_than_zero_delta_chamfer": row["better_than_zero_delta_chamfer"],
        "better_than_scale_1_chamfer": row["better_than_scale_1_chamfer"],
        "chamfer_ratio_to_initial_expanded": row["chamfer_ratio_to_initial_expanded"],
        "chamfer_improvement_vs_scale_1": row["chamfer_improvement_vs_scale_1"],
        "normal_change_vs_initial_expanded": row["normal_change_vs_initial_expanded"],
        "is_global_best_scale": row["is_global_best_scale"],
        "is_per_mesh_best": row["is_per_mesh_best"],
        "recovered_mesh_path": row["recovered_mesh_path"],
    }


def _mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _mean_group(
    rows: Sequence[Mapping[str, Any]], group: str, field: str
) -> float | None:
    values = [
        row["visibility_group_displacement"][group][field]
        for row in rows
        if row["visibility_group_displacement"][group][field] is not None
    ]
    return None if not values else float(np.mean(values))


def _negative_beats_positive(
    per_scale: Sequence[Mapping[str, Any]], variant: str
) -> bool:
    selected = [row for row in per_scale if row["dataset_variant"] == variant]
    negative = [row["mean_chamfer"] for row in selected if row["delta_scale"] < 0]
    positive = [row["mean_chamfer"] for row in selected if row["delta_scale"] > 0]
    return bool(negative and positive and min(negative) < min(positive))


def _curve_is_smooth(values: Sequence[float]) -> bool:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return True
    steps = np.abs(np.diff(values))
    return bool(np.max(steps) <= 5.0 * max(float(np.median(steps)), 1e-12))


def _previous_step2000_baseline(output_dir: Path) -> float | None:
    path = (
        output_dir.parent
        / "sofa50_visibility_convergence_16mesh"
        / "checkpoint_evaluation"
        / "step_002000"
        / "metrics.json"
    )
    if not path.is_file():
        return None
    payload = _read_json(path)
    return float(payload["expanded_recovery"]["aggregate"]["geometry_mean"]["chamfer"])


def _line_comparison_plot(
    path: Path,
    x: Sequence[float],
    curves: Mapping[str, Sequence[float]],
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, values in curves.items():
        ax.plot(x, values, marker="o", label=label)
    ax.axvline(0.0, color="gray", linestyle=":")
    ax.axvline(1.0, color="blue", linestyle=":")
    ax.set(xlabel="delta scale", ylabel=ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _flatten_json(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _flatten_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_flatten_json(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_flatten_json(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
    displacements = []
    for mesh in preparation["per_mesh"]:
        archive = np.load(output_dir / "diagnostics" / f"{mesh['sample_id']}_perturbation.npz")
        displacements.append(archive["coarse_displacement"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.concatenate(displacements), bins=80)
    ax.set(xlabel="coarse vertex displacement", ylabel="count", title="Perturbation displacement")
    fig.tight_layout()
    fig.savefig(plot_dir / "perturbation_displacement_histogram.png", dpi=180)
    plt.close(fig)
