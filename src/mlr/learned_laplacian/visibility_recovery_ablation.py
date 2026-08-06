from __future__ import annotations

import copy
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh

from .diagnostics import _amp_settings
from .evaluation import reconstruct_and_evaluate
from .multi_dataset import PreparedMeshDataset
from .multi_trainer import _build_model, _prepare_item_for_use, _prepare_object_static
from .target_scaling import denormalize_laplacian_by_edge_scale
from .trainer import load_checkpoint
from .visibility_recovery import (
    hard_any_view_recovery_mask,
    visibility_coverage_diagnostics,
)


RECOVERY_VARIANTS = ("baseline", "hard_mask", "hard_mask_unseen_anchor")


def run_visibility_recovery_ablation(
    run_dir: str | Path,
    expanded_manifest: str | Path,
    experiment_config: str | Path,
    output_dir: str | Path,
    *,
    split: str = "validation",
    device: str = "cuda",
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    expanded_manifest = Path(expanded_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_experiment = _read_json(Path(experiment_config).resolve())
    model_config = _read_json(run_dir / "config.json")
    seed = int(recovery_experiment.get("seed", 7))
    condition = str(
        recovery_experiment.get(
            "renderer_visibility_condition", "backface_and_occlusion"
        )
    )
    if condition != "backface_and_occlusion":
        raise ValueError(
            "This first-stage recovery ablation requires renderer-native "
            "backface_and_occlusion visibility."
        )
    visibility_field = "visibility_backface_and_occlusion"
    visibility_settings = recovery_experiment.get("visibility_recovery", {})
    if str(visibility_settings.get("mode", "hard_any_view")) != "hard_any_view":
        raise ValueError("Only visibility_recovery.mode='hard_any_view' is supported.")
    unseen_anchor_weight = float(
        visibility_settings.get("unseen_anchor_weight", 0.0)
    )
    reconstruction_config = dict(recovery_experiment.get("reconstruction", {}))
    reconstruction_config["evaluate_oracle"] = False

    dataset = PreparedMeshDataset.from_manifest(expanded_manifest, split)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _build_model(model_config, None, False).to(resolved_device)
    checkpoint = load_checkpoint(run_dir / "best.pt", model, map_location=resolved_device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(model_config, resolved_device)
    query_config = copy.deepcopy(model_config)
    query_config["renderer_visibility"] = {"condition": condition}
    query_config.setdefault("query_training", {})["enabled"] = False
    query_config["query_training"]["zero_initial_laplacian"] = True

    mesh_records = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        num_vertices = int(static["vertices"].shape[0])
        mask = hard_any_view_recovery_mask(
            static[visibility_field], num_vertices=num_vertices
        )
        coverage = visibility_coverage_diagnostics(mask)
        prepared = _prepare_item_for_use(
            _prepare_object_static(static, query_config),
            query_config,
            resolved_device,
            cache_on_device=False,
            non_blocking=False,
            decode_images=True,
        )
        sample = dict(prepared.sample)
        sample["query_positions"] = sample["vertices"]
        sample["query_is_exact"] = torch.ones(
            num_vertices, dtype=torch.bool, device=resolved_device
        )
        with torch.no_grad(), torch.autocast(
            device_type=resolved_device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            normalized_prediction = model(sample).predicted_laplacian.float()
        if not torch.isfinite(normalized_prediction).all():
            raise FloatingPointError(f"Non-finite prediction for {sample['sample_id']}.")
        raw_prediction = denormalize_laplacian_by_edge_scale(
            normalized_prediction, sample["local_edge_length"]
        )
        sample_id = str(sample["sample_id"])
        mesh_dir = output_dir / sample_id
        mesh_dir.mkdir(parents=True, exist_ok=True)
        initial_vertices = static["vertices"].detach().cpu().numpy()
        faces = static["faces"].detach().cpu().numpy()
        save_mesh(Mesh(initial_vertices, faces), mesh_dir / "initial_expanded.obj")
        _write_visibility_ply(
            mesh_dir / "visibility_count.ply",
            initial_vertices,
            faces,
            mask.visibility_count.cpu().numpy(),
            mask.num_views,
            binary=False,
        )
        _write_visibility_ply(
            mesh_dir / "visibility_any_mask.ply",
            initial_vertices,
            faces,
            mask.visibility_count.cpu().numpy(),
            mask.num_views,
            binary=True,
        )

        variant_records = {}
        variant_vertices = {}
        for variant in RECOVERY_VARIANTS:
            laplacian_weight = None if variant == "baseline" else mask.laplacian_weight
            anchor_weight = (
                unseen_anchor_weight if variant == "hard_mask_unseen_anchor" else 0.0
            )
            variant_dir = mesh_dir / variant
            metrics = reconstruct_and_evaluate(
                static,
                raw_prediction.detach().cpu(),
                variant_dir,
                reconstruction_config,
                normalized_prediction=normalized_prediction.detach().cpu(),
                edge_scale_epsilon=float(
                    model_config.get("target_scaling", {}).get("epsilon", 1e-12)
                ),
                laplacian_weight=laplacian_weight,
                unseen_anchor_weight=anchor_weight,
            )
            recovered = load_mesh(variant_dir / "predicted_refined.obj").vertices
            variant_vertices[variant] = recovered
            displacement = np.linalg.norm(recovered - initial_vertices, axis=1)
            displacement_metrics = _displacement_metrics(
                displacement, mask.visibility_count.cpu().numpy()
            )
            variant_records[variant] = {
                "geometry": metrics["geometry"]["predicted"],
                "improves_over_initial": metrics["predicted_improves_over_coarse"],
                "reconstruction": metrics["reconstruction"],
                "displacement": displacement_metrics,
            }
            shutil.copyfile(
                variant_dir / "predicted_refined.obj",
                mesh_dir
                / {
                    "baseline": "recovered_baseline.obj",
                    "hard_mask": "recovered_visibility_mask.obj",
                    "hard_mask_unseen_anchor": (
                        "recovered_visibility_mask_unseen_anchor.obj"
                    ),
                }[variant],
            )
        np.savez_compressed(
            mesh_dir / "per_vertex_diagnostics.npz",
            visibility_count=mask.visibility_count.cpu().numpy(),
            visible_any=mask.visible_any.cpu().numpy(),
            laplacian_weight=mask.laplacian_weight.cpu().numpy(),
            initial_vertices=initial_vertices,
            baseline_vertices=variant_vertices["baseline"],
            masked_vertices=variant_vertices["hard_mask"],
            masked_anchor_vertices=variant_vertices["hard_mask_unseen_anchor"],
            predicted_delta=normalized_prediction.detach().cpu().numpy(),
        )
        _write_json(mesh_dir / "visibility_diagnostics.json", coverage)
        record = {
            "sample_id": sample_id,
            "visibility": coverage,
            "variants": variant_records,
        }
        _write_json(mesh_dir / "recovery_metrics.json", record)
        mesh_records.append(record)
        print(
            f"{sample_id}: invisible={coverage['invisible_all_ratio']:.3%} "
            f"Chamfer baseline={variant_records['baseline']['geometry']['chamfer']:.6g} "
            f"mask={variant_records['hard_mask']['geometry']['chamfer']:.6g} "
            f"mask+anchor={variant_records['hard_mask_unseen_anchor']['geometry']['chamfer']:.6g}",
            flush=True,
        )
        del prepared, sample, normalized_prediction, raw_prediction
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "checkpoint": str(run_dir / "best.pt"),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "expanded_manifest": str(expanded_manifest),
        "split": split,
        "visibility": {
            "field": visibility_field,
            "stored_shape": "[views, vertices]",
            "dtype": "bool",
            "definition": (
                "frustum AND renderer-native OpenGL back-face culling AND depth-tested "
                "face-ID incident-face visibility in a 3x3 pixel neighborhood"
            ),
            "depth_image_used": False,
        },
        "recovery": {
            "mode": "hard_any_view",
            "equation": "sqrt(weight) * (L @ X - predicted_delta)",
            "unseen_anchor_weight": unseen_anchor_weight,
            "reconstruction_config": reconstruction_config,
        },
        "oracle": {
            "available": False,
            "reason": (
                "Expanded samples contain an identity-placeholder target, not a valid "
                "expanded-graph GT-delta oracle."
            ),
        },
        "aggregate": _aggregate(mesh_records),
        "per_mesh": mesh_records,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "per_mesh.csv", mesh_records)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _displacement_metrics(
    displacement: np.ndarray, visibility_count: np.ndarray
) -> dict[str, Any]:
    groups = {
        "visible": visibility_count > 0,
        "all_view_invisible": visibility_count == 0,
        "low_view_count_1_2": (visibility_count >= 1) & (visibility_count <= 2),
        "well_observed_3_plus": visibility_count >= 3,
    }
    result = {}
    for name, keep in groups.items():
        values = displacement[keep]
        result[name] = {
            "count": int(keep.sum()),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }
    return result


def _aggregate(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "visibility": {
            key: float(np.mean([row["visibility"][key] for row in records]))
            for key in (
                "visible_any_ratio",
                "invisible_all_ratio",
                "mean_visible_view_count",
                "median_visible_view_count",
            )
        }
    }
    for variant in RECOVERY_VARIANTS:
        rows = [row["variants"][variant] for row in records]
        geometry_fields = (
            "chamfer",
            "point_to_surface_forward_mean",
            "point_to_surface_reverse_mean",
            "point_to_surface_bidirectional_mean",
            "normal_consistency",
        )
        result[variant] = {
            "geometry_mean": {
                field: float(np.mean([row["geometry"][field] for row in rows]))
                for field in geometry_fields
            },
            "geometry_median": {
                field: float(np.median([row["geometry"][field] for row in rows]))
                for field in geometry_fields
            },
            "improved_meshes": int(sum(row["improves_over_initial"] for row in rows)),
            "worsened_meshes": int(sum(not row["improves_over_initial"] for row in rows)),
            "collapsed_or_exploded_meshes": int(
                sum(row["geometry"]["collapsed_or_exploded"] for row in rows)
            ),
            "displacement_mean_across_meshes": {
                group: float(
                    np.mean([row["displacement"][group]["mean"] for row in rows])
                )
                for group in (
                    "visible",
                    "all_view_invisible",
                    "low_view_count_1_2",
                    "well_observed_3_plus",
                )
            },
        }
    return result


def _write_visibility_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    counts: np.ndarray,
    num_views: int,
    *,
    binary: bool,
) -> None:
    ratio = counts.astype(np.float64) / max(num_views, 1)
    colors = np.zeros((len(vertices), 3), dtype=np.uint8)
    invisible = counts == 0
    colors[invisible] = (255, 0, 255)
    if binary:
        colors[~invisible] = (0, 200, 80)
    else:
        colors[~invisible, 0] = (255 * (1.0 - ratio[~invisible])).astype(np.uint8)
        colors[~invisible, 1] = (255 * ratio[~invisible]).astype(np.uint8)
        colors[~invisible, 2] = 32
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex, color in zip(vertices, colors, strict=True):
            handle.write(
                f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def _write_csv(path: Path, records: list[Mapping[str, Any]]) -> None:
    fields = [
        "sample_id",
        "invisible_all_ratio",
        "mean_visible_view_count",
        "variant",
        "chamfer",
        "point_to_surface_bidirectional_mean",
        "normal_consistency",
        "visible_displacement_mean",
        "invisible_displacement_mean",
        "low_view_displacement_mean",
        "well_observed_displacement_mean",
        "improves_over_initial",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for variant in RECOVERY_VARIANTS:
                value = record["variants"][variant]
                writer.writerow(
                    {
                        "sample_id": record["sample_id"],
                        "invisible_all_ratio": record["visibility"][
                            "invisible_all_ratio"
                        ],
                        "mean_visible_view_count": record["visibility"][
                            "mean_visible_view_count"
                        ],
                        "variant": variant,
                        "chamfer": value["geometry"]["chamfer"],
                        "point_to_surface_bidirectional_mean": value["geometry"][
                            "point_to_surface_bidirectional_mean"
                        ],
                        "normal_consistency": value["geometry"][
                            "normal_consistency"
                        ],
                        "visible_displacement_mean": value["displacement"]["visible"][
                            "mean"
                        ],
                        "invisible_displacement_mean": value["displacement"][
                            "all_view_invisible"
                        ]["mean"],
                        "low_view_displacement_mean": value["displacement"][
                            "low_view_count_1_2"
                        ]["mean"],
                        "well_observed_displacement_mean": value["displacement"][
                            "well_observed_3_plus"
                        ]["mean"],
                        "improves_over_initial": value["improves_over_initial"],
                    }
                )


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Visibility-aware Laplacian recovery",
        "",
        "All variants use one frozen prediction per mesh. Hard masking applies "
        "`sqrt(W) * (L @ X - delta_pred)`; it does not replace unseen targets with zero.",
        "",
        "| variant | Chamfer mean | bidirectional P2S | normal consistency | visible displacement | invisible displacement | improved | worsened |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in RECOVERY_VARIANTS:
        metrics = summary["aggregate"][variant]
        lines.append(
            f"| {variant} | {metrics['geometry_mean']['chamfer']:.6g} | "
            f"{metrics['geometry_mean']['point_to_surface_bidirectional_mean']:.6g} | "
            f"{metrics['geometry_mean']['normal_consistency']:.4f} | "
            f"{metrics['displacement_mean_across_meshes']['visible']:.6g} | "
            f"{metrics['displacement_mean_across_meshes']['all_view_invisible']:.6g} | "
            f"{metrics['improved_meshes']} | {metrics['worsened_meshes']} |"
        )
    lines.extend(
        (
            "",
            f"Mean all-view-invisible ratio: {summary['aggregate']['visibility']['invisible_all_ratio']:.3%}.",
            "",
            f"Expanded-graph oracle: unavailable — {summary['oracle']['reason']}",
            "",
        )
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(value), indent=2) + "\n", encoding="utf-8")


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value
