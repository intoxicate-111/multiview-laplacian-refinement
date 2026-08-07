from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .graph_layers import faces_to_edge_index
from .multi_dataset import PreparedMeshDataset
from .target_scaling import mean_incident_edge_length


def run_sofa50_transfer_diagnostics(
    canonical_config: str | Path,
    canonical_run_dir: str | Path,
    gt_manifest: str | Path,
    expanded_manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Measure expanded-query gap, mesh normalization, and magnitude error groups."""

    config_path = Path(canonical_config).resolve()
    run = Path(canonical_run_dir).resolve()
    gt_manifest_path = Path(gt_manifest).resolve()
    expanded_manifest_path = Path(expanded_manifest).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = _read_json(config_path)
    query_config = dict(config["query_training"])
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))
    recovery = dict(config["recovery"])
    if float(recovery.get("unseen_anchor_weight", math.nan)) != 0.0:
        raise ValueError("Canonical unseen_anchor_weight must be 0.0.")

    expanded_dataset = PreparedMeshDataset.from_manifest(
        expanded_manifest_path, "validation"
    )
    expanded_rows, ratio_arrays = _expanded_query_gap(
        expanded_dataset,
        output / "per_vertex",
        epsilon=epsilon,
        max_offset_h=float(query_config["max_offset_h"]),
    )
    aggregate_ratio = _ratio_summary(np.concatenate(ratio_arrays))
    aggregate_ratio.update(
        _outside_augmentation_summary(
            np.concatenate(ratio_arrays), float(query_config["max_offset_h"])
        )
    )

    mesh_rows, mesh_summary, render_spec = _mesh_normalization_diagnostics(
        gt_manifest_path
    )
    curvature_rows, curvature_summary = _curvature_error_groups(
        run,
        PreparedMeshDataset.from_manifest(gt_manifest_path, "validation").sample_ids,
        epsilon,
    )
    query_gap_clear = bool(
        aggregate_ratio["fraction_above_training_max_offset_h"] > 0.5
        or aggregate_ratio["median"] > float(query_config["max_offset_h"])
    )
    summary = {
        "diagnostic": "sofa50_gt_query_to_real_expanded_distribution_gap",
        "dataset": "Sofa50 only",
        "canonical_config": str(config_path),
        "canonical_run_dir": str(run),
        "gt_manifest": str(gt_manifest_path),
        "expanded_manifest": str(expanded_manifest_path),
        "canonical_recovery": {
            "laplacian_weight": "renderer_visible_any * confidence_prediction",
            "lambda_anchor": float(recovery["lambda_anchor"]),
            "unseen_anchor_weight": float(recovery["unseen_anchor_weight"]),
        },
        "training_query_augmentation_h_units": {
            "normal_std_h": float(query_config["normal_std_h"]),
            "tangent_std_h": float(query_config["tangent_std_h"]),
            "max_offset_h": float(query_config["max_offset_h"]),
        },
        "expanded_validation_mesh_count": len(expanded_rows),
        "expanded_query_gap_aggregate": aggregate_ratio,
        "expanded_query_gap_per_mesh": expanded_rows,
        "query_gap_clearly_exceeds_training_field": query_gap_clear,
        "render_spec": render_spec,
        "mesh_normalization": mesh_summary,
        "curvature_grouping_policy": {
            "signal": "relative percentile of ||delta_hat_gt||",
            "strict_plane_detector": False,
            "changes_training_or_recovery": False,
            "caveat": (
                "Uniform Laplacian magnitude depends on connectivity, sampling "
                "irregularity, and discretization."
            ),
        },
        "curvature_error_groups": curvature_rows,
        "curvature_error_summary": curvature_summary,
        "conclusion": (
            "Real expanded queries lie clearly outside the trained local query field. "
            "The 50 meshes are consistently centered/scaled for the fixed cameras, "
            "so the evidence supports query-distribution/cross-graph transfer as the "
            "main current bottleneck."
            if query_gap_clear and not mesh_summary["has_anomalies"]
            else "The available diagnostics do not isolate a clean query-distribution gap."
        ),
    }
    shutil.copyfile(config_path, output / "canonical_config.json")
    _write_csv(output / "expanded_query_gap.csv", expanded_rows)
    _write_csv(output / "mesh_camera_normalization.csv", mesh_rows)
    _write_csv(output / "curvature_error_groups.csv", curvature_rows)
    _write_json(output / "summary.json", summary)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _expanded_query_gap(
    dataset: PreparedMeshDataset,
    per_vertex_dir: Path,
    *,
    epsilon: float,
    max_offset_h: float,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    ratios: list[np.ndarray] = []
    per_vertex_dir.mkdir(parents=True, exist_ok=True)
    for index in range(len(dataset)):
        sample = dataset.load_static(index)
        vertices = torch.as_tensor(sample["vertices"])
        faces = torch.as_tensor(sample["faces"], dtype=torch.long)
        h_current = mean_incident_edge_length(
            vertices,
            faces_to_edge_index(faces, int(vertices.shape[0])),
            eps=epsilon,
        ).numpy()
        distances = _nearest_surface_distances(
            vertices.numpy(),
            torch.as_tensor(sample["gt_vertices"]).numpy(),
            torch.as_tensor(sample["gt_faces"], dtype=torch.long).numpy(),
        )
        ratio = distances / np.maximum(h_current, epsilon)
        ratios.append(ratio)
        row = {
            "sample_id": str(sample["sample_id"]),
            "vertex_count": int(len(vertices)),
            **_ratio_summary(ratio),
            **_outside_augmentation_summary(ratio, max_offset_h),
            "nearest_surface_engine": "trimesh_rtree_exact",
        }
        rows.append(row)
        np.savez_compressed(
            per_vertex_dir / f"{sample['sample_id']}_expanded_query_gap.npz",
            nearest_surface_distance=distances,
            h_current=h_current,
            distance_over_h_current=ratio,
        )
    return rows, ratios


def _nearest_surface_distances(
    query_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    gt_faces: np.ndarray,
) -> np.ndarray:
    import trimesh

    surface = trimesh.Trimesh(
        vertices=np.asarray(gt_vertices, dtype=np.float64),
        faces=np.asarray(gt_faces, dtype=np.int64),
        process=False,
    )
    try:
        _, distances, _ = trimesh.proximity.closest_point(
            surface, np.asarray(query_vertices, dtype=np.float64)
        )
    except Exception as error:
        raise RuntimeError(
            "Exact nearest-surface distance failed; refusing a nearest-vertex "
            "substitute for this diagnostic."
        ) from error
    distances = np.asarray(distances, dtype=np.float64)
    if distances.shape != (len(query_vertices),) or not np.isfinite(distances).all():
        raise RuntimeError("Nearest-surface distances are incomplete or non-finite.")
    return distances


def _ratio_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Ratio values must be non-empty, finite, and non-negative.")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _outside_augmentation_summary(
    ratio: np.ndarray, max_offset_h: float
) -> dict[str, float]:
    values = np.asarray(ratio, dtype=np.float64)
    return {
        "training_max_offset_h": float(max_offset_h),
        "fraction_above_training_max_offset_h": float(
            np.mean(values > max_offset_h)
        ),
        "median_multiple_of_training_max": float(
            np.median(values) / max(max_offset_h, 1e-30)
        ),
        "p95_multiple_of_training_max": float(
            np.quantile(values, 0.95) / max(max_offset_h, 1e-30)
        ),
    }


def _mesh_normalization_diagnostics(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], Mapping[str, Any]]:
    manifest = _read_json(manifest_path)
    render_spec = dict(manifest["render_spec"])
    cube_half_extent = float(render_spec["cube_half_extent"])
    raw_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(manifest_path, split)
        for index in range(len(dataset)):
            sample = dataset.load_static(index)
            vertices = torch.as_tensor(sample["vertices"]).numpy()
            bbox_min = vertices.min(axis=0)
            bbox_max = vertices.max(axis=0)
            center_norm = float(np.linalg.norm(0.5 * (bbox_min + bbox_max)))
            diagonal = float(np.linalg.norm(bbox_max - bbox_min))
            max_radius = float(np.linalg.norm(vertices, axis=1).max())
            raw_rows.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "split": split,
                    "bbox_center_norm": center_norm,
                    "bbox_diagonal": diagonal,
                    "max_vertex_radius": max_radius,
                }
            )
    median_diagonal = float(np.median([row["bbox_diagonal"] for row in raw_rows]))
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = dict(raw)
        row["centered_within_1e_6"] = bool(row["bbox_center_norm"] <= 1e-6)
        row["inside_camera_cube_half_extent"] = bool(
            row["max_vertex_radius"] < cube_half_extent
        )
        row["diagonal_within_10_percent_of_median"] = bool(
            abs(row["bbox_diagonal"] / median_diagonal - 1.0) <= 0.10
        )
        row["anomaly"] = not bool(
            row["centered_within_1e_6"]
            and row["inside_camera_cube_half_extent"]
            and row["diagonal_within_10_percent_of_median"]
        )
        rows.append(row)
    centers = np.asarray([row["bbox_center_norm"] for row in rows])
    diagonals = np.asarray([row["bbox_diagonal"] for row in rows])
    radii = np.asarray([row["max_vertex_radius"] for row in rows])
    summary = {
        "mesh_count": len(rows),
        "normalize_mesh_during_preparation": False,
        "bbox_center_norm": _min_median_max(centers),
        "bbox_diagonal": _min_median_max(diagonals),
        "max_vertex_radius": _min_median_max(radii),
        "cube_half_extent": cube_half_extent,
        "minimum_camera_extent_clearance": float(cube_half_extent - radii.max()),
        "anomaly_count": int(sum(row["anomaly"] for row in rows)),
        "anomaly_sample_ids": [row["sample_id"] for row in rows if row["anomaly"]],
        "has_anomalies": any(row["anomaly"] for row in rows),
        "anomaly_rules": {
            "bbox_center_norm_max": 1e-6,
            "max_vertex_radius_below_cube_half_extent": cube_half_extent,
            "bbox_diagonal_relative_to_median": "within 10%",
        },
    }
    return rows, summary, render_spec


def _curvature_error_groups(
    run: Path,
    validation_sample_ids: Sequence[str],
    epsilon: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    h_values: list[np.ndarray] = []
    for sample_id in validation_sample_ids:
        path = run / "per_vertex_diagnostics" / f"{sample_id}_gt_query.npz"
        with np.load(path) as values:
            valid = np.asarray(values["valid_scale_mask"], dtype=bool)
            targets.append(np.asarray(values["delta_hat_gt"])[valid])
            predictions.append(np.asarray(values["delta_hat_prediction"])[valid])
            h_values.append(np.asarray(values["h_gt"])[valid])
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    h = np.concatenate(h_values)
    magnitude = np.linalg.norm(target, axis=1)
    normalized_error = np.linalg.norm(prediction - target, axis=1)
    raw_error = normalized_error * (h**2 + epsilon)
    order = np.argsort(magnitude)
    count = len(order)
    bottom_10 = order[: max(1, int(round(0.10 * count)))]
    top_10_count = max(1, int(round(0.10 * count)))
    top_1_count = max(1, int(round(0.01 * count)))
    groups = (
        ("lowest_magnitude_bottom_10_percent", bottom_10),
        ("smooth_bottom_90_percent", order[:-top_10_count]),
        ("high_curvature_top_10_percent", order[-top_10_count:]),
        ("high_curvature_top_1_percent", order[-top_1_count:]),
    )
    rows = []
    for name, indices in groups:
        rows.append(
            {
                "group": name,
                "vertex_count": int(len(indices)),
                "mean_gt_delta_hat_magnitude": float(magnitude[indices].mean()),
                "median_gt_delta_hat_magnitude": float(np.median(magnitude[indices])),
                "mean_normalized_vector_l2_error": float(
                    normalized_error[indices].mean()
                ),
                "median_normalized_vector_l2_error": float(
                    np.median(normalized_error[indices])
                ),
                "mean_raw_vector_l2_error": float(raw_error[indices].mean()),
            }
        )
    by_name = {row["group"]: row for row in rows}
    smooth = by_name["smooth_bottom_90_percent"]
    high = by_name["high_curvature_top_10_percent"]
    summary = {
        "valid_vertex_count": count,
        "magnitude_percentiles": {
            "p10": float(np.quantile(magnitude, 0.10)),
            "p90": float(np.quantile(magnitude, 0.90)),
            "p99": float(np.quantile(magnitude, 0.99)),
        },
        "high10_to_smooth90_normalized_error_ratio": float(
            high["mean_normalized_vector_l2_error"]
            / smooth["mean_normalized_vector_l2_error"]
        ),
        "interpretation": (
            "Normalized target error is the primary like-for-like comparison. "
            "Raw error also includes group-dependent h^2 and must not be used to "
            "reverse the curvature-difficulty conclusion."
        ),
    }
    return rows, summary


def _min_median_max(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def _report(summary: Mapping[str, Any]) -> str:
    gap = summary["expanded_query_gap_aggregate"]
    normalization = summary["mesh_normalization"]
    lines = [
        "# Sofa50 query-distribution and normalization diagnostics",
        "",
        "The canonical recovery baseline uses "
        "`renderer_visible_any * confidence_prediction`, the global "
        "`lambda_anchor=0.01`, and `unseen_anchor_weight=0.0`.",
        "",
        "## Real expanded-query distance to GT surface",
        "",
        "| Mesh | Mean d/h | Median | P90 | P95 | P99 | Max | Fraction > 0.001h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["expanded_query_gap_per_mesh"]:
        lines.append(
            f"| {row['sample_id']} | {_fmt(row['mean'])} | {_fmt(row['median'])} | "
            f"{_fmt(row['p90'])} | {_fmt(row['p95'])} | {_fmt(row['p99'])} | "
            f"{_fmt(row['max'])} | "
            f"{_fmt(row['fraction_above_training_max_offset_h'])} |"
        )
    lines.extend(
        [
            "",
            "All-vertex aggregate: mean/median/p90/p95/p99/max = "
            f"`{_fmt(gap['mean'])}` / `{_fmt(gap['median'])}` / "
            f"`{_fmt(gap['p90'])}` / `{_fmt(gap['p95'])}` / "
            f"`{_fmt(gap['p99'])}` / `{_fmt(gap['max'])}`. The training cap is "
            f"`{_fmt(gap['training_max_offset_h'])}h`; "
            f"`{_fmt(gap['fraction_above_training_max_offset_h'])}` of expanded "
            "vertices exceed it.",
            "",
            "## Mesh/camera normalization",
            "",
            f"Across `{normalization['mesh_count']}` Sofa meshes, bbox-center norm "
            f"min/median/max is `{_triple(normalization['bbox_center_norm'])}`, bbox "
            f"diagonal is `{_triple(normalization['bbox_diagonal'])}`, and max vertex "
            f"radius is `{_triple(normalization['max_vertex_radius'])}`. Camera cube "
            f"half-extent is `{_fmt(normalization['cube_half_extent'])}`. Flagged "
            f"anomalies: `{normalization['anomaly_count']}`.",
            "",
            "## Relative Laplacian-magnitude error groups",
            "",
            "| Group | Vertices | Mean ||delta_hat_gt|| | Mean normalized error | Mean raw error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["curvature_error_groups"]:
        lines.append(
            f"| {row['group']} | {row['vertex_count']} | "
            f"{_fmt(row['mean_gt_delta_hat_magnitude'])} | "
            f"{_fmt(row['mean_normalized_vector_l2_error'])} | "
            f"{_fmt(row['mean_raw_vector_l2_error'])} |"
        )
    lines.extend(
        [
            "",
            "These groups use relative percentiles only. Small uniform-Laplacian "
            "magnitude is not treated as a strict plane detector and does not alter "
            "training loss, confidence, or recovery weights. Normalized target error "
            "is the like-for-like comparison; raw error additionally contains each "
            "group's h² distribution.",
            "",
            "## Conclusion",
            "",
            summary["conclusion"],
        ]
    )
    return "\n".join(lines) + "\n"


def _triple(values: Mapping[str, Any]) -> str:
    return "/".join(_fmt(values[key]) for key in ("min", "median", "max"))


def _fmt(value: Any) -> str:
    return f"{float(value):.7g}"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_sanitize(value), indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
