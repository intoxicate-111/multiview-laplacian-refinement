from __future__ import annotations

import copy
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from mlr.data import Camera, Mesh
from mlr.synthetic import SyntheticRenderConfig, render_mesh_face_ids

from .diagnostics import _amp_settings, _loss_kwargs
from .image_ablation import (
    IMAGE_CONDITIONS,
    _plot_condition_metrics,
    _predict_conditions,
    _write_condition_csv,
    summarize_image_ablation,
)
from .multi_dataset import PreparedMeshDataset
from .multi_trainer import _build_model
from .projection import project_vertices
from .renderer_visibility import VISIBILITY_CONDITIONS
from .trainer import load_checkpoint


def run_renderer_visibility_ablation(
    run_dir: str | Path,
    manifest: str | Path,
    *,
    output_dir: str | Path | None = None,
    device: str = "cuda",
    seed: int = 7,
    split: str = "validation",
    overwrite: bool = False,
    visualizations: int = 3,
) -> dict[str, Any]:
    """Compare one frozen checkpoint under four precomputed visibility masks."""

    run_dir = Path(run_dir).resolve()
    manifest = Path(manifest).resolve()
    output_dir = Path(output_dir or run_dir / "renderer_visibility_ablation").resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_json(run_dir / "config.json")
    checkpoint_path = run_dir / "best.pt"
    dataset = PreparedMeshDataset.from_manifest(manifest, split)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _build_model(config, None, False).to(resolved_device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=resolved_device)
    model.eval()
    loss_kwargs = _loss_kwargs(config)
    amp_enabled, amp_dtype = _amp_settings(config, resolved_device)

    visibility = _visibility_diagnostics(dataset, output_dir / "visibility", visualizations)
    predictions: dict[str, Any] = {}
    for condition in VISIBILITY_CONDITIONS:
        print(f"Prediction ablation: {condition}", flush=True)
        condition_config = copy.deepcopy(config)
        condition_config["renderer_visibility"] = {"condition": condition}
        condition_config.setdefault("query_training", {})["enabled"] = False
        prediction_dir = output_dir / "predictions" / condition
        records = _predict_conditions(
            model,
            dataset,
            condition_config,
            resolved_device,
            amp_enabled,
            amp_dtype,
            loss_kwargs,
            seed,
            prediction_dir / "arrays",
        )
        metrics = summarize_image_ablation(records, loss_kwargs)
        metrics["per_mesh_metrics"] = _per_mesh_metrics(records)
        _write_json(prediction_dir / "metrics.json", metrics)
        _write_condition_csv(prediction_dir / "metrics.csv", metrics)
        _write_per_mesh_csv(prediction_dir / "per_mesh_metrics.csv", metrics["per_mesh_metrics"])
        _plot_condition_metrics(prediction_dir / "metrics.png", metrics)
        predictions[condition] = metrics

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "manifest": str(manifest),
        "split": split,
        "seed": seed,
        "amp_enabled": amp_enabled,
        "visibility_definition": (
            "frustum intersection with precomputed renderer-native face-ID visibility; "
            "no depth image is loaded or compared"
        ),
        "visibility": visibility,
        "predictions": predictions,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _visibility_diagnostics(
    dataset: PreparedMeshDataset, output_dir: Path, visualizations: int
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(len(dataset)):
        sample = dataset.load_static(index)
        vertices = sample["vertices"].float()
        image_size = int(sample["prepared_image_size"])
        projection = project_vertices(
            vertices,
            sample["intrinsics"].float(),
            sample["extrinsics"].float(),
            (image_size, image_size),
        )
        frustum = projection.frustum_valid.cpu().numpy()
        masks = {
            "frustum_only": np.ones_like(frustum),
            "backface_only": sample["visibility_backface_only"].cpu().numpy(),
            "occlusion_only": sample["visibility_occlusion_only"].cpu().numpy(),
            "backface_and_occlusion": sample[
                "visibility_backface_and_occlusion"
            ].cpu().numpy(),
        }
        target = sample["normalized_laplacian_target"].cpu().numpy()
        target_magnitude = np.linalg.norm(target, axis=1)
        high = target_magnitude >= np.quantile(target_magnitude, 0.9)
        conditions = {}
        counts_for_plot = {}
        for name, renderer_mask in masks.items():
            final = frustum & renderer_mask
            counts = final.sum(axis=0)
            counts_for_plot[name] = counts
            conditions[name] = {
                "mean_visible_views_per_vertex": float(counts.mean()),
                "median_visible_views_per_vertex": float(np.median(counts)),
                "zero_visible_vertex_ratio": float(np.mean(counts == 0)),
                "final_visible_ratio": float(final.mean()),
                "per_view_visible_vertex_ratio": final.mean(axis=1).tolist(),
                "high_laplacian_mean_visible_views": float(counts[high].mean()),
                "high_laplacian_zero_visible_ratio": float(np.mean(counts[high] == 0)),
            }
        record = {
            "sample_id": str(sample["sample_id"]),
            "vertices": int(vertices.shape[0]),
            "frustum_valid_ratio": float(frustum.mean()),
            "backface_rejected_ratio": float(
                np.mean(frustum & ~masks["backface_only"])
            ),
            "occlusion_rejected_ratio": float(
                np.mean(frustum & ~masks["occlusion_only"])
            ),
            "conditions": conditions,
        }
        records.append(record)
        if index < visualizations:
            sample_dir = output_dir / _safe_name(str(sample["sample_id"]))
            sample_dir.mkdir(parents=True, exist_ok=True)
            _plot_vertex_counts(sample_dir / "vertex_visibility_counts.png", vertices.numpy(), counts_for_plot)
            _write_render_examples(sample, sample_dir)
    _write_visibility_csv(output_dir / "per_mesh_visibility.csv", records)
    aggregate = {
        condition: {
            key: float(np.mean([row["conditions"][condition][key] for row in records]))
            for key in (
                "mean_visible_views_per_vertex",
                "median_visible_views_per_vertex",
                "zero_visible_vertex_ratio",
                "final_visible_ratio",
                "high_laplacian_mean_visible_views",
                "high_laplacian_zero_visible_ratio",
            )
        }
        for condition in VISIBILITY_CONDITIONS
    }
    result = {"aggregate_mesh_mean": aggregate, "per_mesh": records}
    _write_json(output_dir / "metrics.json", result)
    return result


def _per_mesh_metrics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        with np.load(record["prediction_path"]) as archive:
            valid = archive["valid_mask"].astype(bool)
            target = archive["target"][valid]
            target_mag = np.linalg.norm(target, axis=1)
            thresholds = {q: np.quantile(target_mag, q) for q in (0.9, 0.95, 0.99)}
            original = archive["original_rgb"][valid]
            for condition in IMAGE_CONDITIONS:
                prediction = archive[condition][valid]
                pred_mag = np.linalg.norm(prediction, axis=1)
                denom = np.maximum(target_mag * pred_mag, 1e-12)
                cosine = np.sum(target * prediction, axis=1) / denom
                row = {
                    "sample_id": str(record["sample_id"]),
                    "image_condition": condition,
                    "validation_loss": float(record["losses"][condition]),
                    "zero_predictor_loss": float(record["zero_predictor_loss"]),
                    "relative_improvement_vs_zero": _relative_improvement(
                        float(record["zero_predictor_loss"]),
                        float(record["losses"][condition]),
                    ),
                    "prediction_target_magnitude_ratio": _safe_div(
                        float(pred_mag.mean()), float(target_mag.mean())
                    ),
                    "cosine_similarity": float(cosine.mean()),
                    "prediction_change_vs_original": float(
                        np.linalg.norm(prediction - original, axis=1).mean()
                    ),
                }
                for name, threshold in (
                    ("high10_cosine", thresholds[0.9]),
                    ("top5_cosine", thresholds[0.95]),
                    ("top1_cosine", thresholds[0.99]),
                ):
                    row[name] = float(cosine[target_mag >= threshold].mean())
                rows.append(row)
    return rows


def _write_render_examples(sample: Mapping[str, Any], output_dir: Path) -> None:
    root = Path(str(sample["_dataset_root"]))
    image_path = Path(sample["image_paths"][0])
    if not image_path.is_absolute():
        image_path = root / image_path
    with Image.open(image_path) as image:
        image.convert("RGB").save(output_dir / "rgb_view_0000.png")
    mesh = Mesh(sample["vertices"].numpy(), sample["faces"].numpy()).ensure_normals()
    camera = _camera(sample, 0)
    size = int(sample["prepared_image_size"])
    for name, culling in (("occlusion_only", False), ("backface_and_occlusion", True)):
        config = SyntheticRenderConfig(
            width=size,
            height=size,
            backend="opengl",
            normalize_mesh=False,
            antialiasing="none",
            backface_culling=culling,
            front_face_winding="ccw",
        )
        face_ids = render_mesh_face_ids(mesh, camera, config)
        Image.fromarray(_color_face_ids(face_ids)).save(output_dir / f"face_ids_{name}.png")


def _camera(sample: Mapping[str, Any], index: int) -> Camera:
    size = int(sample["prepared_image_size"])
    extrinsic = sample["extrinsics"][index].numpy()
    return Camera(
        sample["intrinsics"][index].numpy(),
        extrinsic[:3, :3],
        extrinsic[:3, 3],
        (size, size),
    )


def _color_face_ids(face_ids: np.ndarray) -> np.ndarray:
    ids = face_ids.astype(np.int64)
    rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
    valid = ids >= 0
    rgb[..., 0][valid] = (ids[valid] * 53 + 67) % 255
    rgb[..., 1][valid] = (ids[valid] * 97 + 31) % 255
    rgb[..., 2][valid] = (ids[valid] * 193 + 17) % 255
    return rgb


def _plot_vertex_counts(path: Path, vertices: np.ndarray, counts: Mapping[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    keep = np.linspace(0, len(vertices) - 1, min(len(vertices), 30000), dtype=np.int64)
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for axis, name in zip(axes.flat, VISIBILITY_CONDITIONS, strict=True):
        artist = axis.scatter(
            vertices[keep, 0], vertices[keep, 2], c=counts[name][keep], s=1, cmap="viridis"
        )
        axis.set_title(name)
        axis.set_aspect("equal")
        figure.colorbar(artist, ax=axis, label="visible views")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_visibility_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for record in records:
        for condition, metrics in record["conditions"].items():
            rows.append({"sample_id": record["sample_id"], "condition": condition, **metrics})
    _write_csv(path, rows)


def _write_per_mesh_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(path, rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Renderer-native visibility ablation",
        "",
        f"Checkpoint: `{summary['checkpoint']}` (epoch {summary['checkpoint_epoch']})",
        "",
        "No depth image was loaded or compared. Visibility comes from the RGB renderer's face-ID pass.",
        "",
        "| visibility | visible views | zero-view vertices | original loss | original vs zero predictor | |pred|/|GT| | high-10% cosine |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in VISIBILITY_CONDITIONS:
        visible = summary["visibility"]["aggregate_mesh_mean"][condition]
        prediction = summary["predictions"][condition]["conditions"]["original_rgb"]
        lines.append(
            f"| {condition} | {visible['mean_visible_views_per_vertex']:.3f} | "
            f"{visible['zero_visible_vertex_ratio']:.3%} | "
            f"{prediction['validation_loss']:.8g} | "
            f"{prediction['relative_improvement_vs_zero_predictor']:.3%} | "
            f"{prediction['mean_prediction_to_target_magnitude_ratio']:.3%} | "
            f"{prediction['magnitude_bins']['high_top10']['cosine_similarity']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(payload), indent=2) + "\n", encoding="utf-8")


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _relative_improvement(baseline: float, value: float) -> float:
    return (baseline - value) / baseline if baseline > 0 else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-12 else 0.0


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
