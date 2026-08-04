#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.data import Camera, Mesh
from mlr.io import load_mesh
from mlr.learned_laplacian.dataset import load_prepared_sample
from mlr.synthetic import (
    SyntheticRenderConfig,
    create_orbit_cameras,
    look_at_world_to_camera,
    render_mesh_view,
)


MESH_SPECS = (
    ("gt", "GT", "gt_cleaned.obj"),
    ("coarse", "Coarse", "coarse_corrupted.obj"),
    ("oracle", "Oracle", "oracle_refined.obj"),
    ("raw_geometry", "Raw geometry", "raw_geometry_refined.obj"),
    ("normalized_geometry", "Normalised geometry", "normalized_geometry_refined.obj"),
    ("raw_multiview", "Raw RGB", "raw_multiview_refined.obj"),
    ("normalized_multiview", "Normalised RGB", "normalized_multiview_refined.obj"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render cleaned Bunny target-mode comparisons.")
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--wireframe-only",
        action="store_true",
        help="Regenerate only the two close-up wireframe comparison images.",
    )
    args = parser.parse_args()
    root = args.output_root
    renders_dir = root / "renders"
    errors_dir = root / "errors"
    plots_dir = root / "plots"
    for directory in (renders_dir, errors_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sample = load_prepared_sample(args.sample)
    meshes = {key: load_mesh(root / filename).ensure_normals() for key, _, filename in MESH_SPECS}
    metrics = _mesh_metrics(root)
    cameras = create_orbit_cameras(
        meshes["gt"], 8, (args.image_size, args.image_size), radius_scale=1.25,
        elevation_degrees=20.0, fov_degrees=42.0
    )
    views = {"front": cameras[0], "side": cameras[2], "rear": cameras[5]}
    if args.wireframe_only:
        _write_wireframe_closeups(meshes, views["front"], renders_dir, args.image_size)
        return 0
    render_cache = _render_lit_views(meshes, views, args.image_size)
    _write_lit_outputs(meshes, metrics, render_cache, renders_dir, args.image_size)

    position_errors = _position_errors(meshes, meshes["gt"], errors_dir)
    position_limit = float(np.quantile(np.concatenate(list(position_errors.values())), 0.99))
    _write_error_outputs(
        meshes, position_errors, views["front"], errors_dir,
        args.image_size, position_limit, "position"
    )
    laplacian_errors = _laplacian_errors(root, errors_dir)
    laplacian_limit = float(np.quantile(np.concatenate(list(laplacian_errors.values())), 0.99))
    lap_meshes = {key: meshes[key] for key in laplacian_errors}
    _write_error_outputs(
        lap_meshes, laplacian_errors, views["front"], errors_dir,
        args.image_size, laplacian_limit, "laplacian"
    )
    _write_wireframe_closeups(meshes, views["front"], renders_dir, args.image_size)
    plot_statistics = _write_plots(sample, metrics, root, plots_dir)
    visualization = {
        "camera_definition": {
            "source": "cleaned GT bounding box via create_orbit_cameras",
            "radius_scale": 1.25,
            "elevation_degrees": 20.0,
            "fov_degrees": 42.0,
            "image_size": args.image_size,
            "view_indices": {"front": 0, "side": 2, "rear": 5},
        },
        "position_error_shared_range": [0.0, position_limit],
        "laplacian_error_shared_range": [0.0, laplacian_limit],
        "position_error_definition": "per-vertex exact closest distance to cleaned GT surface",
        "laplacian_error_definition": "endpoint error in recovered raw Laplacian space",
        "plot_statistics": plot_statistics,
    }
    (root / "visualization.json").write_text(
        json.dumps(visualization, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(visualization, indent=2))
    return 0


def _mesh_metrics(root: Path) -> dict[str, dict]:
    comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    modes = comparison["modes"]
    common = modes["raw_geometry"]["geometry"]
    result = {"gt": {"point_to_surface_mean": 0.0, "chamfer": 0.0, "target_position_rmse": 0.0}}
    result["coarse"] = common["coarse"]
    result["oracle"] = common["oracle"]
    for mode in ("raw_geometry", "normalized_geometry", "raw_multiview", "normalized_multiview"):
        result[mode] = modes[mode]["geometry"]["predicted"]
    return result


def _render_lit_views(meshes: dict[str, Mesh], views: dict, image_size: int) -> dict:
    config = SyntheticRenderConfig(
        width=image_size, height=image_size, render_mode="lit", normalize_mesh=False, backend="cpu"
    )
    cache = {}
    for view_name, camera in views.items():
        for key, mesh in meshes.items():
            cache[(view_name, key)] = render_mesh_view(mesh, camera, config)[0]
    return cache


def _panel_label(key: str, label: str, mesh: Mesh, metrics: dict[str, dict]) -> str:
    values = metrics[key]
    return (
        f"{label} | V {mesh.num_vertices} F {mesh.num_faces}\n"
        f"P2S {values.get('point_to_surface_mean', 0.0):.4g}  "
        f"Chamfer {values.get('chamfer', 0.0):.4g}  "
        f"RMSE {values.get('target_position_rmse', 0.0):.4g}"
    )


def _write_lit_outputs(meshes, metrics, cache, output_dir: Path, image_size: int) -> None:
    labels = {key: label for key, label, _ in MESH_SPECS}
    filenames = {
        "gt": "gt_cleaned.png",
        "coarse": "coarse_corrupted.png",
        "oracle": "oracle_refined.png",
        "raw_geometry": "raw_geometry_refined.png",
        "normalized_geometry": "normalized_geometry_refined.png",
        "raw_multiview": "raw_multiview_refined.png",
        "normalized_multiview": "normalized_multiview_refined.png",
    }
    for key in meshes:
        Image.fromarray(cache[("front", key)]).save(output_dir / filenames[key])
    for view_name in ("front", "side", "rear"):
        panels = [
            (_panel_label(key, labels[key], meshes[key], metrics), cache[(view_name, key)])
            for key, _, _ in MESH_SPECS
        ]
        grid = _make_grid(panels, image_size, columns=3, label_height=42)
        grid.save(output_dir / f"comparison_{view_name}.png")
        if view_name == "front":
            grid.save(output_dir / "mesh_comparison_grid.png")


def _position_errors(meshes: dict[str, Mesh], gt: Mesh, output_dir: Path) -> dict[str, np.ndarray]:
    import trimesh

    surface = trimesh.Trimesh(vertices=gt.vertices, faces=gt.faces, process=False)
    errors = {}
    for key, mesh in meshes.items():
        if key == "gt":
            values = np.zeros(mesh.num_vertices, dtype=np.float64)
        else:
            _, values, _ = trimesh.proximity.closest_point(surface, mesh.vertices)
            values = np.asarray(values, dtype=np.float64)
        errors[key] = values
        np.save(output_dir / f"{key}_position_error.npy", values)
    return errors


def _laplacian_errors(root: Path, output_dir: Path) -> dict[str, np.ndarray]:
    result = {}
    for mode in ("raw_geometry", "normalized_geometry", "raw_multiview", "normalized_multiview"):
        values = np.load(root / mode / "laplacian_error.npy")
        result[mode] = values
        np.save(output_dir / f"{mode}_laplacian_error.npy", values)
    return result


def _write_error_outputs(
    meshes: dict[str, Mesh], errors: dict[str, np.ndarray], camera, output_dir: Path,
    image_size: int, limit: float, kind: str
) -> None:
    panels = []
    for key, mesh in meshes.items():
        image = _error_render(mesh, errors[key], camera, image_size, limit)
        image.save(output_dir / f"{key}_{kind}_error.png")
        panels.append((f"{key.replace('_', ' ')} | shared 0..{limit:.4g}", np.asarray(image)))
    name = "position_error_comparison.png" if kind == "position" else "laplacian_error_comparison.png"
    _make_grid(panels, image_size, columns=3, label_height=26).save(output_dir / name)


def _error_render(mesh: Mesh, values: np.ndarray, camera, image_size: int, limit: float) -> Image.Image:
    import matplotlib

    config = SyntheticRenderConfig(
        width=image_size, height=image_size, render_mode="lit", normalize_mesh=False, backend="cpu"
    )
    rgb, mask, depth = render_mesh_view(mesh, camera, config)
    image = Image.fromarray((rgb.astype(np.float32) * 0.35 + 165.0).clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    pixels, z = camera.project(mesh.vertices)
    norm = np.clip(values / max(limit, 1e-12), 0.0, 1.0)
    colours = (matplotlib.colormaps["turbo"](norm)[:, :3] * 255).astype(np.uint8)
    valid = (z > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < image_size) & (pixels[:, 1] >= 0) & (pixels[:, 1] < image_size)
    order = np.argsort(z)[::-1]
    for index in order:
        if not valid[index]:
            continue
        x, y = np.rint(pixels[index]).astype(int)
        if x >= image_size or y >= image_size:
            continue
        if mask[y, x] and abs(float(depth[y, x]) - float(z[index])) < 0.025:
            colour = tuple(int(v) for v in colours[index])
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=colour)
    _draw_colour_bar(image, limit)
    return image


def _draw_colour_bar(image: Image.Image, limit: float) -> None:
    import matplotlib

    draw = ImageDraw.Draw(image)
    width, height = image.size
    x0, x1, y0, y1 = 12, width - 12, height - 16, height - 8
    for x in range(x0, x1):
        value = (x - x0) / max(x1 - x0 - 1, 1)
        colour = tuple(int(v * 255) for v in matplotlib.colormaps["turbo"](value)[:3])
        draw.line((x, y0, x, y1), fill=colour)
    draw.text((x0, y0 - 11), "0", fill=(0, 0, 0))
    draw.text((x1 - 54, y0 - 11), f"{limit:.3g}", fill=(0, 0, 0))


def _write_wireframe_closeups(meshes, camera, output_dir: Path, image_size: int) -> None:
    keys = ("gt", "coarse", "oracle", "normalized_geometry")
    gt = meshes["gt"]
    y = gt.vertices[:, 1]
    z = gt.vertices[:, 2]
    ear_camera = _closeup_camera(
        gt,
        (y >= np.quantile(y, 0.85)) & (z <= np.quantile(z, 0.35)),
        camera,
        image_size,
    )
    feet_camera = _closeup_camera(
        gt,
        (y <= np.quantile(y, 0.10)) & (z <= np.quantile(z, 0.50)),
        camera,
        image_size,
    )
    labels = {"gt": "GT", "coarse": "Coarse", "oracle": "Oracle", "normalized_geometry": "Normalised"}
    ear_panels = [
        (labels[key], np.asarray(_wireframe_render(meshes[key], ear_camera, image_size)))
        for key in keys
    ]
    feet_panels = [
        (labels[key], np.asarray(_wireframe_render(meshes[key], feet_camera, image_size)))
        for key in keys
    ]
    _make_grid(ear_panels, image_size, columns=4, label_height=24).save(output_dir / "wireframe_ear_comparison.png")
    _make_grid(feet_panels, image_size, columns=4, label_height=24).save(output_dir / "wireframe_feet_comparison.png")


def _closeup_camera(gt: Mesh, region_mask: np.ndarray, base_camera: Camera, image_size: int) -> Camera:
    region = gt.vertices[np.asarray(region_mask, dtype=bool)]
    if region.shape[0] < 3:
        raise ValueError("A wireframe close-up region must contain at least three vertices.")
    region_min = region.min(axis=0)
    region_max = region.max(axis=0)
    target = 0.5 * (region_min + region_max)
    projected_region = (region - target) @ base_camera.rotation.T
    screen_extent = projected_region[:, :2].max(axis=0) - projected_region[:, :2].min(axis=0)
    extent = max(float(np.linalg.norm(screen_extent)), 1e-3)
    full_target = 0.5 * (gt.vertices.min(axis=0) + gt.vertices.max(axis=0))
    direction = base_camera.center - full_target
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    center = target + 1.15 * extent * direction
    rotation, translation = look_at_world_to_camera(center, target)
    focal = 0.5 * image_size / math.tan(math.radians(42.0) * 0.5)
    intrinsics = np.array(
        [[focal, 0.0, image_size * 0.5], [0.0, focal, image_size * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return Camera(
        intrinsics=intrinsics,
        rotation=rotation,
        translation=translation,
        image_size=(image_size, image_size),
        name="wireframe_closeup",
    )


def _wireframe_render(mesh: Mesh, camera: Camera, image_size: int) -> Image.Image:
    config = SyntheticRenderConfig(
        width=image_size, height=image_size, render_mode="lit", normalize_mesh=False, backend="cpu"
    )
    rgb, mask, depth = render_mesh_view(mesh, camera, config)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    pixels, z = camera.project(mesh.vertices)
    for face in mesh.faces:
        if np.any(z[face] <= 0):
            continue
        centroid = np.rint(pixels[face].mean(axis=0)).astype(int)
        x, y = int(centroid[0]), int(centroid[1])
        if x < 0 or x >= image_size or y < 0 or y >= image_size or not mask[y, x]:
            continue
        face_depth = float(z[face].mean())
        if abs(float(depth[y, x]) - face_depth) > max(0.0025, 0.015 * face_depth):
            continue
        points = [tuple(pixels[index]) for index in (face[0], face[1], face[2], face[0])]
        draw.line(points, fill=(120, 235, 255), width=1)
    return image


def _write_plots(sample: dict, metrics: dict, root: Path, output_dir: Path) -> dict:
    import matplotlib.pyplot as plt

    h = sample["local_edge_length"].double().numpy()
    h2 = sample["local_edge_scale"].double().numpy()
    raw = np.linalg.norm(sample["raw_laplacian_target"].double().numpy(), axis=1)
    normalized = np.linalg.norm(sample["normalized_laplacian_target"].double().numpy(), axis=1)
    valid = sample["valid_scale_mask"].numpy()
    series = {
        "local_edge_length_histogram.png": (h[valid], "Local edge length h", "length"),
        "local_edge_scale_histogram.png": (h2[valid], "Local edge scale h²", "length²"),
        "raw_target_magnitude_histogram.png": (raw[valid], "Raw target magnitude", "Laplacian magnitude"),
        "normalized_target_magnitude_histogram.png": (normalized[valid], "Normalised target magnitude", "1/length"),
    }
    for filename, (values, title, xlabel) in series.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        positive = values[values > 0]
        ax.hist(positive, bins=np.logspace(np.log10(positive.min()), np.log10(positive.max()), 80))
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("vertex count")
        ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(output_dir / filename, dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    positive_raw = raw[valid & (raw > 0)]
    positive_norm = normalized[valid & (normalized > 0)]
    low = min(positive_raw.min(), positive_norm.min()); high = max(positive_raw.max(), positive_norm.max())
    bins = np.logspace(np.log10(low), np.log10(high), 100)
    ax.hist(positive_raw, bins=bins, histtype="step", label="raw")
    ax.hist(positive_norm, bins=bins, histtype="step", label="normalised")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(); ax.set_xlabel("target magnitude"); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(output_dir / "raw_vs_normalized_target_log_histogram.png", dpi=150); plt.close(fig)
    indices = np.flatnonzero(valid)[::max(int(valid.sum()) // 10000, 1)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(h2[indices], raw[indices], s=3, alpha=0.35, label="raw")
    ax.scatter(h2[indices], normalized[indices], s=3, alpha=0.35, label="normalised")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("h²"); ax.set_ylabel("target magnitude"); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "target_magnitude_vs_edge_scale_scatter.png", dpi=150); plt.close(fig)
    comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode in ("raw_geometry", "normalized_geometry", "raw_multiview", "normalized_multiview"):
        history = json.loads((root / mode / "training_history.json").read_text(encoding="utf-8"))
        ax.plot([x["step"] for x in history], [x["loss"] for x in history], label=mode.replace("_", " "))
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("training target-space loss"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "training_loss_comparison.png", dpi=150); plt.close(fig)
    names = ["coarse", "oracle", "raw_geometry", "normalized_geometry", "raw_multiview", "normalized_multiview"]
    values = {"P2S mean": [], "Chamfer": [], "Target RMSE": []}
    for name in names:
        item = metrics[name]
        values["P2S mean"].append(item["point_to_surface_mean"])
        values["Chamfer"].append(item["chamfer"])
        values["Target RMSE"].append(item["target_position_rmse"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (metric_name, metric_values) in zip(axes, values.items()):
        ax.bar(range(len(names)), metric_values); ax.set_title(metric_name); ax.set_yscale("log")
        ax.set_xticks(range(len(names)), [name.replace("_", "\n") for name in names], rotation=30, ha="right", fontsize=7)
    fig.tight_layout(); fig.savefig(output_dir / "geometry_metric_comparison.png", dpi=150); plt.close(fig)
    stats = {
        "valid_vertices": int(valid.sum()), "excluded_invalid_vertices": int((~valid).sum()),
        "histogram_zero_values_excluded": True,
        "h": _stats(h[valid]), "h2": _stats(h2[valid]), "raw": _stats(raw[valid]), "normalized": _stats(normalized[valid]),
    }
    (output_dir / "plot_statistics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def _stats(values: np.ndarray) -> dict[str, float]:
    return {"min": float(values.min()), "median": float(np.median(values)), "mean": float(values.mean()), "p95": float(np.quantile(values, 0.95)), "p99": float(np.quantile(values, 0.99)), "max": float(values.max())}


def _make_grid(panels, image_size: int, columns: int, label_height: int) -> Image.Image:
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * image_size, rows * (image_size + label_height)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(panels):
        x = (index % columns) * image_size; y = (index // columns) * (image_size + label_height)
        canvas.paste(Image.fromarray(array.astype(np.uint8)), (x, y + label_height))
        draw.multiline_text((x + 4, y + 3), label, fill=(245, 245, 245), spacing=2)
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
