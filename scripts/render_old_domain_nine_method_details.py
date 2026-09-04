#!/usr/bin/env python3
from __future__ import annotations

"""Render the two locked old-domain eight-method qualitative detail panels."""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.data import Camera, Mesh
from mlr.io import load_mesh
from mlr.synthetic import SyntheticRenderConfig, look_at_world_to_camera, render_mesh_views_opengl


METHODS = (
    ("GT", "gt"),
    ("Previous Ours", "previous_ours"),
    ("NDS", "nds"),
    ("nvdiffrec", "nvdiffrec"),
    ("ExMesh", "exmesh"),
    ("Arm B", "arm_b"),
    ("Arm E", "arm_e"),
    ("Frozen B+E", "frozen_b_e"),
)


@dataclass(frozen=True)
class PanelSpec:
    sample_id: str
    title: str
    output_name: str
    reference_image: str
    reference_aspect_ratio: float
    azimuth_degrees: float
    elevation_degrees: float
    target_x_fraction: float
    target_y_fraction: float
    distance_scale: float
    fov_degrees: float


SPECS = (
    PanelSpec(
        sample_id="43bd0910-1dd1-4b1e-9ba2-e9801e6b5761__v01",
        title="Reference-matched full front view",
        output_name="43bd0910_v01_front_8method_transparent.png",
        reference_image="QQ_1787904603868.png",
        reference_aspect_ratio=2582.0 / 1410.0,
        azimuth_degrees=90.0,
        elevation_degrees=-90.0,
        target_x_fraction=0.0,
        target_y_fraction=0.0,
        distance_scale=1.60,
        fov_degrees=36.0,
    ),
    PanelSpec(
        sample_id="5c226f2b-aad3-4371-a3f9-ea2ee9a63327__v02",
        title="Reference-matched low-left oblique view",
        output_name="5c226f2b_v02_oblique_8method_transparent.png",
        reference_image="QQ_1787904538059.png",
        reference_aspect_ratio=1776.0 / 1587.0,
        azimuth_degrees=135.0,
        elevation_degrees=-20.0,
        target_x_fraction=0.0,
        target_y_fraction=-0.02,
        distance_scale=1.12,
        fov_degrees=44.0,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def _fabric_modulation(height: int, width: int, sample_id: str) -> np.ndarray:
    """Return a subtle deterministic image-space linen weave shared by every method."""

    seed = int.from_bytes(hashlib.sha256(sample_id.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((height, width), dtype=np.float64)
    warp = 0.018 * np.sin(2.0 * math.pi * xx / 4.0)
    weft = 0.014 * np.sin(2.0 * math.pi * yy / 3.0)
    grain = rng.normal(0.0, 0.010, size=(height, width))
    return np.clip(1.0 + warp + weft + grain, 0.93, 1.07)


def _apply_fabric(rgb: np.ndarray, mask: np.ndarray, modulation: np.ndarray) -> np.ndarray:
    textured = np.asarray(rgb, dtype=np.float64).copy()
    textured[mask] *= modulation[mask, None]
    return np.asarray(np.clip(np.rint(textured), 0.0, 255.0), dtype=np.uint8)


def _camera(
    gt: Mesh, spec: PanelSpec, width: int, height: int
) -> tuple[Camera, dict[str, object]]:
    vertices = np.asarray(gt.vertices, dtype=np.float64)
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = 0.5 * (lower + upper)
    extent = upper - lower
    target = center.copy()
    target[0] = center[0] + spec.target_x_fraction * extent[0]
    target[1] = center[1] + spec.target_y_fraction * extent[1]
    radius = spec.distance_scale * float(np.linalg.norm(extent))
    azimuth = math.radians(spec.azimuth_degrees)
    elevation = math.radians(spec.elevation_degrees)
    camera_center = target + radius * np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.sin(azimuth),
        ],
        dtype=np.float64,
    )
    focal = 0.5 * width / math.tan(math.radians(spec.fov_degrees) * 0.5)
    intrinsics = np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation, translation = look_at_world_to_camera(camera_center, target)
    camera = Camera(intrinsics, rotation, translation, (width, height), spec.title)
    metadata = {
        "azimuth_degrees": spec.azimuth_degrees,
        "elevation_degrees": spec.elevation_degrees,
        "fov_degrees": spec.fov_degrees,
        "reference_image": spec.reference_image,
        "reference_aspect_ratio": spec.reference_aspect_ratio,
        "target": target.tolist(),
        "camera_center": camera_center.tolist(),
        "distance": radius,
    }
    return camera, metadata


def _render_panel(mesh_root: Path, output_dir: Path, spec: PanelSpec, width: int) -> dict[str, object]:
    height = int(round(width / spec.reference_aspect_ratio))
    entries: list[tuple[str, str, Path, Mesh]] = []
    for label, directory in METHODS:
        path = mesh_root / directory / f"{spec.sample_id}.obj"
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append((label, directory, path, load_mesh(path)))

    camera, camera_metadata = _camera(entries[0][3], spec, width, height)
    config = SyntheticRenderConfig(
        width=width,
        height=height,
        render_mode="lit",
        backend="opengl",
        normalize_mesh=False,
        background_color=(255, 255, 255),
        object_color=(151, 101, 67),
        light_direction=(0.35, -0.45, 0.82),
        antialiasing="msaa4",
        backface_culling=False,
    )
    renders = []
    fabric = _fabric_modulation(height, width, spec.sample_id)
    for _, _, _, mesh in entries:
        rgb, mask, _ = render_mesh_views_opengl(mesh, [camera], config)[0]
        mask = np.asarray(mask, dtype=bool)
        rgb = _apply_fabric(rgb, mask, fabric)
        rgba = np.concatenate(
            [np.asarray(rgb, dtype=np.uint8), (mask.astype(np.uint8) * 255)[..., None]],
            axis=2,
        )
        renders.append(rgba)

    label_height = max(52, width // 13)
    canvas = Image.new("RGBA", (len(entries) * width, height + label_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    label_font = _font(max(20, width // 28), bold=True)

    for index, ((label, _, _, _), rgba) in enumerate(zip(entries, renders, strict=True)):
        x = index * width
        label_color = (
            (24, 83, 142, 255)
            if label in {"Arm B", "Arm E", "Frozen B+E"}
            else (28, 31, 35, 255)
        )
        tile = Image.fromarray(rgba, mode="RGBA")
        canvas.alpha_composite(tile, (x, 0))
        bounds = draw.textbbox((0, 0), label, font=label_font, stroke_width=2)
        text_width = bounds[2] - bounds[0]
        draw.text(
            (x + 0.5 * (width - text_width), height + 8),
            label,
            fill=label_color,
            font=label_font,
            stroke_width=2,
            stroke_fill=(255, 255, 255, 235),
        )

    output_path = output_dir / spec.output_name
    paper_path = output_path.with_name(output_path.name.replace("_transparent.png", "_paper_white.png"))
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)
    paper = Image.new("RGBA", canvas.size, (255, 255, 255, 255))
    paper.alpha_composite(canvas)
    paper.convert("RGB").save(paper_path, optimize=True)
    return {
        "sample_id": spec.sample_id,
        "emphasis": spec.title,
        "output": str(output_path),
        "paper_white_output": str(paper_path),
        "reference_image": str(mesh_root / spec.reference_image),
        "reference_aspect_ratio": spec.reference_aspect_ratio,
        "tile_size": [width, height],
        "layout": "1x8",
        "transparent_background": True,
        "paper_white_background_exported": True,
        "material": "shared procedural brown linen; identical image-space weave for every method",
        "global_content_header": False,
        "method_labels": "centered below each mesh",
        "fixed_camera_for_all_methods": True,
        "camera": camera_metadata,
        "methods": [
            {"label": label, "directory": directory, "mesh": str(path), "sha256": _sha256(path)}
            for label, directory, path, _ in entries
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=Path.home() / "results" / "2",
        help="Directory containing the eight displayed method subdirectories.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tile-size", default=720, type=int)
    args = parser.parse_args()
    if args.tile_size < 256:
        raise ValueError("--tile-size must be at least 256")

    records = [
        _render_panel(args.mesh_root, args.output_dir, spec, args.tile_size)
        for spec in SPECS
    ]
    manifest = {
        "status": "completed",
        "mesh_root": str(args.mesh_root),
        "layout": "1x8",
        "method_order": [label for label, _ in METHODS],
        "renderer": "mlr.synthetic OpenGL/EGL lit renderer; MSAA4",
        "output_color_mode": "transparent RGBA master plus flattened RGB white-background paper export",
        "material": "shared procedural brown linen; identical image-space weave for every method",
        "shared_camera_and_material_within_each_panel": True,
        "records": records,
    }
    manifest_path = args.output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "transparent_outputs": [row["output"] for row in records],
                "paper_white_outputs": [row["paper_white_output"] for row in records],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
