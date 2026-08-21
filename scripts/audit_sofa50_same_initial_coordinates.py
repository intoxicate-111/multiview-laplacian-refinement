#!/usr/bin/env python3
from __future__ import annotations

"""Audit prepared-mesh/RGB/camera/GT alignment without running refinement."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def resolve_sample(manifest: dict[str, Any], manifest_path: Path, sample_id: str) -> tuple[dict[str, Any], Path]:
    row = next((dict(item) for item in manifest["samples"] if item["sample_id"] == sample_id), None)
    if row is None:
        raise ValueError(f"Unknown sample ID: {sample_id}")
    path = Path(row["path"])
    return row, path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def project(vertices: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = vertices @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    valid = camera[:, 2] > 1e-8
    pixels_h = camera @ intrinsic.T
    pixels = np.full((len(vertices), 2), np.nan, dtype=np.float64)
    pixels[valid] = pixels_h[valid, :2] / pixels_h[valid, 2:3]
    return pixels, valid


def foreground_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.any(rgb != 0, axis=-1)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.max_pool2d(tensor, 2 * radius + 1, stride=1, padding=radius)[0, 0].bool().numpy()


def inside_fraction(pixels: np.ndarray, selected: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    height, width = mask.shape
    rounded = np.rint(pixels).astype(np.int64, copy=False)
    valid = selected & np.isfinite(pixels).all(axis=1)
    valid &= (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
    valid &= (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    count = int(valid.sum())
    if count == 0:
        return 0.0, 0
    return float(mask[rounded[valid, 1], rounded[valid, 0]].mean()), count


def bbox_iou(pixels: np.ndarray, selected: np.ndarray, mask: np.ndarray) -> float:
    height, width = mask.shape
    valid = selected & np.isfinite(pixels).all(axis=1)
    points = pixels[valid]
    ys, xs = np.nonzero(mask)
    if len(points) == 0 or len(xs) == 0:
        return 0.0
    a = np.asarray([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])
    b = np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
    a[[0, 2]] = np.clip(a[[0, 2]], 0, width - 1)
    a[[1, 3]] = np.clip(a[[1, 3]], 0, height - 1)
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def overlay(image: Image.Image, current: np.ndarray, gt: np.ndarray, current_sel: np.ndarray, gt_sel: np.ndarray, output: Path) -> None:
    scale = min(1.0, 960.0 / max(image.size))
    canvas = image.convert("RGB").resize((round(image.width * scale), round(image.height * scale)))
    draw = ImageDraw.Draw(canvas)
    for points, selected, color in ((gt, gt_sel, (255, 0, 255)), (current, current_sel, (0, 255, 64))):
        indices = np.flatnonzero(selected & np.isfinite(points).all(axis=1))[::4]
        for index in indices:
            x, y = points[index] * scale
            if 0 <= x < canvas.width and 0 <= y < canvas.height:
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--views", default="0,7,14,21")
    parser.add_argument("--dilation-radius", type=int, default=6)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_id = args.sample_id or str(manifest["representative_sample_id"])
    sample_row, sample_path = resolve_sample(manifest, manifest_path, sample_id)
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    current = sample["vertices"].double().numpy()
    gt = sample["gt_vertices"].double().numpy()
    intrinsics = sample["intrinsics"].double().numpy()
    extrinsics = sample["extrinsics"].double().numpy()
    visibility = sample["visibility_backface_and_occlusion"].bool().numpy()
    image_paths = [Path(path) for path in sample_row["image_paths"]]
    view_indices = [int(value) for value in args.views.split(",")]
    rows = []
    for view in view_indices:
        image = Image.open(image_paths[view]).convert("RGB")
        mask = foreground_mask(image)
        expanded = dilate(mask, args.dilation_radius)
        current_pixels, current_depth = project(current, intrinsics[view], extrinsics[view])
        gt_pixels, gt_depth = project(gt, intrinsics[view], extrinsics[view])
        current_selected = visibility[view] & current_depth
        # Current and GT share topology/vertex order in this prepared dataset.
        gt_selected = visibility[view] & gt_depth
        current_inside, current_count = inside_fraction(current_pixels, current_selected, expanded)
        gt_inside, gt_count = inside_fraction(gt_pixels, gt_selected, expanded)
        row_result = {
            "view_index": view,
            "image": str(image_paths[view]),
            "visible_current_vertices_projected": current_count,
            "visible_gt_vertices_projected": gt_count,
            "current_inside_dilated_rgb_foreground_fraction": current_inside,
            "gt_inside_dilated_rgb_foreground_fraction": gt_inside,
            "current_projection_foreground_bbox_iou": bbox_iou(current_pixels, current_selected, mask),
            "gt_projection_foreground_bbox_iou": bbox_iou(gt_pixels, gt_selected, mask),
        }
        rows.append(row_result)
        overlay(
            image,
            current_pixels,
            gt_pixels,
            current_selected,
            gt_selected,
            args.output_dir / "projection_overlays" / f"{sample_id}_view_{view:02d}.png",
        )
    passed = bool(
        min(row["gt_inside_dilated_rgb_foreground_fraction"] for row in rows) >= 0.98
        and min(row["current_inside_dilated_rgb_foreground_fraction"] for row in rows) >= 0.90
        and min(row["visible_current_vertices_projected"] for row in rows) > 0
    )
    payload = {
        "contract_audit": passed,
        "sample_id": sample_id,
        "sample_path": str(sample_path),
        "common_initial_mesh": sample_row["common_initial_mesh"],
        "coordinate_convention": "sample extrinsics are world_to_camera; p_cam=R@p_world+t; K projects positive-z camera coordinates",
        "prepared_mesh_to_gt_transform": "identity",
        "method_output_target_frame": "prepared/GT world frame",
        "rgb_foreground_definition": "non-black pixels; used only for coordinate sanity, not evaluation or tuning",
        "dilation_radius_pixels": args.dilation_radius,
        "views": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "coordinate_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contract_audit": passed, "sample_id": sample_id, "output": str(args.output_dir.resolve())}, indent=2))
    if not passed:
        raise RuntimeError("Coordinate/projection audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
