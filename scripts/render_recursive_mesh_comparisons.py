#!/usr/bin/env python3
"""Render six-stage GT/coarse/recursive-refinement comparison images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from mlr.data import Camera
from mlr.io import load_mesh
from mlr.learned_laplacian.visualization import render_mesh_comparison_grid


LABELS = (
    "GT",
    "COARSE",
    "REFINED (ROUND 0)",
    "ITERATION 1",
    "ITERATION 2",
    "ITERATION 3",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-mesh-dir", type=Path, required=True)
    parser.add_argument("--recursive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--expected-count", type=int, default=25)
    return parser.parse_args()


def _camera(record: dict[str, Any], image_size: int) -> Camera:
    payload = record["camera"]
    intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
    source_size = int(payload["prepared_image_size"])
    scaled = intrinsics.copy() * (float(image_size) / float(source_size))
    scaled[2, 2] = 1.0
    extrinsics = np.asarray(payload["extrinsics"], dtype=np.float64)
    return Camera(
        intrinsics=scaled,
        rotation=extrinsics[:3, :3],
        translation=extrinsics[:3, 3],
        image_size=(image_size, image_size),
        name="fixed_view_0",
    )


def _overview(paths: list[tuple[str, Path]], output_path: Path) -> None:
    columns = 2
    thumb_width = 960
    title_height = 26
    margin = 12
    opened: list[tuple[str, Image.Image]] = []
    for sample_id, path in paths:
        image = Image.open(path).convert("RGB")
        thumb_height = round(image.height * thumb_width / image.width)
        opened.append((sample_id, image.resize((thumb_width, thumb_height))))

    rows = (len(opened) + columns - 1) // columns
    cell_height = max(image.height for _, image in opened) + title_height
    canvas = Image.new(
        "RGB",
        (
            columns * thumb_width + (columns + 1) * margin,
            rows * cell_height + (rows + 1) * margin,
        ),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (sample_id, image) in enumerate(opened):
        column = index % columns
        row = index // columns
        x = margin + column * (thumb_width + margin)
        y = margin + row * (cell_height + margin)
        draw.text((x + 3, y + 5), f"{index + 1:02d}  {sample_id}", fill=(245, 245, 245))
        canvas.paste(image, (x, y + title_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = _parse_args()
    base_mesh_dir = args.base_mesh_dir.resolve()
    recursive_dir = args.recursive_dir.resolve()
    manifest = json.loads(
        (base_mesh_dir / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    samples = manifest["samples"]
    if len(samples) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} samples, found {len(samples)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[str, Path]] = []
    for index, record in enumerate(samples, start=1):
        sample_id = str(record["sample_id"])
        base_paths = record["mesh_paths"]
        recursive_paths = [
            recursive_dir / f"round_{round_index:02d}" / sample_id / "predicted_refined.obj"
            for round_index in range(1, 4)
        ]
        required = [
            base_mesh_dir / base_paths["gt"],
            base_mesh_dir / base_paths["coarse"],
            base_mesh_dir / base_paths["refined"],
            *recursive_paths,
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing meshes for {sample_id}: " + ", ".join(map(str, missing))
            )

        entries = [
            (label, load_mesh(path)) for label, path in zip(LABELS, required, strict=True)
        ]
        output_path = output_dir / f"{index:02d}_{sample_id}.png"
        render_mesh_comparison_grid(
            entries,
            _camera(record, args.image_size),
            output_path,
            image_size=args.image_size,
            columns=6,
        )
        rendered.append((sample_id, output_path))
        print(f"[{index:02d}/{len(samples):02d}] {output_path.name}")

    overview_path = output_dir / "overview_25_recursive_rounds.png"
    _overview(rendered, overview_path)
    metadata = {
        "format": "mlr_recursive_mesh_comparison_images_v1",
        "policy": "recomputed_opengl_960",
        "count": len(rendered),
        "labels": list(LABELS),
        "image_size_per_panel": args.image_size,
        "base_mesh_dir": str(base_mesh_dir),
        "recursive_dir": str(recursive_dir),
        "overview": overview_path.name,
        "images": [path.name for _, path in rendered],
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Rendered {len(rendered)} six-stage comparison images to {output_dir}")
    print(f"Overview: {overview_path}")


if __name__ == "__main__":
    main()
