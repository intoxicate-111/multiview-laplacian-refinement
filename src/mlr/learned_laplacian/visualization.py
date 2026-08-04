from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from mlr.data import Camera, Mesh
from mlr.synthetic import SyntheticRenderConfig, render_mesh_view


def render_mesh_comparison_grid(
    entries: Sequence[tuple[str, Mesh]],
    camera: Camera,
    output_path: str | Path,
    image_size: int = 256,
    columns: int = 3,
) -> Path:
    """Render meshes with one fixed camera and save a labelled comparison grid."""

    if not entries:
        raise ValueError("entries must contain at least one labelled mesh")
    if image_size < 1 or columns < 1:
        raise ValueError("image_size and columns must be positive")
    rows = (len(entries) + columns - 1) // columns
    label_height = 24
    canvas = Image.new(
        "RGB", (columns * image_size, rows * (image_size + label_height)), (20, 20, 20)
    )
    draw = ImageDraw.Draw(canvas)
    config = SyntheticRenderConfig(
        width=image_size,
        height=image_size,
        render_mode="lit",
        normalize_mesh=False,
        backend="cpu",
    )
    for index, (label, mesh) in enumerate(entries):
        rgb, _, _ = render_mesh_view(mesh, camera, config)
        x = (index % columns) * image_size
        y = (index // columns) * (image_size + label_height)
        canvas.paste(Image.fromarray(rgb), (x, y + label_height))
        draw.text((x + 4, y + 5), label, fill=(245, 245, 245))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path
