#!/usr/bin/env python3
from __future__ import annotations

"""Render MP4 videos for the Sofa50 three-round recursive-refinement experiment."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mlr.data import Camera, Mesh
from mlr.io import load_mesh
from mlr.synthetic import SyntheticRenderConfig, render_mesh_view


STAGES = ("COARSE", "REFINED (ROUND 0)", "ITERATION 1", "ITERATION 2", "ITERATION 3")
COLORS = ((115, 151, 255), (78, 205, 142), (255, 193, 92), (255, 137, 92), (242, 92, 92))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-mesh-dir", required=True, type=Path)
    parser.add_argument("--recursive-dir", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--aggregate-csv", required=True, type=Path)
    parser.add_argument("--comparison-image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold-seconds", type=float, default=0.8)
    parser.add_argument("--transition-seconds", type=float, default=1.2)
    return parser.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def writer(path: Path, size: tuple[int, int], fps: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not result.isOpened():
        raise RuntimeError(f"Could not open MP4 writer: {path}")
    return result


def write_frame(output: cv2.VideoWriter, image: Image.Image) -> None:
    output.write(cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR))


def camera(record: dict[str, Any], size: int) -> Camera:
    payload = record["camera"]
    intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
    intrinsics *= float(size) / float(payload["prepared_image_size"])
    intrinsics[2, 2] = 1.0
    extrinsics = np.asarray(payload["extrinsics"], dtype=np.float64)
    return Camera(
        intrinsics=intrinsics,
        rotation=extrinsics[:3, :3],
        translation=extrinsics[:3, 3],
        image_size=(size, size),
        name="fixed_view_0",
    )


def render(mesh: Mesh, view: Camera, size: int) -> Image.Image:
    config = SyntheticRenderConfig(
        width=size,
        height=size,
        render_mode="lit",
        backend="opengl",
        normalize_mesh=False,
        background_color=(20, 20, 20),
        opengl_context_backend="egl",
        antialiasing="msaa4",
    )
    rgb, _, _ = render_mesh_view(mesh.ensure_normals(), view, config)
    return Image.fromarray(rgb)


def sample_metrics(path: Path, sample_id: str) -> tuple[float, list[float]]:
    rows = [
        row
        for row in csv.DictReader(path.open(encoding="utf-8"))
        if row["policy"] == "recomputed_opengl_960" and row["sample_id"] == sample_id
    ]
    by_round = {int(row["round"]): row for row in rows}
    if set(by_round) != {0, 1, 2, 3}:
        raise ValueError(f"Incomplete recomputed metrics for {sample_id}")
    coarse = float(by_round[0]["original_initial_chamfer"])
    refined = [float(by_round[index]["reconstruction_chamfer"]) for index in range(4)]
    return coarse, [coarse, *refined]


def choose_sample(metrics_path: Path) -> str:
    rows = [
        row
        for row in csv.DictReader(metrics_path.open(encoding="utf-8"))
        if row["policy"] == "recomputed_opengl_960"
    ]
    grouped: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["sample_id"], {})[int(row["round"])] = row
    candidates = []
    for sample_id, values in grouped.items():
        if set(values) != {0, 1, 2, 3}:
            continue
        coarse = float(values[0]["original_initial_chamfer"])
        round0 = float(values[0]["reconstruction_chamfer"])
        round3 = float(values[3]["reconstruction_chamfer"])
        if round0 < coarse:
            candidates.append((round3 - round0, sample_id))
    if not candidates:
        raise ValueError("No sample improves at round 0")
    return max(candidates)[1]


def timeline(draw: ImageDraw.ImageDraw, active: float, values: list[float]) -> None:
    x0, x1, y = 105, 1175, 650
    draw.line((x0, y, x1, y), fill=(100, 105, 115), width=4)
    xs = np.linspace(x0, x1, len(STAGES))
    for index, (x, label, color, value) in enumerate(zip(xs, STAGES, COLORS, values, strict=True)):
        reached = active >= index - 1e-6
        radius = 12 if abs(active - index) < 0.12 else 9
        fill = color if reached else (70, 73, 80)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
        draw.text((x, y + 18), label, font=font(14), fill=(235, 235, 235), anchor="ma")
        draw.text((x, y - 20), f"{value:.6f}", font=font(13), fill=fill, anchor="ms")


def representative_video(
    base_dir: Path,
    recursive_dir: Path,
    metrics_path: Path,
    output_path: Path,
    sample_id: str,
    fps: int,
    hold_seconds: float,
    transition_seconds: float,
) -> None:
    manifest = json.loads((base_dir / "comparison_manifest.json").read_text(encoding="utf-8"))
    record = next(item for item in manifest["samples"] if item["sample_id"] == sample_id)
    paths = [
        base_dir / record["mesh_paths"]["coarse"],
        base_dir / record["mesh_paths"]["refined"],
        *[
            recursive_dir / f"round_{index:02d}" / sample_id / "predicted_refined.obj"
            for index in range(1, 4)
        ],
    ]
    stages = [load_mesh(path) for path in paths]
    gt = load_mesh(base_dir / record["mesh_paths"]["gt"])
    faces = stages[0].faces
    for mesh in stages[1:]:
        if not np.array_equal(mesh.faces, faces) or mesh.vertices.shape != stages[0].vertices.shape:
            raise ValueError(f"Topology changed for {sample_id}")
    coarse_value, metrics = sample_metrics(metrics_path, sample_id)
    size = 480
    view = camera(record, size)
    gt_image = render(gt, view, size)
    hold_frames = max(1, round(hold_seconds * fps))
    transition_frames = max(2, round(transition_seconds * fps))
    output = writer(output_path, (1280, 720), fps)
    try:
        segments: list[tuple[int, int, float]] = []
        for stage_index in range(len(stages)):
            segments.extend((stage_index, stage_index, 0.0) for _ in range(hold_frames))
            if stage_index + 1 < len(stages):
                for step in range(1, transition_frames + 1):
                    t = step / transition_frames
                    smooth = t * t * (3.0 - 2.0 * t)
                    segments.append((stage_index, stage_index + 1, smooth))
        for frame_index, (left, right, alpha) in enumerate(segments):
            vertices = (1.0 - alpha) * stages[left].vertices + alpha * stages[right].vertices
            current = Mesh(vertices, faces.copy()).ensure_normals()
            current_image = render(current, view, size)
            canvas = Image.new("RGB", (1280, 720), (14, 16, 20))
            canvas.paste(gt_image, (100, 115))
            canvas.paste(current_image, (700, 115))
            draw = ImageDraw.Draw(canvas)
            draw.text((40, 18), "Recursive Mesh Refinement", font=font(30), fill=(245, 245, 245))
            draw.text((40, 58), sample_id, font=font(15), fill=(165, 170, 180))
            draw.text((340, 105), "GROUND TRUTH", font=font(18), fill=(235, 235, 235), anchor="ms")
            stage_position = left + alpha
            label = STAGES[left] if left == right else f"{STAGES[left]}  ->  {STAGES[right]}"
            color = tuple(int((1 - alpha) * COLORS[left][i] + alpha * COLORS[right][i]) for i in range(3))
            draw.text((940, 105), label, font=font(18), fill=color, anchor="ms")
            endpoint = min(round(stage_position), len(metrics) - 1)
            delta = metrics[endpoint] - coarse_value
            status = "better than coarse" if delta < 0 else "worse than coarse"
            draw.text(
                (940, 618),
                f"Recorded Chamfer: {metrics[endpoint]:.6f}  ({delta:+.6f}, {status})",
                font=font(15),
                fill=(120, 220, 155) if delta < 0 else (245, 135, 120),
                anchor="ms",
            )
            timeline(draw, stage_position, metrics)
            write_frame(output, canvas)
            if frame_index % fps == 0:
                print(f"representative frame {frame_index + 1}/{len(segments)}", flush=True)
    finally:
        output.release()


def aggregate_metrics(path: Path) -> list[tuple[float, int]]:
    rows = [
        row
        for row in csv.DictReader(path.open(encoding="utf-8"))
        if row["policy"] == "recomputed_opengl_960"
    ]
    by_round = {int(row["round"]): row for row in rows}
    coarse = float(by_round[0]["original_initial_chamfer"])
    return [(coarse, 0)] + [
        (float(by_round[index]["reconstruction_chamfer"]), int(by_round[index]["cumulative_improved_over_original"]))
        for index in range(4)
    ]


def overview_video(image_dir: Path, aggregate_path: Path, output_path: Path, fps: int) -> None:
    manifest = json.loads((image_dir / "render_manifest.json").read_text(encoding="utf-8"))
    images = [Image.open(image_dir / name).convert("RGB") for name in manifest["images"]]
    metrics = aggregate_metrics(aggregate_path)
    if len(images) != 25:
        raise ValueError(f"Expected 25 comparison images, found {len(images)}")
    stages: list[Image.Image] = []
    for stage_index, label in enumerate(("COARSE", "ROUND 0", "ITERATION 1", "ITERATION 2", "ITERATION 3")):
        canvas = Image.new("RGB", (1280, 1080), (14, 16, 20))
        draw = ImageDraw.Draw(canvas)
        chamfer, improved = metrics[stage_index]
        draw.text((40, 25), "25-Sample Recursive Refinement Overview", font=font(30), fill=(245, 245, 245))
        draw.text((40, 68), label, font=font(23), fill=COLORS[stage_index])
        draw.text(
            (1240, 45),
            f"Mean Chamfer {chamfer:.6f}    Improved {improved}/25" if stage_index else f"Mean Chamfer {chamfer:.6f}",
            font=font(18),
            fill=(225, 225, 225),
            anchor="ra",
        )
        thumb = 180
        gap = 14
        grid_width = 5 * thumb + 4 * gap
        grid_x = (1280 - grid_width) // 2
        grid_y = 115
        source_panel = stage_index + 1
        for index, source in enumerate(images):
            panel_width = source.width // 6
            crop = source.crop((source_panel * panel_width, 24, (source_panel + 1) * panel_width, 24 + panel_width))
            crop = crop.resize((thumb, thumb), Image.Resampling.LANCZOS)
            x = grid_x + (index % 5) * (thumb + gap)
            y = grid_y + (index // 5) * (thumb + gap)
            canvas.paste(crop, (x, y))
            draw.text((x + 5, y + 5), f"{index + 1:02d}", font=font(13), fill=(250, 250, 250))
        stages.append(canvas)
    output = writer(output_path, (1280, 1080), fps)
    hold = fps
    transition = fps // 2
    try:
        for index, image in enumerate(stages):
            for _ in range(hold):
                write_frame(output, image)
            if index + 1 < len(stages):
                for step in range(1, transition + 1):
                    write_frame(output, Image.blend(image, stages[index + 1], step / transition))
    finally:
        output.release()


def main() -> None:
    args = parse_args()
    base_dir = args.base_mesh_dir.resolve()
    recursive_dir = args.recursive_dir.resolve()
    output_dir = args.output_dir.resolve()
    sample_id = args.sample_id or choose_sample(args.metrics_csv.resolve())
    representative = output_dir / f"recursive_refinement_{sample_id}.mp4"
    overview = output_dir / "recursive_refinement_overview_25.mp4"
    representative_video(
        base_dir,
        recursive_dir,
        args.metrics_csv.resolve(),
        representative,
        sample_id,
        args.fps,
        args.hold_seconds,
        args.transition_seconds,
    )
    overview_video(
        args.comparison_image_dir.resolve(),
        args.aggregate_csv.resolve(),
        overview,
        args.fps,
    )
    metadata = {
        "format": "mlr_recursive_refinement_video_v1",
        "representative_sample": sample_id,
        "representative_video": representative.name,
        "overview_video": overview.name,
        "fps": args.fps,
        "stages": list(STAGES),
        "selection": "largest round3-minus-round0 degradation among samples improved at round0",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "video_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
