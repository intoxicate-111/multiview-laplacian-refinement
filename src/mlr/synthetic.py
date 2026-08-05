from __future__ import annotations

import json
import hashlib
import math
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .data import Array, Camera, Mesh, normalize_rows
from .io import load_mesh, save_mesh


@dataclass(frozen=True)
class SyntheticRenderConfig:
    num_views: int = 24
    width: int = 512
    height: int = 512
    trajectory: str = "orbit"
    radius_scale: float = 2.5
    elevation_degrees: float = 20.0
    min_elevation_degrees: float = -60.0
    max_elevation_degrees: float = 60.0
    fov_degrees: float = 50.0
    render_mode: str = "lit"
    backend: str = "cpu"
    normalize_mesh: bool = True
    background_color: tuple[int, int, int] = (0, 0, 0)
    object_color: tuple[int, int, int] = (190, 205, 220)
    light_direction: tuple[float, float, float] = (0.4, -0.6, 0.7)
    opengl_context_backend: str = "egl"
    cube_half_extent: float = 1.5
    antialiasing: str = "msaa4"
    camera_layout_version: str = "unit_sphere_cube_surface_faces6_corners8_v1"


@dataclass
class SyntheticDataset:
    image_paths: list[Path]
    mask_paths: list[Path]
    depth_paths: list[Path]
    cameras: list[Camera]
    mesh_path: Path
    dataset_path: Path


def generate_synthetic_dataset_from_mesh(
    mesh_path: str | Path,
    out_dir: str | Path,
    config: SyntheticRenderConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> SyntheticDataset:
    if progress is not None:
        progress(f"Loading mesh: {mesh_path}")
    mesh = load_mesh(mesh_path).ensure_normals()
    return generate_synthetic_dataset(
        mesh,
        out_dir,
        config=config,
        source_mesh_path=Path(mesh_path),
        progress=progress,
    )


def generate_synthetic_datasets_from_mesh_dir(
    mesh_dir: str | Path,
    out_root: str | Path,
    config: SyntheticRenderConfig | None = None,
    suffixes: tuple[str, ...] = (".obj", ".ply"),
    progress: Callable[[str], None] | None = None,
) -> list[SyntheticDataset]:
    mesh_dir = Path(mesh_dir)
    out_root = Path(out_root)
    mesh_paths = sorted(
        path for path in mesh_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes
    )
    if not mesh_paths:
        raise ValueError(f"No mesh files found in {mesh_dir} with suffixes {suffixes}.")
    datasets = []
    for mesh_index, mesh_path in enumerate(mesh_paths, start=1):
        if progress is not None:
            progress(f"[{mesh_index}/{len(mesh_paths)}] Rendering dataset for {mesh_path.name}")
        datasets.append(
            generate_synthetic_dataset_from_mesh(
                mesh_path,
                out_root / mesh_path.stem,
                config=config,
                progress=progress,
            )
        )
    return datasets


def generate_synthetic_dataset(
    mesh: Mesh,
    out_dir: str | Path,
    config: SyntheticRenderConfig | None = None,
    source_mesh_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> SyntheticDataset:
    config = config or SyntheticRenderConfig()
    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    depth_dir = out_dir / "depth"
    for stale_path in (
        image_dir, mask_dir, depth_dir, out_dir / "cameras.json",
        out_dir / "dataset.json", out_dir / "mesh.obj",
    ):
        if stale_path.is_dir():
            shutil.rmtree(stale_path)
        elif stale_path.exists():
            stale_path.unlink()
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    render_mesh = normalize_mesh_for_rendering(mesh) if config.normalize_mesh else mesh
    if progress is not None:
        progress(
            f"Preparing {config.num_views} {config.trajectory} views at {config.width}x{config.height}; "
            f"mesh has {render_mesh.num_vertices} vertices and {render_mesh.num_faces} faces"
        )
    cameras = create_synthetic_cameras(
        render_mesh,
        num_views=config.num_views,
        image_size=(config.width, config.height),
        trajectory=config.trajectory,
        radius_scale=config.radius_scale,
        elevation_degrees=config.elevation_degrees,
        min_elevation_degrees=config.min_elevation_degrees,
        max_elevation_degrees=config.max_elevation_degrees,
        fov_degrees=config.fov_degrees,
        cube_half_extent=config.cube_half_extent,
    )

    normalized_mesh_path = out_dir / "mesh.obj"
    save_mesh(render_mesh, normalized_mesh_path)

    image_paths: list[Path] = []
    mask_paths: list[Path] = []
    depth_paths: list[Path] = []
    rendered_views, actual_backend = _render_dataset_views(render_mesh, cameras, config, progress)
    for idx, (camera, rendered_view) in enumerate(zip(cameras, rendered_views, strict=True)):
        if progress is not None:
            progress(f"  view {idx + 1}/{len(cameras)}")
        rgb, mask, depth = rendered_view
        image_path = image_dir / f"{idx:04d}.png"
        mask_path = mask_dir / f"{idx:04d}.png"
        depth_path = depth_dir / f"{idx:04d}.npy"
        Image.fromarray(rgb).save(image_path)
        Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
        np.save(depth_path, depth)
        image_paths.append(image_path)
        mask_paths.append(mask_path)
        depth_paths.append(depth_path)

    cameras_path = out_dir / "cameras.json"
    dataset_path = out_dir / "dataset.json"
    _write_cameras_json(cameras_path, cameras, image_paths, mask_paths, depth_paths, out_dir)
    _write_dataset_json(
        dataset_path,
        cameras=cameras,
        cameras_path=cameras_path,
        image_paths=image_paths,
        mask_paths=mask_paths,
        depth_paths=depth_paths,
        mesh_path=normalized_mesh_path,
        source_mesh_path=source_mesh_path,
        out_dir=out_dir,
        config=config,
        actual_backend=actual_backend,
    )
    return SyntheticDataset(image_paths, mask_paths, depth_paths, cameras, normalized_mesh_path, dataset_path)


def normalize_mesh_for_rendering(mesh: Mesh) -> Mesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    scale = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    if scale < 1e-12:
        scale = 1.0
    normalized = (vertices - center[None, :]) / scale * 2.0
    new_mesh = Mesh(normalized, mesh.faces.copy())
    new_mesh.ensure_normals()
    new_mesh.attributes.update(mesh.attributes)
    new_mesh.attributes["synthetic_normalization"] = {
        "center": center.tolist(),
        "scale": float(scale),
        "formula": "(V - center) / scale * 2",
    }
    return new_mesh


def create_synthetic_cameras(
    mesh: Mesh,
    num_views: int,
    image_size: tuple[int, int],
    trajectory: str = "orbit",
    radius_scale: float = 2.5,
    elevation_degrees: float = 20.0,
    min_elevation_degrees: float = -60.0,
    max_elevation_degrees: float = 60.0,
    fov_degrees: float = 50.0,
    cube_half_extent: float = 1.5,
) -> list[Camera]:
    if trajectory == "orbit":
        return create_orbit_cameras(
            mesh,
            num_views=num_views,
            image_size=image_size,
            radius_scale=radius_scale,
            elevation_degrees=elevation_degrees,
            fov_degrees=fov_degrees,
        )
    if trajectory == "sphere":
        return create_sphere_cameras(
            mesh,
            num_views=num_views,
            image_size=image_size,
            radius_scale=radius_scale,
            min_elevation_degrees=min_elevation_degrees,
            max_elevation_degrees=max_elevation_degrees,
            fov_degrees=fov_degrees,
        )
    if trajectory == "cube_surface":
        return create_cube_surface_cameras(
            mesh,
            num_views=num_views,
            image_size=image_size,
            cube_half_extent=cube_half_extent,
            fov_degrees=fov_degrees,
        )
    raise ValueError(f"Unsupported camera trajectory: {trajectory}")


CUBE_SURFACE_VIEW_NAMES = (
    "pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z",
    "neg_x_neg_y_neg_z", "neg_x_neg_y_pos_z", "neg_x_pos_y_neg_z",
    "neg_x_pos_y_pos_z", "pos_x_neg_y_neg_z", "pos_x_neg_y_pos_z",
    "pos_x_pos_y_neg_z", "pos_x_pos_y_pos_z",
)


def create_cube_surface_cameras(
    mesh: Mesh,
    num_views: int,
    image_size: tuple[int, int],
    cube_half_extent: float = 1.5,
    fov_degrees: float = 90.0,
) -> list[Camera]:
    del mesh
    if num_views != len(CUBE_SURFACE_VIEW_NAMES):
        raise ValueError(f"cube_surface requires exactly 14 views, got {num_views}")
    if cube_half_extent <= 1.0:
        raise ValueError("cube_half_extent must be greater than the unit-sphere radius")
    width, height = image_size
    focal = 0.5 * width / math.tan(math.radians(fov_degrees) * 0.5)
    intrinsics = np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    a = float(cube_half_extent)
    centers = (
        (a, 0.0, 0.0), (-a, 0.0, 0.0), (0.0, a, 0.0), (0.0, -a, 0.0),
        (0.0, 0.0, a), (0.0, 0.0, -a),
        (-a, -a, -a), (-a, -a, a), (-a, a, -a), (-a, a, a),
        (a, -a, -a), (a, -a, a), (a, a, -a), (a, a, a),
    )
    target = np.zeros(3, dtype=np.float64)
    cameras = []
    for name, position in zip(CUBE_SURFACE_VIEW_NAMES, centers, strict=True):
        center = np.asarray(position, dtype=np.float64)
        rotation, translation = look_at_world_to_camera(center, target)
        cameras.append(Camera(intrinsics.copy(), rotation, translation, image_size, name))
    return cameras


def create_orbit_cameras(
    mesh: Mesh,
    num_views: int,
    image_size: tuple[int, int],
    radius_scale: float = 2.5,
    elevation_degrees: float = 20.0,
    fov_degrees: float = 50.0,
) -> list[Camera]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    target = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    extent = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    radius = max(1e-3, radius_scale * extent)
    elevation = math.radians(elevation_degrees)
    width, height = image_size
    focal = 0.5 * width / math.tan(math.radians(fov_degrees) * 0.5)
    intrinsics = np.array(
        [
            [focal, 0.0, width * 0.5],
            [0.0, focal, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    cameras = []
    for idx in range(num_views):
        azimuth = 2.0 * math.pi * idx / num_views
        center = target + radius * np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.sin(elevation),
                math.cos(elevation) * math.sin(azimuth),
            ],
            dtype=np.float64,
        )
        rotation, translation = look_at_world_to_camera(center, target)
        cameras.append(
            Camera(
                intrinsics=intrinsics,
                rotation=rotation,
                translation=translation,
                image_size=image_size,
                name=f"view_{idx:04d}",
            )
        )
    return cameras


def create_sphere_cameras(
    mesh: Mesh,
    num_views: int,
    image_size: tuple[int, int],
    radius_scale: float = 2.5,
    min_elevation_degrees: float = -60.0,
    max_elevation_degrees: float = 60.0,
    fov_degrees: float = 50.0,
) -> list[Camera]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    target = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    extent = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    radius = max(1e-3, radius_scale * extent)
    width, height = image_size
    focal = 0.5 * width / math.tan(math.radians(fov_degrees) * 0.5)
    intrinsics = np.array(
        [
            [focal, 0.0, width * 0.5],
            [0.0, focal, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    min_sin = math.sin(math.radians(min_elevation_degrees))
    max_sin = math.sin(math.radians(max_elevation_degrees))
    if min_sin > max_sin:
        min_sin, max_sin = max_sin, min_sin
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    cameras = []
    for idx in range(num_views):
        t = (idx + 0.5) / max(num_views, 1)
        sin_elevation = min_sin + t * (max_sin - min_sin)
        cos_elevation = math.sqrt(max(0.0, 1.0 - sin_elevation * sin_elevation))
        azimuth = idx * golden_angle
        center = target + radius * np.array(
            [
                cos_elevation * math.cos(azimuth),
                sin_elevation,
                cos_elevation * math.sin(azimuth),
            ],
            dtype=np.float64,
        )
        rotation, translation = look_at_world_to_camera(center, target)
        cameras.append(
            Camera(
                intrinsics=intrinsics,
                rotation=rotation,
                translation=translation,
                image_size=image_size,
                name=f"view_{idx:04d}",
            )
        )
    return cameras


def look_at_world_to_camera(
    camera_center: Array,
    target: Array,
    world_up: Array | None = None,
) -> tuple[Array, Array]:
    camera_center = np.asarray(camera_center, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64) if world_up is None else np.asarray(world_up)
    forward = target - camera_center
    forward = forward / max(np.linalg.norm(forward), 1e-12)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right = right / max(np.linalg.norm(right), 1e-12)
    down = np.cross(forward, right)
    down = down / max(np.linalg.norm(down), 1e-12)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ camera_center
    return rotation, translation


def render_mesh_view(
    mesh: Mesh,
    camera: Camera,
    config: SyntheticRenderConfig | None = None,
) -> tuple[Array, Array, Array]:
    config = config or SyntheticRenderConfig()
    if config.backend == "cuda":
        return render_mesh_view_cuda(mesh, camera, config)
    if config.backend == "opengl":
        return render_mesh_view_opengl(mesh, camera, config)
    if config.backend != "cpu":
        raise ValueError(f"Unsupported synthetic renderer backend: {config.backend}")
    width, height = camera.image_size or (config.width, config.height)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :] = np.asarray(config.background_color, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    depth = np.full((height, width), np.inf, dtype=np.float64)

    mesh.ensure_normals()
    pixels, z = camera.project(mesh.vertices)
    face_normals = _face_normals(mesh.vertices, mesh.faces)
    light = np.asarray(config.light_direction, dtype=np.float64)
    light = light / max(np.linalg.norm(light), 1e-12)

    for face_idx, face in enumerate(mesh.faces):
        face_z = z[face]
        if np.any(face_z <= 1e-8):
            continue
        pts = pixels[face]
        min_x = max(0, int(math.floor(np.min(pts[:, 0]))))
        max_x = min(width - 1, int(math.ceil(np.max(pts[:, 0]))))
        min_y = max(0, int(math.floor(np.min(pts[:, 1]))))
        max_y = min(height - 1, int(math.ceil(np.max(pts[:, 1]))))
        if min_x > max_x or min_y > max_y:
            continue

        normal = face_normals[face_idx]
        face_color = _shade_face(normal, light, config)
        xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        bary = _barycentric_grid(grid_x, grid_y, pts)
        if bary is None:
            continue
        inside = np.all(bary >= -1e-8, axis=2)
        if not np.any(inside):
            continue
        point_depth = np.tensordot(bary, face_z, axes=([2], [0]))
        depth_patch = depth[min_y : max_y + 1, min_x : max_x + 1]
        update = inside & (point_depth < depth_patch)
        if not np.any(update):
            continue
        depth_patch[update] = point_depth[update]
        mask[min_y : max_y + 1, min_x : max_x + 1][update] = True
        rgb[min_y : max_y + 1, min_x : max_x + 1][update] = face_color

    if config.render_mode == "depth":
        rgb = _depth_to_rgb(depth, mask)
    return rgb, mask, depth


def _render_dataset_views(
    mesh: Mesh,
    cameras: list[Camera],
    config: SyntheticRenderConfig,
    progress: Callable[[str], None] | None,
) -> tuple[list[tuple[Array, Array, Array]], str]:
    if config.backend != "opengl":
        return [render_mesh_view(mesh, camera, config) for camera in cameras], config.backend
    try:
        return render_mesh_views_opengl(mesh, cameras, config), "opengl"
    except Exception as opengl_error:  # noqa: BLE001
        warning = (
            "OpenGL/EGL production renderer unavailable; using CPU reference renderer. "
            "Current CUDA rasterizer remains disabled for production. "
            f"OpenGL/EGL error: {opengl_error}"
        )
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
        if progress is not None:
            progress(warning)
        try:
            cpu_config = SyntheticRenderConfig(**{**config.__dict__, "backend": "cpu"})
            return [render_mesh_view(mesh, camera, cpu_config) for camera in cameras], "cpu_reference"
        except Exception as cpu_error:  # noqa: BLE001
            raise RuntimeError(
                "Stable production renderer unavailable. Current CUDA rasterizer is disabled "
                "for production because it may generate rasterization artifacts. "
                f"OpenGL/EGL error: {opengl_error}; CPU reference error: {cpu_error}"
            ) from cpu_error


def render_mesh_view_cuda(
    mesh: Mesh,
    camera: Camera,
    config: SyntheticRenderConfig | None = None,
) -> tuple[Array, Array, Array]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CUDA backend requires PyTorch. Install a CUDA-enabled PyTorch build, "
            "then install this package with: python -m pip install -e .[cuda]"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requires a CUDA-enabled PyTorch install and NVIDIA GPU.")
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError("CUDA backend requires PyTorch with Tensor.scatter_reduce_ support.")

    config = config or SyntheticRenderConfig(backend="cuda")
    width, height = camera.image_size or (config.width, config.height)
    mesh.ensure_normals()

    device = torch.device("cuda")
    vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
    rotation = torch.as_tensor(camera.rotation, dtype=torch.float32, device=device)
    translation = torch.as_tensor(camera.translation, dtype=torch.float32, device=device)
    intrinsics = torch.as_tensor(camera.intrinsics, dtype=torch.float32, device=device)

    cam_vertices = vertices @ rotation.T + translation.unsqueeze(0)
    z = cam_vertices[:, 2]
    safe_z = torch.where(torch.abs(z) < 1e-12, torch.full_like(z, 1e-12), z)
    pixels_h = cam_vertices @ intrinsics.T
    pixels = pixels_h[:, :2] / safe_z.unsqueeze(1)

    tris_3d = vertices[faces]
    face_normals = torch.cross(tris_3d[:, 1] - tris_3d[:, 0], tris_3d[:, 2] - tris_3d[:, 0], dim=1)
    face_normals = face_normals / torch.linalg.norm(face_normals, dim=1, keepdim=True).clamp_min(1e-12)
    face_colors = _cuda_face_colors(face_normals, config, device)

    tri_pixels = pixels[faces]
    tri_z = z[faces]
    valid_faces = torch.all(tri_z > 1e-8, dim=1)
    valid_faces &= _cuda_triangle_area2(tri_pixels).abs() > 1e-12

    num_pixels = int(width * height)
    depth_flat = torch.full((num_pixels,), float("inf"), dtype=torch.float32, device=device)
    face_flat = torch.full((num_pixels,), -1, dtype=torch.long, device=device)
    pixel_indices = torch.arange(num_pixels, device=device, dtype=torch.long)
    ys = (pixel_indices // width).to(torch.float32) + 0.5
    xs = (pixel_indices % width).to(torch.float32) + 0.5

    max_face_pixel_pairs = 8_000_000
    chunk_size = max(1, min(int(faces.shape[0]), max_face_pixel_pairs // max(num_pixels, 1)))
    for start in range(0, int(faces.shape[0]), chunk_size):
        end = min(start + chunk_size, int(faces.shape[0]))
        chunk_valid = valid_faces[start:end]
        if not bool(torch.any(chunk_valid)):
            continue

        chunk_pixels = tri_pixels[start:end][chunk_valid]
        chunk_z = tri_z[start:end][chunk_valid]
        chunk_face_ids = torch.arange(start, end, device=device, dtype=torch.long)[chunk_valid]
        _cuda_rasterize_face_chunk(
            chunk_pixels=chunk_pixels,
            chunk_z=chunk_z,
            chunk_face_ids=chunk_face_ids,
            xs=xs,
            ys=ys,
            pixel_indices=pixel_indices,
            depth_flat=depth_flat,
            face_flat=face_flat,
        )

    mask_flat = face_flat >= 0
    rgb_flat = torch.empty((num_pixels, 3), dtype=torch.uint8, device=device)
    background = torch.as_tensor(config.background_color, dtype=torch.uint8, device=device)
    rgb_flat[:] = background
    if bool(torch.any(mask_flat)) and config.render_mode != "depth":
        rgb_flat[mask_flat] = face_colors[face_flat[mask_flat]]

    depth = depth_flat.reshape(height, width).detach().cpu().numpy().astype(np.float64)
    mask = mask_flat.reshape(height, width).detach().cpu().numpy().astype(bool)
    if config.render_mode == "depth":
        rgb = _depth_to_rgb(depth, mask)
    else:
        rgb = rgb_flat.reshape(height, width, 3).detach().cpu().numpy()
    return rgb, mask, depth


def _cuda_triangle_area2(tri_pixels):
    ab = tri_pixels[:, 1] - tri_pixels[:, 0]
    ac = tri_pixels[:, 2] - tri_pixels[:, 0]
    return ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]


def _cuda_face_colors(face_normals, config: SyntheticRenderConfig, device):
    import torch

    if config.render_mode == "normal":
        return torch.clamp((face_normals * 0.5 + 0.5) * 255.0, 0.0, 255.0).to(torch.uint8)

    object_color = torch.as_tensor(config.object_color, dtype=torch.float32, device=device)
    light = torch.as_tensor(config.light_direction, dtype=torch.float32, device=device)
    light = light / torch.linalg.norm(light).clamp_min(1e-12)
    diffuse = torch.clamp(torch.sum(face_normals * light.unsqueeze(0), dim=1), min=0.0)
    intensity = 0.25 + 0.75 * diffuse
    colors = object_color.unsqueeze(0) * intensity.unsqueeze(1)
    return torch.clamp(colors, 0.0, 255.0).to(torch.uint8)


def _cuda_rasterize_face_chunk(
    chunk_pixels,
    chunk_z,
    chunk_face_ids,
    xs,
    ys,
    pixel_indices,
    depth_flat,
    face_flat,
) -> None:
    import torch

    a = chunk_pixels[:, 0]
    b = chunk_pixels[:, 1]
    c = chunk_pixels[:, 2]
    v0 = b - a
    v1 = c - a

    d00 = torch.sum(v0 * v0, dim=1)
    d01 = torch.sum(v0 * v1, dim=1)
    d11 = torch.sum(v1 * v1, dim=1)
    denom = d00 * d11 - d01 * d01
    valid = torch.abs(denom) > 1e-12
    if not bool(torch.any(valid)):
        return

    a = a[valid]
    v0 = v0[valid]
    v1 = v1[valid]
    d00 = d00[valid]
    d01 = d01[valid]
    d11 = d11[valid]
    denom = denom[valid]
    chunk_z = chunk_z[valid]
    chunk_face_ids = chunk_face_ids[valid]

    v2x = xs.unsqueeze(0) - a[:, 0].unsqueeze(1)
    v2y = ys.unsqueeze(0) - a[:, 1].unsqueeze(1)
    d20 = v2x * v0[:, 0].unsqueeze(1) + v2y * v0[:, 1].unsqueeze(1)
    d21 = v2x * v1[:, 0].unsqueeze(1) + v2y * v1[:, 1].unsqueeze(1)

    denom = denom.unsqueeze(1)
    v = (d11.unsqueeze(1) * d20 - d01.unsqueeze(1) * d21) / denom
    w = (d00.unsqueeze(1) * d21 - d01.unsqueeze(1) * d20) / denom
    u = 1.0 - v - w
    inside = (u >= -1e-8) & (v >= -1e-8) & (w >= -1e-8)
    if not bool(torch.any(inside)):
        return

    point_depth = (
        u * chunk_z[:, 0].unsqueeze(1)
        + v * chunk_z[:, 1].unsqueeze(1)
        + w * chunk_z[:, 2].unsqueeze(1)
    )
    inside &= point_depth > 1e-8
    if not bool(torch.any(inside)):
        return

    candidate_pixels = pixel_indices.unsqueeze(0).expand(point_depth.shape)[inside]
    candidate_depths = point_depth[inside]
    candidate_faces = chunk_face_ids.unsqueeze(1).expand(point_depth.shape)[inside]

    depth_flat.scatter_reduce_(0, candidate_pixels, candidate_depths, reduce="amin", include_self=True)
    winners = candidate_depths <= (depth_flat[candidate_pixels] + 1e-6)
    if bool(torch.any(winners)):
        face_flat[candidate_pixels[winners]] = candidate_faces[winners]


def render_mesh_view_opengl(
    mesh: Mesh,
    camera: Camera,
    config: SyntheticRenderConfig | None = None,
) -> tuple[Array, Array, Array]:
    config = config or SyntheticRenderConfig(backend="opengl")
    return render_mesh_views_opengl(mesh, [camera], config)[0]


_OPENGL_RENDERERS: dict[str, "_OpenGLRenderer"] = {}


def render_mesh_views_opengl(
    mesh: Mesh,
    cameras: list[Camera],
    config: SyntheticRenderConfig | None = None,
) -> list[tuple[Array, Array, Array]]:
    config = config or SyntheticRenderConfig(backend="opengl")
    renderer = _OPENGL_RENDERERS.get(config.opengl_context_backend)
    if renderer is None:
        renderer = _OpenGLRenderer(config.opengl_context_backend)
        _OPENGL_RENDERERS[config.opengl_context_backend] = renderer
    return renderer.render_mesh(mesh, cameras, config)


class _OpenGLRenderer:
    def __init__(self, context_backend: str) -> None:
        try:
            import moderngl
        except ImportError as exc:
            raise RuntimeError(
                "OpenGL backend requires ModernGL. Install it with: python -m pip install -e .[gpu]"
            ) from exc
        self.moderngl = moderngl
        try:
            self.ctx = moderngl.create_context(
                standalone=True, backend=context_backend, require=330
            )
        except (AttributeError, TypeError):
            self.ctx = moderngl.create_standalone_context(
                backend=context_backend, require=330
            )
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.program = self.ctx.program(
            vertex_shader=_OPENGL_VERTEX_SHADER,
            fragment_shader=_OPENGL_FRAGMENT_SHADER,
        )

    def render_mesh(
        self,
        mesh: Mesh,
        cameras: list[Camera],
        config: SyntheticRenderConfig,
    ) -> list[tuple[Array, Array, Array]]:
        mesh.ensure_normals()
        packed = np.concatenate(
            [np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.normals, dtype=np.float32)],
            axis=1,
        )
        vbo = self.ctx.buffer(packed.tobytes())
        ibo = self.ctx.buffer(np.asarray(mesh.faces, dtype=np.uint32).tobytes())
        vao = self.ctx.vertex_array(
            self.program,
            [(vbo, "3f 3f", "in_position", "in_normal")],
            index_buffer=ibo,
        )
        try:
            return [
                self._render_camera(vao, camera, config, mesh.vertices)
                for camera in cameras
            ]
        finally:
            vao.release()
            ibo.release()
            vbo.release()

    def _render_camera(
        self,
        vao: Any,
        camera: Camera,
        config: SyntheticRenderConfig,
        vertices: Array,
    ) -> tuple[Array, Array, Array]:
        width, height = camera.image_size or (config.width, config.height)
        camera_vertices = camera.world_to_camera(vertices)
        positive_z = camera_vertices[:, 2][camera_vertices[:, 2] > 1e-8]
        if positive_z.size:
            near_z = max(1e-4, float(positive_z.min()) * 0.5)
            far_z = max(near_z + 1e-3, float(positive_z.max()) * 1.5)
        else:
            near_z, far_z = 1e-4, 10.0
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = camera.rotation
        view[:3, 3] = camera.translation
        projection = _cv_projection_matrix(camera.intrinsics, width, height, near_z, far_z)
        self.program["view_matrix"].write(view.T.astype("f4").tobytes())
        self.program["projection_matrix"].write(projection.T.astype("f4").tobytes())
        self.program["near_z"].value = near_z
        self.program["far_z"].value = far_z
        self.program["render_mode"].value = {"lit": 0, "normal": 1, "depth": 2}[config.render_mode]
        self.program["object_color"].value = tuple(float(c) / 255.0 for c in config.object_color)
        light = np.asarray(config.light_direction, dtype=np.float64)
        light /= max(np.linalg.norm(light), 1e-12)
        self.program["light_dir"].value = tuple(float(x) for x in light)

        color_tex = self.ctx.texture((width, height), 4, dtype="f4")
        depth_rb = self.ctx.depth_renderbuffer((width, height))
        fbo = self.ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_rb)
        try:
            self._draw(vao, fbo, config)
            rgba = np.frombuffer(color_tex.read(alignment=1), dtype=np.float32).reshape(height, width, 4)
            rgba = np.flipud(rgba)
            camera_z = rgba[:, :, 3].astype(np.float64)
            mask = camera_z > 0.0
            depth = np.full((height, width), np.inf, dtype=np.float64)
            depth[mask] = camera_z[mask]
            if config.antialiasing == "msaa4" and config.render_mode != "depth":
                rgb = self._render_msaa_rgb(vao, width, height, config)
            else:
                rgb = np.clip(rgba[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
            rgb[~mask] = np.asarray(config.background_color, dtype=np.uint8)
            return rgb, mask, depth
        finally:
            fbo.release()
            depth_rb.release()
            color_tex.release()

    def _draw(self, vao: Any, fbo: Any, config: SyntheticRenderConfig) -> None:
        fbo.use()
        bg = tuple(float(c) / 255.0 for c in config.background_color)
        self.ctx.clear(bg[0], bg[1], bg[2], 0.0, depth=1.0)
        vao.render(self.moderngl.TRIANGLES)

    def _render_msaa_rgb(
        self, vao: Any, width: int, height: int, config: SyntheticRenderConfig
    ) -> Array:
        color_msaa = self.ctx.renderbuffer((width, height), components=4, samples=4)
        depth_msaa = self.ctx.depth_renderbuffer((width, height), samples=4)
        msaa_fbo = self.ctx.framebuffer(color_attachments=[color_msaa], depth_attachment=depth_msaa)
        resolved_tex = self.ctx.texture((width, height), 4, dtype="f1")
        resolved_fbo = self.ctx.framebuffer(color_attachments=[resolved_tex])
        try:
            self._draw(vao, msaa_fbo, config)
            self.ctx.copy_framebuffer(resolved_fbo, msaa_fbo)
            rgba = np.frombuffer(resolved_tex.read(alignment=1), dtype=np.uint8).reshape(height, width, 4)
            return np.flipud(rgba[:, :, :3]).copy()
        finally:
            resolved_fbo.release()
            resolved_tex.release()
            msaa_fbo.release()
            depth_msaa.release()
            color_msaa.release()


def _cv_projection_matrix(
    intrinsics: Array,
    width: int,
    height: int,
    near_z: float,
    far_z: float,
) -> Array:
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    return np.array(
        [
            [2.0 * fx / width, 0.0, 2.0 * cx / width - 1.0, 0.0],
            [0.0, -2.0 * fy / height, 1.0 - 2.0 * cy / height, 0.0],
            [0.0, 0.0, (far_z + near_z) / (far_z - near_z), -2.0 * far_z * near_z / (far_z - near_z)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


_OPENGL_VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_normal;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

out vec3 v_normal;
out float v_cam_z;

void main() {
    vec4 cam = view_matrix * vec4(in_position, 1.0);
    gl_Position = projection_matrix * cam;
    v_normal = normalize(in_normal);
    v_cam_z = cam.z;
}
"""


_OPENGL_FRAGMENT_SHADER = """
#version 330
in vec3 v_normal;
in float v_cam_z;

uniform int render_mode;
uniform vec3 object_color;
uniform vec3 light_dir;
uniform float near_z;
uniform float far_z;

out vec4 frag_color;

void main() {
    if (v_cam_z <= 0.0) {
        discard;
    }

    vec3 n = normalize(v_normal);
    vec3 color;
    if (render_mode == 1) {
        color = n * 0.5 + 0.5;
    } else if (render_mode == 2) {
        float d = clamp((v_cam_z - near_z) / max(far_z - near_z, 1e-6), 0.0, 1.0);
        color = vec3(1.0 - d);
    } else {
        float diffuse = max(dot(n, normalize(light_dir)), 0.0);
        float intensity = 0.25 + 0.75 * diffuse;
        color = object_color * intensity;
    }
    frag_color = vec4(color, v_cam_z);
}
"""


def _shade_face(normal: Array, light: Array, config: SyntheticRenderConfig) -> Array:
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    if config.render_mode == "normal":
        return np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    diffuse = max(0.0, float(np.dot(normal, light)))
    intensity = 0.25 + 0.75 * diffuse
    color = np.asarray(config.object_color, dtype=np.float64) * intensity
    return np.clip(color, 0, 255).astype(np.uint8)


def _face_normals(vertices: Array, faces: Array) -> Array:
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return normalize_rows(normals)


def _barycentric(point: Array, triangle: Array) -> Array | None:
    a, b, c = triangle
    v0 = b - a
    v1 = c - a
    v2 = point - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float64)


def _barycentric_grid(grid_x: Array, grid_y: Array, triangle: Array) -> Array | None:
    a, b, c = triangle
    v0 = b - a
    v1 = c - a
    v2x = grid_x - a[0]
    v2y = grid_y - a[1]
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = v2x * v0[0] + v2y * v0[1]
    d21 = v2x * v1[0] + v2y * v1[1]
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.stack([u, v, w], axis=2)


def _depth_to_rgb(depth: Array, mask: Array) -> Array:
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(mask):
        return rgb
    values = depth[mask]
    near = float(np.min(values))
    far = float(np.max(values))
    normalized = (values - near) / max(far - near, 1e-12)
    gray = np.clip((1.0 - normalized) * 255.0, 0, 255).astype(np.uint8)
    rgb[mask] = np.stack([gray, gray, gray], axis=1)
    return rgb


def _write_cameras_json(
    path: Path,
    cameras: list[Camera],
    image_paths: list[Path],
    mask_paths: list[Path],
    depth_paths: list[Path],
    root: Path,
) -> None:
    payload = _camera_records(cameras, image_paths, mask_paths, depth_paths, root)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _camera_records(
    cameras: list[Camera],
    image_paths: list[Path],
    mask_paths: list[Path],
    depth_paths: list[Path],
    root: Path,
) -> list[dict[str, Any]]:
    payload = []
    for camera, image_path, mask_path, depth_path in zip(
        cameras, image_paths, mask_paths, depth_paths, strict=True
    ):
        extrinsics = np.eye(4, dtype=np.float64)
        extrinsics[:3, :3] = camera.rotation
        extrinsics[:3, 3] = camera.translation
        payload.append(
            {
                "name": camera.name,
                "image_path": _rel(image_path, root),
                "mask_path": _rel(mask_path, root),
                "depth_path": _rel(depth_path, root),
                "intrinsics": camera.intrinsics.tolist(),
                "extrinsics": extrinsics.tolist(),
                "rotation": camera.rotation.tolist(),
                "translation": camera.translation.tolist(),
                "image_size": list(camera.image_size) if camera.image_size else None,
                "convention": "world_to_camera_cv_z_forward_y_down",
            }
        )
    return payload


def _write_dataset_json(
    path: Path,
    cameras: list[Camera],
    cameras_path: Path,
    image_paths: list[Path],
    mask_paths: list[Path],
    depth_paths: list[Path],
    mesh_path: Path,
    source_mesh_path: Path | None,
    out_dir: Path,
    config: SyntheticRenderConfig,
    actual_backend: str,
) -> None:
    payload = {
        "mesh_path": _rel(mesh_path, out_dir),
        "source_mesh_path": str(source_mesh_path) if source_mesh_path is not None else None,
        "cameras_path": _rel(cameras_path, out_dir),
        "image_paths": [_rel(path, out_dir) for path in image_paths],
        "mask_paths": [_rel(path, out_dir) for path in mask_paths],
        "depth_paths": [_rel(path, out_dir) for path in depth_paths],
        "cameras": _camera_records(
            cameras, image_paths, mask_paths, depth_paths, out_dir
        ),
        "config": {
            "num_views": config.num_views,
            "width": config.width,
            "height": config.height,
            "trajectory": config.trajectory,
            "radius_scale": config.radius_scale,
            "elevation_degrees": config.elevation_degrees,
            "min_elevation_degrees": config.min_elevation_degrees,
            "max_elevation_degrees": config.max_elevation_degrees,
            "fov_degrees": config.fov_degrees,
            "render_mode": config.render_mode,
            "backend": actual_backend,
            "requested_backend": config.backend,
            "opengl_context_backend": config.opengl_context_backend,
            "cube_half_extent": config.cube_half_extent,
            "antialiasing": config.antialiasing,
            "camera_layout_version": config.camera_layout_version,
            "normalized_mesh_checksum": _mesh_checksum(load_mesh(mesh_path)),
            "normalize_mesh": config.normalize_mesh,
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _mesh_checksum(mesh: Mesh) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.asarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()
