from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    )

    normalized_mesh_path = out_dir / "mesh.obj"
    save_mesh(render_mesh, normalized_mesh_path)

    image_paths: list[Path] = []
    mask_paths: list[Path] = []
    depth_paths: list[Path] = []
    for idx, camera in enumerate(cameras):
        if progress is not None:
            progress(f"  view {idx + 1}/{len(cameras)}")
        rgb, mask, depth = render_mesh_view(render_mesh, camera, config)
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
        cameras_path=cameras_path,
        image_paths=image_paths,
        mask_paths=mask_paths,
        depth_paths=depth_paths,
        mesh_path=normalized_mesh_path,
        source_mesh_path=source_mesh_path,
        out_dir=out_dir,
        config=config,
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
    raise ValueError(f"Unsupported camera trajectory: {trajectory}")


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


def render_mesh_view_opengl(
    mesh: Mesh,
    camera: Camera,
    config: SyntheticRenderConfig | None = None,
) -> tuple[Array, Array, Array]:
    try:
        import moderngl
    except ImportError as exc:
        raise RuntimeError(
            "OpenGL backend requires ModernGL. Install it with: "
            "python -m pip install -e .[gpu]"
        ) from exc

    config = config or SyntheticRenderConfig(backend="opengl")
    width, height = camera.image_size or (config.width, config.height)
    mesh.ensure_normals()

    cam_vertices = camera.world_to_camera(mesh.vertices)
    positive = cam_vertices[:, 2] > 1e-8
    if np.any(positive):
        near_z = max(1e-4, float(np.min(cam_vertices[positive, 2])) * 0.5)
        far_z = max(near_z + 1e-3, float(np.max(cam_vertices[positive, 2])) * 1.5)
    else:
        near_z, far_z = 1e-4, 10.0

    ctx = moderngl.create_standalone_context()
    color_tex = ctx.texture((width, height), 4, dtype="f4")
    depth_rb = ctx.depth_renderbuffer((width, height))
    fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_rb)
    fbo.use()
    bg = tuple(float(c) / 255.0 for c in config.background_color)
    ctx.clear(bg[0], bg[1], bg[2], 0.0, depth=1.0)
    ctx.enable(moderngl.DEPTH_TEST)

    program = ctx.program(vertex_shader=_OPENGL_VERTEX_SHADER, fragment_shader=_OPENGL_FRAGMENT_SHADER)
    packed = np.concatenate(
        [
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.normals, dtype=np.float32),
        ],
        axis=1,
    )
    vbo = ctx.buffer(packed.tobytes())
    ibo = ctx.buffer(np.asarray(mesh.faces, dtype=np.uint32).tobytes())
    vao = ctx.vertex_array(program, [(vbo, "3f 3f", "in_position", "in_normal")], index_buffer=ibo)

    program["rot0"].value = tuple(float(x) for x in camera.rotation[0])
    program["rot1"].value = tuple(float(x) for x in camera.rotation[1])
    program["rot2"].value = tuple(float(x) for x in camera.rotation[2])
    program["translation"].value = tuple(float(x) for x in camera.translation)
    program["fx"].value = float(camera.intrinsics[0, 0])
    program["fy"].value = float(camera.intrinsics[1, 1])
    program["cx"].value = float(camera.intrinsics[0, 2])
    program["cy"].value = float(camera.intrinsics[1, 2])
    program["viewport_size"].value = (float(width), float(height))
    program["near_z"].value = float(near_z)
    program["far_z"].value = float(far_z)
    program["render_mode"].value = {"lit": 0, "normal": 1, "depth": 2}[config.render_mode]
    program["object_color"].value = tuple(float(c) / 255.0 for c in config.object_color)
    light = np.asarray(config.light_direction, dtype=np.float64)
    light = light / max(np.linalg.norm(light), 1e-12)
    program["light_dir"].value = tuple(float(x) for x in light)

    vao.render(moderngl.TRIANGLES)
    data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32)
    rgba = data.reshape((height, width, 4))
    rgba = np.flipud(rgba)
    alpha_depth = rgba[:, :, 3].astype(np.float64)
    mask = alpha_depth > 0.0
    depth = np.full((height, width), np.inf, dtype=np.float64)
    depth[mask] = alpha_depth[mask]
    rgb = np.clip(rgba[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
    rgb[~mask] = np.asarray(config.background_color, dtype=np.uint8)
    return rgb, mask, depth


_OPENGL_VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_normal;

uniform vec3 rot0;
uniform vec3 rot1;
uniform vec3 rot2;
uniform vec3 translation;
uniform float fx;
uniform float fy;
uniform float cx;
uniform float cy;
uniform vec2 viewport_size;
uniform float near_z;
uniform float far_z;

out vec3 v_normal;
out float v_cam_z;

void main() {
    vec3 cam;
    cam.x = dot(rot0, in_position) + translation.x;
    cam.y = dot(rot1, in_position) + translation.y;
    cam.z = dot(rot2, in_position) + translation.z;

    float safe_z = max(cam.z, 1e-6);
    float px = fx * cam.x / safe_z + cx;
    float py = fy * cam.y / safe_z + cy;
    float x_ndc = 2.0 * px / viewport_size.x - 1.0;
    float y_ndc = 1.0 - 2.0 * py / viewport_size.y;
    float z_ndc = 2.0 * (safe_z - near_z) / max(far_z - near_z, 1e-6) - 1.0;

    gl_Position = vec4(x_ndc, y_ndc, z_ndc, 1.0);
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
    payload = []
    for camera, image_path, mask_path, depth_path in zip(
        cameras, image_paths, mask_paths, depth_paths, strict=True
    ):
        payload.append(
            {
                "name": camera.name,
                "image_path": _rel(image_path, root),
                "mask_path": _rel(mask_path, root),
                "depth_path": _rel(depth_path, root),
                "intrinsics": camera.intrinsics.tolist(),
                "rotation": camera.rotation.tolist(),
                "translation": camera.translation.tolist(),
                "image_size": list(camera.image_size) if camera.image_size else None,
                "convention": "world_to_camera_cv_z_forward_y_down",
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_dataset_json(
    path: Path,
    cameras_path: Path,
    image_paths: list[Path],
    mask_paths: list[Path],
    depth_paths: list[Path],
    mesh_path: Path,
    source_mesh_path: Path | None,
    out_dir: Path,
    config: SyntheticRenderConfig,
) -> None:
    payload = {
        "mesh_path": _rel(mesh_path, out_dir),
        "source_mesh_path": str(source_mesh_path) if source_mesh_path is not None else None,
        "cameras_path": _rel(cameras_path, out_dir),
        "image_paths": [_rel(path, out_dir) for path in image_paths],
        "mask_paths": [_rel(path, out_dir) for path in mask_paths],
        "depth_paths": [_rel(path, out_dir) for path in depth_paths],
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
            "backend": config.backend,
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
