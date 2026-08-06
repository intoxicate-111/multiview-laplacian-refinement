from __future__ import annotations

import numpy as np
import pytest

from mlr.data import Camera, Mesh
from mlr.learned_laplacian.renderer_visibility import (
    compute_renderer_visibility,
    union_of_visible_faces,
    vertex_visibility_from_face_id_buffer,
)
from mlr.synthetic import SyntheticRenderConfig, render_mesh_face_ids


def _camera(size: int = 64) -> Camera:
    focal = size / 2.0
    intrinsics = np.array(
        [[focal, 0.0, (size - 1) / 2.0], [0.0, focal, (size - 1) / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )
    return Camera(intrinsics, np.eye(3), np.zeros(3), (size, size))


def _config(backend: str, *, culling: bool = False, size: int = 64):
    return SyntheticRenderConfig(
        width=size,
        height=size,
        backend=backend,
        normalize_mesh=False,
        antialiasing="none",
        backface_culling=culling,
        front_face_winding="ccw",
    )


@pytest.mark.parametrize("backend", ["cpu", "opengl"])
def test_back_facing_triangle_is_removed_by_renderer_culling(backend):
    vertices = np.array([[-0.5, -0.5, 1.0], [0.0, 0.5, 1.0], [0.5, -0.5, 1.0]])
    camera = _camera()
    front = Mesh(vertices, np.array([[0, 1, 2]])).ensure_normals()
    back = Mesh(vertices, np.array([[0, 2, 1]])).ensure_normals()
    try:
        front_two_sided = render_mesh_face_ids(front, camera, _config(backend))
        back_two_sided = render_mesh_face_ids(back, camera, _config(backend))
        front_culled = render_mesh_face_ids(front, camera, _config(backend, culling=True))
        back_culled = render_mesh_face_ids(back, camera, _config(backend, culling=True))
    except Exception as exc:  # pragma: no cover - EGL availability is environment specific
        if backend == "opengl":
            pytest.skip(f"OpenGL unavailable: {exc}")
        raise
    assert np.any(front_two_sided >= 0)
    assert np.any(back_two_sided >= 0)
    assert np.any(front_culled >= 0)
    assert not np.any(back_culled >= 0)


def test_two_layer_occlusion_rejects_rear_vertices_without_reading_depth_map():
    vertices = np.array(
        [
            [-0.5, -0.5, 1.0], [0.0, 0.5, 1.0], [0.5, -0.5, 1.0],
            [-1.0, -1.0, 2.0], [0.0, 1.0, 2.0], [1.0, -1.0, 2.0],
        ]
    )
    mesh = Mesh(vertices, np.array([[0, 1, 2], [3, 4, 5]])).ensure_normals()
    camera = _camera()
    result = compute_renderer_visibility(
        mesh, [camera], _config("cpu"), neighborhood_radius=1
    )
    assert result.frustum_valid.all()
    assert result.occlusion_visible[0, :3].all()
    assert not result.occlusion_visible[0, 3:].any()


def test_projected_pixel_membership_is_stricter_than_visible_face_union():
    vertices = np.array(
        [
            [-0.6, -0.6, 1.0], [0.0, 0.6, 1.0], [0.6, -0.6, 1.0],
            [0.0, 0.0, 2.0], [-1.6, -1.2, 2.0], [1.6, -1.2, 2.0],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    mesh = Mesh(vertices, faces).ensure_normals()
    camera = _camera()
    face_ids = render_mesh_face_ids(mesh, camera, _config("cpu"))
    projected = vertex_visibility_from_face_id_buffer(
        vertices, faces, camera, face_ids, neighborhood_radius=1
    )
    union = union_of_visible_faces(face_ids, faces)
    assert union[3:].all()
    assert not projected[3]
    assert projected[4:].all()


def test_three_by_three_neighborhood_recovers_silhouette_and_subpixel_vertices():
    vertices = np.array(
        [[-0.04, -0.04, 1.0], [0.0, 0.05, 1.0], [0.04, -0.04, 1.0]]
    )
    faces = np.array([[0, 1, 2]])
    mesh = Mesh(vertices, faces).ensure_normals()
    camera = _camera(32)
    ids = render_mesh_face_ids(mesh, camera, _config("cpu", size=32))
    exact = vertex_visibility_from_face_id_buffer(
        vertices, faces, camera, ids, neighborhood_radius=0
    )
    neighborhood = vertex_visibility_from_face_id_buffer(
        vertices, faces, camera, ids, neighborhood_radius=1
    )
    assert neighborhood.sum() >= exact.sum()
    assert neighborhood.any()


def test_cpu_and_opengl_agree_on_vertex_visibility_when_available():
    vertices = np.array(
        [
            [-0.5, -0.5, 1.0], [0.0, 0.5, 1.0], [0.5, -0.5, 1.0],
            [-1.0, -1.0, 2.0], [0.0, 1.0, 2.0], [1.0, -1.0, 2.0],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    mesh = Mesh(vertices, faces).ensure_normals()
    camera = _camera()
    cpu = render_mesh_face_ids(mesh, camera, _config("cpu"))
    try:
        opengl = render_mesh_face_ids(mesh, camera, _config("opengl"))
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"OpenGL unavailable: {exc}")
    cpu_visibility = vertex_visibility_from_face_id_buffer(
        vertices, faces, camera, cpu, neighborhood_radius=1
    )
    opengl_visibility = vertex_visibility_from_face_id_buffer(
        vertices, faces, camera, opengl, neighborhood_radius=1
    )
    np.testing.assert_array_equal(cpu_visibility, opengl_visibility)
    assert np.array_equal(np.unique(cpu[cpu >= 0]), np.unique(opengl[opengl >= 0]))
