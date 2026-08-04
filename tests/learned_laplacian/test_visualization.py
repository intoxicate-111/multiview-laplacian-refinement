import numpy as np

from mlr.data import Mesh
from mlr.learned_laplacian.visualization import render_mesh_comparison_grid
from mlr.synthetic import create_orbit_cameras


def test_comparison_renderer_saves_tiny_mesh_grid(tmp_path):
    mesh = Mesh(
        np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]),
        np.array([[0, 1, 2]]),
    ).ensure_normals()
    camera = create_orbit_cameras(mesh, 1, (48, 48))[0]
    output = tmp_path / "comparison.png"

    render_mesh_comparison_grid([("A", mesh), ("B", mesh)], camera, output, 48, 2)

    assert output.exists()
    assert output.stat().st_size > 0
