from __future__ import annotations

import numpy as np

from scripts.diagnose_sofa50_topology_quality import topology_row


def test_closed_tetrahedron_is_watertight() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    row, detail = topology_row("object__A1", "test", vertices, faces)
    assert row["watertight"]
    assert row["boundary_edges"] == 0
    assert row["nonmanifold_edges"] == 0
    assert row["connected_components"] == 1
    assert row["euler_characteristic"] == 2
    assert len(detail["boundary_vertices"]) == 0


def test_open_triangle_has_three_boundary_edges() -> None:
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.asarray([[0, 1, 2]])
    row, detail = topology_row("object__A1", "test", vertices, faces)
    assert not row["watertight"]
    assert row["boundary_edges"] == 3
    assert row["boundary_vertices"] == 3
    assert row["boundary_edge_ratio"] == 1.0
    assert np.array_equal(np.sort(detail["boundary_vertices"]), np.arange(3))


def test_three_faces_on_one_edge_are_nonmanifold() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.5, 0.0, 1.0]]
    )
    faces = np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    row, _ = topology_row("object__A1", "test", vertices, faces)
    assert row["nonmanifold_edges"] == 1
    assert row["nonmanifold_vertices_edge_induced"] == 2
    assert not row["watertight"]
