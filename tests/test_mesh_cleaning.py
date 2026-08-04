import numpy as np

from mlr.mesh_cleaning import remove_unreferenced_vertices


def test_remove_unreferenced_vertices_preserves_triangle_geometry_and_mappings():
    vertices = np.array(
        [[9.0, 9.0, 9.0], [0.0, 0.0, 0.0], [8.0, 8.0, 8.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.array([[1, 3, 4]])
    result = remove_unreferenced_vertices(vertices, faces)

    np.testing.assert_array_equal(result.new_to_old, [1, 3, 4])
    np.testing.assert_array_equal(result.old_to_new, [-1, 0, -1, 1, 2])
    np.testing.assert_array_equal(result.removed_vertex_indices, [0, 2])
    np.testing.assert_array_equal(result.faces, [[0, 1, 2]])
    np.testing.assert_array_equal(result.vertices[result.faces], vertices[faces])


def test_cleaning_removes_degenerate_and_duplicate_faces_deterministically():
    vertices = np.eye(3)
    faces = np.array([[0, 1, 2], [2, 1, 0], [0, 0, 1]])
    result = remove_unreferenced_vertices(vertices, faces)

    np.testing.assert_array_equal(result.faces, [[0, 1, 2]])
    np.testing.assert_array_equal(result.duplicate_face_indices, [1])
    np.testing.assert_array_equal(result.degenerate_face_indices, [2])
    np.testing.assert_array_equal(result.removed_face_indices, [1, 2])
