from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CleanMeshResult:
    vertices: np.ndarray
    faces: np.ndarray
    old_to_new: np.ndarray
    new_to_old: np.ndarray
    removed_vertex_indices: np.ndarray
    removed_face_indices: np.ndarray
    degenerate_face_indices: np.ndarray
    duplicate_face_indices: np.ndarray


def remove_unreferenced_vertices(vertices: np.ndarray, faces: np.ndarray) -> CleanMeshResult:
    """Remove unused vertices and invalid duplicate/degenerate triangles deterministically."""

    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [N, 3].")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3].")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must use an integer dtype.")
    faces = faces.astype(np.int64, copy=False)
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise ValueError("faces contain an out-of-range vertex index.")

    degenerate = np.flatnonzero(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    nondegenerate_mask = np.ones(len(faces), dtype=bool)
    nondegenerate_mask[degenerate] = False
    candidate_indices = np.flatnonzero(nondegenerate_mask)
    seen: set[tuple[int, int, int]] = set()
    duplicate: list[int] = []
    retained_faces: list[int] = []
    for face_index in candidate_indices:
        key = tuple(sorted(int(value) for value in faces[face_index]))
        if key in seen:
            duplicate.append(int(face_index))
        else:
            seen.add(key)
            retained_faces.append(int(face_index))
    retained_face_indices = np.asarray(retained_faces, dtype=np.int64)
    cleaned_source_faces = faces[retained_face_indices]

    referenced = np.zeros(len(vertices), dtype=bool)
    if cleaned_source_faces.size:
        referenced[np.unique(cleaned_source_faces)] = True
    new_to_old = np.flatnonzero(referenced).astype(np.int64)
    removed_vertices = np.flatnonzero(~referenced).astype(np.int64)
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[new_to_old] = np.arange(len(new_to_old), dtype=np.int64)
    cleaned_faces = old_to_new[cleaned_source_faces]
    if cleaned_faces.size and (
        int(cleaned_faces.min()) < 0 or int(cleaned_faces.max()) >= len(new_to_old)
    ):
        raise RuntimeError("face remapping produced an invalid vertex index")
    removed_faces = np.sort(
        np.concatenate((degenerate.astype(np.int64), np.asarray(duplicate, dtype=np.int64)))
    )
    return CleanMeshResult(
        vertices=vertices[new_to_old].copy(),
        faces=cleaned_faces.astype(np.int64, copy=False),
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        removed_vertex_indices=removed_vertices,
        removed_face_indices=removed_faces,
        degenerate_face_indices=degenerate.astype(np.int64),
        duplicate_face_indices=np.asarray(duplicate, dtype=np.int64),
    )
