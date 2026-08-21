from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_exmesh_hf_zero_shot.py"
SPEC = importlib.util.spec_from_file_location("run_exmesh_hf_zero_shot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_uniform_view_selection_is_deterministic_unique_and_spans_sequence() -> None:
    selected = MODULE._select_view_indices(49, 28)
    assert selected == MODULE._select_view_indices(49, 28)
    assert len(selected) == len(set(selected)) == 28
    assert selected[0] == 0
    assert selected[-1] == 48
    assert selected == sorted(selected)


def test_all_view_selection_preserves_official_order() -> None:
    assert MODULE._select_view_indices(49, 49) == list(range(49))


def test_latest_hf_config_audit_rejects_non_raw_or_dynamic_model() -> None:
    valid = {
        "target_mode": "raw_laplacian",
        "image_encoder": {
            "feature_construction": {
                "mode": "original_plus_high_frequency",
                "kernel_size": 5,
                "sigma": 1.0,
            }
        },
        "experiment_metadata": {"views": 28},
        "confidence": {"enabled": True},
        "local_query_jitter": {"enabled": False},
        "model": {"dynamic_residual_expert": {"enabled": False}},
    }
    assert MODULE._audit_model_config(valid)["passed"] is True
    invalid = {**valid, "target_mode": "edge_scale_normalized_laplacian"}
    assert MODULE._audit_model_config(invalid)["passed"] is False
    dynamic = {**valid, "model": {"dynamic_residual_expert": {"enabled": True}}}
    assert MODULE._audit_model_config(dynamic)["passed"] is False


def test_mesh_geometry_identity_checks_vertices_faces_and_order() -> None:
    mesh = SimpleNamespace(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    same = SimpleNamespace(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())
    moved = SimpleNamespace(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())
    moved.vertices[1, 0] += 0.01
    reordered = SimpleNamespace(vertices=mesh.vertices.copy(), faces=np.asarray([[1, 0, 2]]))

    assert MODULE._same_mesh_geometry(mesh, same) is True
    assert MODULE._same_mesh_geometry(mesh, moved) is False
    assert MODULE._same_mesh_geometry(mesh, reordered) is False

    audit = MODULE._mesh_geometry_audit(mesh)
    assert audit["vertices"] == 3
    assert audit["faces"] == 1
    assert audit["bbox_diagonal"] == np.sqrt(2.0)
    assert len(audit["vertex_array_sha256"]) == 64
    assert len(audit["face_array_sha256"]) == 64
