from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _load_preparation_script():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "prepare_future2000_synthetic_current_28view.py"
    )
    spec = importlib.util.spec_from_file_location("future2000_gt_adaptive_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _AdaptiveStub:
    @staticmethod
    def _stats(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float | int]:
        xyz = vertices[faces]
        area = 0.5 * np.linalg.norm(
            np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
        )
        represented = np.zeros(len(vertices), dtype=np.float64)
        for corner in range(3):
            np.add.at(represented, faces[:, corner], area / 3.0)
        return {
            "vertices": len(vertices),
            "faces": len(faces),
            "total_area": float(area.sum()),
            "max_represented_area": float(represented.max()),
        }

    mesh_stats = _stats

    @staticmethod
    def build_variants(vertices, faces, **_kwargs):
        midpoint = 0.5 * (vertices[0] + vertices[1])
        adaptive_vertices = np.concatenate((vertices, midpoint[None]), axis=0)
        adaptive_faces = np.asarray([[0, 3, 2], [3, 1, 2]], dtype=np.int64)
        threshold = _AdaptiveStub._stats(adaptive_vertices, adaptive_faces)[
            "max_represented_area"
        ]
        return {
            "gt": (vertices, faces),
            "gt_sub1": (adaptive_vertices, adaptive_faces),
            "gt_sub2": (adaptive_vertices, adaptive_faces),
            "gt_adaptive": (adaptive_vertices, adaptive_faces),
        }, {
            "reference": "sub2",
            "reference_max_represented_vertex_area": threshold,
            "area_scale": 1.0,
            "threshold": threshold,
            "history": [
                {"oversized_vertices": 1},
                {"oversized_vertices": 0},
            ],
        }


def test_future2000_source_is_replaced_by_audited_gt_adaptive_graph() -> None:
    module = _load_preparation_script()
    source = {
        "vertices": torch.zeros((3, 3)),
        "faces": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "gt_vertices": torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        "gt_faces": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "metadata": {"source": "fixture"},
    }

    result, audit = module.make_gt_adaptive_source(
        source,
        _AdaptiveStub,
        reference="sub2",
        area_scale=1.0,
        max_iters=12,
        max_vertices=100,
    )

    assert tuple(result["vertices"].shape) == (4, 3)
    assert torch.equal(result["vertices"], result["gt_vertices"])
    assert torch.equal(result["faces"], result["gt_faces"])
    assert result["metadata"]["query_graph_variant"] == "gt_adaptive"
    assert result["metadata"]["adaptive_reference"] == "sub2"
    assert result["metadata"]["adaptive_iterations"] == 1
    assert audit["represented_area_contract_pass"] is True
    assert audit["surface_area_abs_difference"] == 0.0
