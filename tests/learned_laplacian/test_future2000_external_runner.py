from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts/evaluate_future2000_external_baseline.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_reports_missing_runtime_once(tmp_path: Path) -> None:
    runner = _load_runner()
    args = Namespace(
        method="openmvs_refinemesh",
        interface_colmap=tmp_path / "missing_interface",
        refine_mesh=tmp_path / "missing_refine",
        external_python=None,
        external_root=tmp_path,
        exmesh_depth_root=None,
    )
    error = runner._preflight(args)
    assert error is not None
    assert "missing_interface" in error
    assert "missing_refine" in error


def test_failure_row_contains_no_metrics_or_gt_payload() -> None:
    runner = _load_runner()
    static = {
        "sample_id": "held_out__v00",
        "gt_vertices": torch.randn(3, 3),
        "target_positions": torch.randn(3, 3),
    }
    row = runner._failure_row(static, "nds", "preflight", "missing dependency")
    assert row["status"] == "failed"
    assert row["failure_reason"] == "missing dependency"
    assert row["refined_chamfer"] == ""
    assert "gt_vertices" not in row
    assert "target_positions" not in row
