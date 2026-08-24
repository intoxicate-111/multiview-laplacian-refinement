from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_nds_28v_full_is_a_strict_full_view_single_step_arm() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/baselines/future2000_same_initial_full1000_blackwell.json"
        ).read_text(encoding="utf-8")
    )
    original = config["methods"]["nds"]
    full = config["methods"]["nds_28v_full"]
    assert original["arguments"]["views_per_iter"] == 1
    assert full["arguments"]["views_per_iter"] == 28
    assert full["arguments"]["iterations"] == 2000
    assert full["arguments"]["image_scale"] == 1
    assert full["arguments"]["upsample_iterations"] == [2001]
    assert full["default_visual_hull_bypassed"] is True
    assert full["view_chunking"]["enabled"] is False
    for key in (
        "lr_vertices",
        "lr_shader",
        "weight_mask",
        "weight_normal",
        "weight_laplacian",
        "weight_shading",
    ):
        assert full["arguments"][key] == original["arguments"][key]


def test_external_runner_routes_nds_28v_full_through_nds_adapter() -> None:
    path = ROOT / "scripts/evaluate_future2000_external_baseline.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "nds_28v_full" in module.METHODS
    assert "nds_28v_full" in module.NDS_METHODS
