from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/prepare_sofa50_synthetic_current_28view_1920.py"


def _module():
    spec = importlib.util.spec_from_file_location("prepare_sofa50_native1920", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scale_intrinsics_doubles_pixel_rows_and_preserves_homogeneous_row():
    module = _module()
    source = torch.tensor(
        [[480.0, 0.0, 480.0], [0.0, 480.0, 480.0], [0.0, 0.0, 1.0]]
    ).repeat(28, 1, 1)
    output = module.scale_intrinsics_to_1920(source)

    torch.testing.assert_close(output[:, :2], source[:, :2] * 2.0)
    torch.testing.assert_close(output[:, 2], source[:, 2])
    torch.testing.assert_close(source[0, 0], torch.tensor([480.0, 0.0, 480.0]))


def test_contract_hash_fields_cover_graph_targets_proxy_and_visibility():
    module = _module()
    fields = set(module.UNCHANGED_TENSOR_FIELDS)
    assert {
        "vertices",
        "faces",
        "target_positions",
        "raw_laplacian_target",
        "normalized_laplacian_target",
        "visibility_backface_and_occlusion",
        "visibility",
        "extrinsics",
    } <= fields
