from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_sofa50_gt_query.py"
SPEC = importlib.util.spec_from_file_location("prepare_sofa50_gt_query", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selected_split_ids_preserves_order_and_limits() -> None:
    selection = {
        "splits": {
            "train": ["a", "b", "c"],
            "validation": ["d", "e"],
            "test": ["f"],
        }
    }
    result = MODULE._selected_split_ids(
        selection, {"train": 2, "validation": 1, "test": 0}
    )
    assert result == {"train": ["a", "b"], "validation": ["d"], "test": []}


def test_selected_split_ids_rejects_leakage() -> None:
    selection = {
        "splits": {
            "train": ["same"],
            "validation": ["same"],
            "test": ["other"],
        }
    }
    with pytest.raises(ValueError, match="not disjoint"):
        MODULE._selected_split_ids(
            selection, {"train": None, "validation": None, "test": None}
        )
