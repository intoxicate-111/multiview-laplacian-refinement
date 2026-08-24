from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_sofa50_predicted_sparse_solve.py"
SPEC = importlib.util.spec_from_file_location("predicted_sparse_solve", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reported_states_match_requested_comparison() -> None:
    assert MODULE.STATES == (
        "initial",
        "frozen_adam_visibility",
        "predicted_sparse",
        "exact_sparse_oracle",
    )
