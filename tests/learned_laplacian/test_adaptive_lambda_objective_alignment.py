from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from diagnose_sofa50_adaptive_lambda_objective_alignment import (  # noqa: E402
    _interpretation,
    _select,
)


def test_tolerance_tie_prefers_lambda_closest_to_fixed() -> None:
    rows = [
        {"lambda": 0.003, "chamfer": 0.00100000},
        {"lambda": 0.01, "chamfer": 0.00100005},
        {"lambda": 0.03, "chamfer": 0.00100006},
    ]
    selected, ties, tolerance = _select(rows, "chamfer")
    assert tolerance == pytest.approx(1e-7)
    assert ties == [0.003, 0.01, 0.03]
    assert selected["lambda"] == 0.01


def test_validation_interpretation_thresholds_are_predeclared() -> None:
    agreement = {
        "cd_vrms_exact_percentage": 0.8,
        "cd_vrms_within_one_grid_step": 46,
        "samples": 50,
    }
    cross = {
        "cd_oracle_vrms_minus_fixed_relative": 0.005,
        "vrms_oracle_cd_minus_fixed_relative": -0.01,
    }
    assert _interpretation(agreement, cross) == ("Case A", "surface-fidelity aligned")
    agreement["cd_vrms_exact_percentage"] = 0.2
    cross["cd_oracle_vrms_minus_fixed_relative"] = 0.03
    cross["vrms_oracle_cd_minus_fixed_relative"] = 0.04
    assert _interpretation(agreement, cross) == ("Case C", "strong objective mismatch")
