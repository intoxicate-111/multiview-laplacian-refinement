from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mlr.learned_laplacian.multitopology_rawlap import (
    STRONG_SMOOTHING_PROFILE,
    VARIANT_NAMES,
)


def load_builder():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "prepare_future2000_multitopology_rawlap20k.py"
    )
    spec = importlib.util.spec_from_file_location("future20k_builder_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_future20k_contract_constants() -> None:
    module = load_builder()
    assert tuple(VARIANT_NAMES) == (
        "A1",
        "A2",
        "B1",
        "B2",
        "C1",
        "C2",
        "C3",
        "C4",
        "D1",
        "D2",
    )
    assert module.OBJECT_COUNT == 2000
    assert module.VARIANT_COUNT == 10
    assert module.SAMPLE_COUNT == 20000
    assert module.SAMPLE_SPLITS == {"train": 16000, "validation": 2000, "test": 2000}
    assert module.SEED_NAMESPACE == "Future2000MultiTopologyRawLap20000_v1"
    assert STRONG_SMOOTHING_PROFILE == "strong_smooth_v2"


@pytest.mark.parametrize(
    ("sample_id", "expected"),
    (("chair-id__v00", ("chair-id", 0)), ("uuid__v04", ("uuid", 4))),
)
def test_observation_object_id(sample_id: str, expected: tuple[str, int]) -> None:
    module = load_builder()
    assert module._observation_object_id(sample_id) == expected


@pytest.mark.parametrize("sample_id", ("chair", "chair__v0", "chair__vAA", "__v00"))
def test_observation_object_id_rejects_invalid_ids(sample_id: str) -> None:
    module = load_builder()
    with pytest.raises(ValueError):
        module._observation_object_id(sample_id)
