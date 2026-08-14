from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / "scripts/prepare_future2000_exmesh_da3_priors.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDataset:
    sample_ids = tuple(
        f"object-{object_index:03d}__v{variant:02d}"
        for object_index in range(4)
        for variant in range(5)
    )


class SharedObservationDataset(FakeDataset):
    def load_static(self, index: int):
        import numpy as np

        return {
            "image_paths": ["a.png", "b.png"],
            "intrinsics": np.eye(3)[None].repeat(2, axis=0),
            "extrinsics": np.eye(4)[None].repeat(2, axis=0),
        }


def test_da3_representatives_choose_one_of_each_five_variants() -> None:
    module = _load_module()
    representatives = module.representative_indices(FakeDataset())
    assert representatives == [
        ("object-000", 0),
        ("object-001", 5),
        ("object-002", 10),
        ("object-003", 15),
    ]


def test_da3_object_id_rejects_ambiguous_sample_names() -> None:
    module = _load_module()
    assert module.object_id_from_sample_id("uuid__v04") == "uuid"
    try:
        module.object_id_from_sample_id("uuid-v04")
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous sample ID was accepted")


def test_da3_reuse_requires_identical_observations_across_variants() -> None:
    module = _load_module()
    module._validate_shared_observations(SharedObservationDataset())

    class Changed(SharedObservationDataset):
        def load_static(self, index: int):
            sample = super().load_static(index)
            if index == 1:
                sample["extrinsics"][0, 0, 3] = 1.0
            return sample

    try:
        module._validate_shared_observations(Changed())
    except ValueError as error:
        assert "identical RGB/camera inputs" in str(error)
    else:
        raise AssertionError("variant-specific cameras were accepted for DA3 reuse")
