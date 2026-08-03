import pytest
import torch

from mlr.learned_laplacian.dataset import load_prepared_sample, save_prepared_sample, validate_sample

from .helpers import tiny_sample


def test_prepared_sample_can_be_saved_and_loaded(tmp_path):
    path = tmp_path / "sample.pt"
    save_prepared_sample(tiny_sample(), path)
    loaded = load_prepared_sample(path)
    assert loaded["sample_id"] == "tiny"
    assert loaded["images"].shape == (1, 3, 16, 16)
    assert loaded["laplacian_target"].shape == (4, 3)


def test_invalid_shape_has_useful_field_name():
    sample = tiny_sample()
    sample["intrinsics"] = torch.eye(4).unsqueeze(0)
    with pytest.raises(ValueError, match="intrinsics must have shape"):
        validate_sample(sample)
