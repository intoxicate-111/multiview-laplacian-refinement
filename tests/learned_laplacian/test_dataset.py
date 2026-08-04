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
    assert loaded["raw_laplacian_target"].shape == (4, 3)
    assert loaded["normalized_laplacian_target"].shape == (4, 3)
    assert loaded["local_edge_length"].shape == (4,)
    assert loaded["valid_scale_mask"].shape == (4,)
    assert loaded["valid_scale_mask"].all()
    torch.testing.assert_close(loaded["local_edge_scale"], loaded["local_edge_length"].square())


def test_legacy_sample_is_upgraded_without_changing_raw_target(tmp_path):
    legacy = tiny_sample()
    original_target = legacy["laplacian_target"].clone()
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)

    loaded = load_prepared_sample(path)

    torch.testing.assert_close(loaded["laplacian_target"], original_target)
    torch.testing.assert_close(loaded["raw_laplacian_target"], original_target)
    assert loaded["metadata"]["edge_scale_definition"] == "square_of_mean_incident_edge_length"
    assert loaded["metadata"]["edge_scale_source"] == "input_prediction_mesh"
    assert loaded["metadata"]["edge_scale_epsilon"] == 1e-12


def test_invalid_shape_has_useful_field_name():
    sample = tiny_sample()
    sample["intrinsics"] = torch.eye(4).unsqueeze(0)
    with pytest.raises(ValueError, match="intrinsics must have shape"):
        validate_sample(sample)
