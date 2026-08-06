from __future__ import annotations

import torch

from mlr.learned_laplacian.image_ablation import _condition_sample


def _sample() -> dict[str, torch.Tensor]:
    return {
        "images": torch.arange(4 * 3 * 2 * 2).reshape(4, 3, 2, 2).float(),
        "intrinsics": torch.arange(4 * 9).reshape(4, 3, 3).float(),
        "extrinsics": torch.arange(4 * 16).reshape(4, 4, 4).float(),
        "visibility": torch.arange(4 * 5).reshape(4, 5).bool(),
    }


def test_image_shuffle_breaks_only_image_camera_correspondence() -> None:
    sample = _sample()
    permutation = torch.tensor([2, 0, 3, 1])
    changed = _condition_sample(sample, sample["images"], permutation, "shuffled_images")
    assert torch.equal(changed["images"], sample["images"][permutation])
    assert torch.equal(changed["intrinsics"], sample["intrinsics"])
    assert torch.equal(changed["extrinsics"], sample["extrinsics"])
    assert torch.equal(changed["visibility"], sample["visibility"])


def test_consistent_view_shuffle_reorders_all_view_indexed_inputs() -> None:
    sample = _sample()
    permutation = torch.tensor([2, 0, 3, 1])
    changed = _condition_sample(sample, sample["images"], permutation, "shuffled_view_order")
    for name in ("images", "intrinsics", "extrinsics", "visibility"):
        assert torch.equal(changed[name], sample[name][permutation])


def test_zero_and_cross_object_conditions_are_exact() -> None:
    sample = _sample()
    donor = sample["images"] + 1000
    permutation = torch.arange(4)
    zero = _condition_sample(sample, donor, permutation, "zero_rgb")
    cross = _condition_sample(sample, donor, permutation, "cross_object_rgb")
    assert torch.count_nonzero(zero["images"]).item() == 0
    assert torch.equal(cross["images"], donor)
