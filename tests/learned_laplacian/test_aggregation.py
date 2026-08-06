import torch

from mlr.learned_laplacian.aggregation import masked_mean_aggregate


def test_masked_mean_is_correct():
    features = torch.tensor([[[1.0], [4.0]], [[3.0], [8.0]]])
    valid = torch.tensor([[True, False], [True, True]])
    aggregated, ratio = masked_mean_aggregate(features, valid)
    torch.testing.assert_close(aggregated, torch.tensor([[2.0], [8.0]]))
    torch.testing.assert_close(ratio, torch.tensor([1.0, 0.5]))


def test_zero_valid_views_produce_zero_without_nan():
    aggregated, ratio = masked_mean_aggregate(
        torch.randn(2, 3, 4), torch.zeros(2, 3, dtype=torch.bool)
    )
    assert torch.isfinite(aggregated).all()
    assert torch.count_nonzero(aggregated) == 0
    assert torch.count_nonzero(ratio) == 0


def test_float16_view_aggregation_is_stable_under_consistent_permutation():
    generator = torch.Generator().manual_seed(23)
    features = (
        torch.randn(14, 37, 11, generator=generator, dtype=torch.float32) * 128.0
    ).to(dtype=torch.float16)
    valid = torch.rand(14, 37, generator=generator) > 0.35
    permutation = torch.randperm(14, generator=generator)

    original, original_ratio = masked_mean_aggregate(features, valid)
    permuted, permuted_ratio = masked_mean_aggregate(
        features[permutation], valid[permutation]
    )

    assert original.dtype == torch.float32
    torch.testing.assert_close(original, permuted, rtol=1.0e-6, atol=1.0e-5)
    torch.testing.assert_close(original_ratio, permuted_ratio, rtol=0.0, atol=0.0)
