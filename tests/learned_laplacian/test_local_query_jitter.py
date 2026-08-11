import copy

import pytest
import torch

from mlr.learned_laplacian.local_query_jitter import (
    LocalQueryJitterSettings,
    apply_local_query_jitter,
    local_query_jitter_settings,
    validate_local_query_jitter_contract,
)

from .helpers import tiny_sample


def _sample() -> dict:
    sample = tiny_sample()
    vertex_count = int(sample["vertices"].shape[0])
    sample["local_edge_length"] = torch.full((vertex_count,), 0.5)
    sample["valid_scale_mask"] = torch.ones(vertex_count, dtype=torch.bool)
    sample["raw_laplacian_target"] = sample["laplacian_target"].clone()
    sample["normalized_laplacian_target"] = sample["laplacian_target"].clone()
    sample["edge_index"] = torch.tensor(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
    )
    sample["sample_id"] = "fixed_current__v00"
    sample["metadata"] = {
        "query_training_mode": "fixed_synthetic_current_graph_v1",
        "proxy_definition": "P_proxy=source_gt_vertices_with_exact_same_topology",
        "target_constructor": "delta_target=L_current@P_proxy",
    }
    return sample


def test_two_runtime_jitters_differ_and_targets_graph_and_scale_are_exactly_frozen():
    sample = _sample()
    before = copy.deepcopy(sample)
    settings = LocalQueryJitterSettings(enabled=True, std_h=0.003)
    first = apply_local_query_jitter(sample, settings, base_seed=7, epoch=1)
    second = apply_local_query_jitter(sample, settings, base_seed=7, epoch=2)

    assert not torch.equal(first["query_positions"], second["query_positions"])
    for field in (
        "vertices",
        "faces",
        "edge_index",
        "initial_laplacian",
        "local_edge_length",
        "raw_laplacian_target",
        "normalized_laplacian_target",
        "target_positions",
    ):
        assert torch.equal(sample[field], before[field])
        assert torch.equal(first[field], before[field])
        assert torch.equal(second[field], before[field])
    assert first["local_query_jitter_diagnostics"]["max_offset_norm_over_h"] <= (
        3.0 * settings.std_h + 1e-6
    )


def test_local_jitter_is_deterministic_for_sample_seed_epoch():
    sample = _sample()
    settings = LocalQueryJitterSettings(enabled=True, std_h=0.003)
    first = apply_local_query_jitter(sample, settings, base_seed=7, epoch=4)
    second = apply_local_query_jitter(sample, settings, base_seed=7, epoch=4)
    torch.testing.assert_close(first["query_positions"], second["query_positions"])


def test_local_jitter_contract_and_single_scale_validation():
    sample = _sample()
    validate_local_query_jitter_contract(sample)
    parsed = local_query_jitter_settings(
        {"local_query_jitter": {"enabled": True, "std_h": 0.003}}
    )
    assert parsed == LocalQueryJitterSettings(enabled=True, std_h=0.003)
    with pytest.raises(ValueError, match="must not exceed"):
        local_query_jitter_settings(
            {"local_query_jitter": {"enabled": True, "std_h": 0.0101}}
        )
    invalid = _sample()
    invalid["metadata"]["query_training_mode"] = "gt_vertex_perturbation_v1"
    with pytest.raises(ValueError, match="fixed_synthetic_current_graph_v1"):
        validate_local_query_jitter_contract(invalid)
