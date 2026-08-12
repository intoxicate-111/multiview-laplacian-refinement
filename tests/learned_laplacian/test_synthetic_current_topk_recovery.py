import numpy as np
import pytest
import torch

from mlr.learned_laplacian.synthetic_current_topk_recovery import (
    PERCENTAGES,
    _descending_residual_ranking,
    _gap_closed,
    _residual_group_masks,
    _selection_audit,
    _topk_count,
    _topk_indices,
)


def test_raw_residual_ranking_is_descending_with_vertex_index_tie_break():
    ranking = _descending_residual_ranking(np.array([1.0, 3.0, 3.0, 2.0]))
    np.testing.assert_array_equal(ranking, np.array([1, 2, 3, 0]))


def test_positive_topk_uses_ceil_and_endpoints_are_exact():
    assert [_topk_count(101, percentage) for percentage in PERCENTAGES] == [
        0,
        2,
        11,
        21,
        51,
        101,
    ]
    ranking = np.arange(101)
    assert len(_topk_indices(ranking, 101, 1)) == 2
    assert len(_topk_indices(ranking, 101, 100)) == 101


def test_residual_groups_partition_vertices_exactly_once():
    ranking = _descending_residual_ranking(np.arange(101, dtype=float))
    groups = _residual_group_masks(ranking, 101)
    covered = np.sum(np.stack(list(groups.values())), axis=0)
    np.testing.assert_array_equal(covered, np.ones(101))
    assert sum(int(mask.sum()) for mask in groups.values()) == 101


def test_selection_audit_requires_nested_exact_endpoints():
    ranking = np.arange(101)
    selections = {
        percentage: _topk_indices(ranking, 101, percentage)
        for percentage in PERCENTAGES
    }
    audit = _selection_audit(
        "sample", "current_query_20k", selections, torch.arange(101.0, 0.0, -1.0)
    )
    assert audit["passed"] is True


def test_gap_closed_uses_baseline_and_exact_endpoint():
    assert _gap_closed(4.0, 4.0, 2.0) == 0.0
    assert _gap_closed(4.0, 3.0, 2.0) == 0.5
    assert _gap_closed(4.0, 2.0, 2.0) == 1.0
    assert _gap_closed(2.0, 2.0, 2.0) == 1.0
