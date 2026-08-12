from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
from mlr.data import Mesh
from mlr.learned_laplacian.synthetic_current_recursive_refinement import (
    GEOMETRY_FIELDS,
    POLICIES,
    PREDICTION_FIELDS,
    aggregate_recursive_rows,
    build_recursive_sample,
    merge_recursive_refinement_shards,
)


def test_recursive_sample_rebuilds_all_geometry_dependent_inputs() -> None:
    original = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    refined = original.copy()
    refined[2, 2] = 0.2
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    source = {
        "sample_id": "mesh__v00",
        "vertices": torch.as_tensor(original, dtype=torch.float32),
        "faces": torch.as_tensor(faces),
        "gt_vertices": torch.as_tensor(original + 0.1, dtype=torch.float32),
        "gt_faces": torch.as_tensor(faces),
        "intrinsics": torch.eye(3).repeat(2, 1, 1),
        "extrinsics": torch.eye(4).repeat(2, 1, 1),
        "vertex_normals": torch.zeros((4, 3)),
        "initial_laplacian": torch.zeros((4, 3)),
        "laplacian_target": torch.zeros((4, 3)),
        "raw_laplacian_target": torch.zeros((4, 3)),
        "normalized_laplacian_target": torch.zeros((4, 3)),
        "target_confidence": torch.ones(4),
        "local_edge_length": torch.ones(4),
        "local_edge_scale": torch.ones(4),
        "valid_scale_mask": torch.ones(4, dtype=torch.bool),
        "visibility_backface_and_occlusion": torch.zeros((2, 4), dtype=torch.bool),
        "position_normalization_center": torch.zeros(3),
        "position_normalization_scale": torch.tensor(1.0),
        "metadata": {"edge_scale_epsilon": 1e-12},
    }
    visibility = np.array(
        [[True, True, False, False], [False, True, True, False]], dtype=bool
    )

    result = build_recursive_sample(
        source, Mesh(refined, faces).ensure_normals(), visibility
    )

    np.testing.assert_allclose(result["vertices"].numpy(), refined)
    assert not torch.equal(result["vertex_normals"], source["vertex_normals"])
    assert torch.equal(result["visibility"], torch.as_tensor(visibility))
    assert torch.equal(result["visibility_backface_and_occlusion"], result["visibility"])
    assert not torch.equal(result["local_edge_length"], source["local_edge_length"])
    assert result["position_normalization_scale"] > 0
    operator = build_uniform_laplacian_data(faces, len(refined))
    np.testing.assert_allclose(
        result["initial_laplacian"].numpy(),
        apply_uniform_laplacian(refined, operator),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result["raw_laplacian_target"].numpy(),
        apply_uniform_laplacian(original + 0.1, operator),
        rtol=1e-6,
        atol=1e-7,
    )


def test_recursive_sample_rejects_topology_change() -> None:
    source = {
        "vertices": torch.zeros((4, 3)),
        "faces": torch.tensor([[0, 1, 2], [0, 2, 3]]),
        "intrinsics": torch.eye(3).reshape(1, 3, 3),
    }
    changed = Mesh(np.zeros((4, 3)), np.array([[0, 1, 3], [0, 2, 3]]))
    with pytest.raises(ValueError, match="preserve faces"):
        build_recursive_sample(source, changed, np.ones((1, 4), dtype=bool))


def test_aggregate_tracks_retained_gained_and_lost_successes() -> None:
    rows = _fake_rows()

    aggregate = aggregate_recursive_rows(rows, rounds=3)

    primary = {
        int(row["round"]): row
        for row in aggregate
        if row["policy"] == POLICIES[0]
    }
    assert primary[0]["cumulative_improved_over_original"] == 19
    assert primary[1]["cumulative_improved_over_original"] == 20
    assert primary[1]["retained_round0_successes"] == 19
    assert primary[1]["gained_from_round0_failures"] == 1
    assert primary[1]["lost_round0_successes"] == 0
    assert primary[2]["cumulative_improved_over_original"] == 18
    assert primary[2]["retained_round0_successes"] == 18
    assert primary[2]["lost_round0_successes"] == 1
    assert primary[3]["cumulative_improved_over_original"] == 21
    assert primary[3]["gained_from_round0_failures"] == 2


def test_three_shards_merge_complete_recursive_contract(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"samples": []}\n', encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = tmp_path / "output"
    rows = _fake_rows()
    for shard_index in range(3):
        shard_rows = [
            row
            for row in rows
            if int(str(row["sample_id"]).split("_")[-1]) % 3 == shard_index
        ]
        sample_ids = sorted({str(row["sample_id"]) for row in shard_rows})
        payload = {
            "shard_index": shard_index,
            "shard_count": 3,
            "rounds": 3,
            "policies": list(POLICIES),
            "manifest_sha256": manifest_hash,
            "checkpoint": "/run/checkpoint_latest.pt",
            "checkpoint_sha256": "checkpoint",
            "baseline_analysis_dir": "/analysis",
            "visibility_size": 960,
            "rows": shard_rows,
            "sample_audits": [
                {"sample_id": sample_id, "faces_preserved": True}
                for sample_id in sample_ids
            ],
        }
        path = output / "shards" / f"shard_{shard_index}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    summary = merge_recursive_refinement_shards(
        manifest, output, rounds=3, shard_count=3
    )

    assert summary["contract_audit"]["passed"] is True
    assert summary["contract_audit"]["row_count"] == 200
    assert summary["decision"]["any_round_exceeds_19_of_25"] is True
    assert summary["decision"]["maximum_improved_over_original"] == 21
    assert (output / "REPORT.md").is_file()


def _fake_rows() -> list[dict[str, object]]:
    success_counts = {0: 19, 1: 20, 2: 18, 3: 21}
    rows: list[dict[str, object]] = []
    for policy in POLICIES:
        for round_index in range(4):
            for sample_index in range(25):
                success = sample_index < success_counts[round_index]
                row: dict[str, object] = {
                    "policy": policy,
                    "round": round_index,
                    "sample_id": f"sample_{sample_index}",
                    "cumulative_improved_over_original": success,
                    "step_improved_over_previous": sample_index % 2 == 0,
                    "improved_over_round0": round_index > 0 and sample_index < 10,
                    "cumulative_introduced_flipped_faces": 2,
                    "step_introduced_flipped_faces": 1,
                    "mean_step_displacement": 0.01,
                    "mean_cumulative_displacement": 0.02,
                    "mean_confidence": 1.0,
                    "visible_vertex_fraction": 0.7,
                    "mean_visible_views_per_vertex": (
                        None if round_index == 0 else 4.0
                    ),
                }
                row.update({field: 1.0 for field in GEOMETRY_FIELDS})
                row.update({field: 0.1 for field in PREDICTION_FIELDS})
                rows.append(row)
    return rows
