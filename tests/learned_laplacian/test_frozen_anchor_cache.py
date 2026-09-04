from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from mlr.learned_laplacian.frozen_anchor_cache import (
    FrozenAnchorCache,
    FrozenAnchorDataset,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshRecord


def test_frozen_anchor_dataset_attaches_exact_audited_vertices(tmp_path) -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    anchor = vertices.numpy() + 0.125
    anchor_path = tmp_path / "cached.npy"
    np.save(anchor_path, anchor.astype(np.float32), allow_pickle=False)
    digest = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
    metadata = {
        "contract_audit": True,
        "arm_e_checkpoint_sha256": "e" * 64,
        "records": [
            {
                "sample_id": "cached",
                "split": "train",
                "path": anchor_path.name,
                "vertex_count": len(anchor),
                "anchor_sha256": digest,
            }
        ],
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    class Dataset:
        records = (PreparedMeshRecord(tmp_path / "unused.pt", "train", "cached", tmp_path),)
        sample_ids = ("cached",)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {"sample_id": "cached", "vertices": vertices}

        def load_static(self, index):
            return self[index]

    dataset = Dataset()
    wrapped = FrozenAnchorDataset(
        dataset,
        FrozenAnchorCache(metadata_path, expected_checkpoint_sha256="e" * 64),
    )

    loaded = wrapped.load_static(0)
    np.testing.assert_array_equal(
        loaded["recovery_anchor_vertices"].numpy(), anchor.astype(np.float32)
    )
    assert not loaded["recovery_anchor_vertices"].requires_grad
