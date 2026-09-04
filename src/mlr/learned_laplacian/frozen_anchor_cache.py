from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from .multi_dataset import PreparedMeshDataset, PreparedMeshRecord


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenAnchorRecord:
    sample_id: str
    split: str
    path: Path
    vertex_count: int
    sha256: str


class FrozenAnchorCache:
    """Audited per-sample frozen positional anchors keyed by exact sample ID."""

    def __init__(
        self,
        metadata_path: str | Path,
        *,
        expected_checkpoint_sha256: str | None = None,
    ) -> None:
        self.metadata_path = Path(metadata_path).resolve()
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("contract_audit") is not True:
            raise ValueError("Frozen-anchor cache metadata did not pass its contract audit.")
        actual_checkpoint = str(payload.get("arm_e_checkpoint_sha256", ""))
        if (
            expected_checkpoint_sha256 is not None
            and actual_checkpoint != expected_checkpoint_sha256
        ):
            raise ValueError("Frozen-anchor cache uses the wrong Arm-E checkpoint.")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError("Frozen-anchor cache metadata has no records.")
        records: dict[str, FrozenAnchorRecord] = {}
        ordered_ids: list[str] = []
        for raw in raw_records:
            if not isinstance(raw, Mapping):
                raise ValueError("Frozen-anchor record must be an object.")
            sample_id = str(raw["sample_id"])
            relative = Path(str(raw["path"]))
            path = (
                relative
                if relative.is_absolute()
                else self.metadata_path.parent / relative
            ).resolve()
            record = FrozenAnchorRecord(
                sample_id=sample_id,
                split=str(raw["split"]),
                path=path,
                vertex_count=int(raw["vertex_count"]),
                sha256=str(raw["anchor_sha256"]),
            )
            if sample_id in records:
                raise ValueError(f"Duplicate frozen anchor sample ID: {sample_id}")
            if not path.is_file() or sha256_file(path) != record.sha256:
                raise ValueError(f"Frozen anchor file/hash mismatch: {path}")
            records[sample_id] = record
            ordered_ids.append(sample_id)
        self.payload = dict(payload)
        self.records = records
        self.ordered_ids = tuple(ordered_ids)

    def anchor(self, sample_id: str, split: str) -> torch.Tensor:
        try:
            record = self.records[sample_id]
        except KeyError as error:
            raise KeyError(f"No frozen positional anchor for {sample_id}") from error
        if record.split != split:
            raise ValueError(
                f"Frozen anchor split mismatch for {sample_id}: "
                f"{record.split!r} != {split!r}"
            )
        value = np.load(record.path, allow_pickle=False)
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (record.vertex_count, 3) or not np.isfinite(array).all():
            raise ValueError(f"Invalid frozen anchor array: {record.path}")
        return torch.from_numpy(np.ascontiguousarray(array).copy()).detach()


class FrozenAnchorDataset(Sequence[dict[str, Any]]):
    """Prepared dataset view that adds a frozen loss-side recovery anchor."""

    def __init__(
        self,
        dataset: PreparedMeshDataset,
        cache: FrozenAnchorCache,
    ) -> None:
        self.dataset = dataset
        self.cache = cache
        self.records: tuple[PreparedMeshRecord, ...] = dataset.records
        self.split = self.records[0].split
        missing = [sample_id for sample_id in dataset.sample_ids if sample_id not in cache.records]
        if missing:
            raise ValueError("Frozen-anchor cache is missing IDs: " + ", ".join(missing))
        for sample_id in dataset.sample_ids:
            if cache.records[sample_id].split != self.split:
                raise ValueError(f"Frozen-anchor split mismatch for {sample_id}")

    def __len__(self) -> int:
        return len(self.dataset)

    def _attach(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(sample)
        sample_id = str(result["sample_id"])
        anchor = self.cache.anchor(sample_id, self.split)
        if tuple(anchor.shape) != tuple(result["vertices"].shape):
            raise ValueError(f"Frozen anchor/input topology mismatch for {sample_id}")
        result["recovery_anchor_vertices"] = anchor
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._attach(self.dataset[index])

    def load_static(self, index: int) -> dict[str, Any]:
        return self._attach(self.dataset.load_static(index))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self.dataset.sample_ids
