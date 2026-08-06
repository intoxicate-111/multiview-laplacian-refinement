from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .dataset import load_prepared_sample


@dataclass(frozen=True)
class PreparedMeshRecord:
    path: Path
    split: str
    sample_id: str | None = None
    dataset_root: Path | None = None


class PreparedMeshDataset(Sequence[dict[str, Any]]):
    """Lazily load variable-size prepared mesh samples from a JSON manifest."""

    def __init__(self, records: Sequence[PreparedMeshRecord]) -> None:
        self.records = tuple(records)
        self._cache: dict[int, dict[str, Any]] = {}
        if not self.records:
            raise ValueError("PreparedMeshDataset requires at least one sample.")
        splits = {record.split for record in self.records}
        if len(splits) != 1:
            raise ValueError("PreparedMeshDataset records must belong to exactly one split.")
        missing = [str(record.path) for record in self.records if not record.path.is_file()]
        if missing:
            raise FileNotFoundError("Prepared sample files do not exist: " + ", ".join(missing))
        declared_ids = [record.sample_id for record in self.records if record.sample_id is not None]
        if len(declared_ids) != len(set(declared_ids)):
            raise ValueError("Manifest sample_id values must be unique within a split.")

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, split: str) -> "PreparedMeshDataset":
        manifest_path = Path(manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or not isinstance(payload.get("samples"), list):
            raise ValueError("Manifest must be an object containing a 'samples' list.")
        records: list[PreparedMeshRecord] = []
        for index, item in enumerate(payload["samples"]):
            if not isinstance(item, Mapping):
                raise ValueError(f"Manifest sample {index} must be an object.")
            item_split = item.get("split")
            if not isinstance(item_split, str) or not item_split:
                raise ValueError(f"Manifest sample {index} requires a non-empty split.")
            path_value = item.get("path")
            if not isinstance(path_value, str) or not path_value:
                raise ValueError(f"Manifest sample {index} requires a non-empty path.")
            if item_split != split:
                continue
            path = Path(path_value)
            if not path.is_absolute():
                path = manifest_path.parent / path
            sample_id = item.get("sample_id")
            if sample_id is not None and (not isinstance(sample_id, str) or not sample_id):
                raise ValueError(f"Manifest sample {index} has an invalid sample_id.")
            records.append(
                PreparedMeshRecord(
                    path=path.resolve(),
                    split=item_split,
                    sample_id=sample_id,
                    dataset_root=manifest_path.parent.resolve(),
                )
            )
        if not records:
            raise ValueError(f"Manifest contains no samples for split {split!r}.")
        return cls(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index in self._cache:
            return self._cache[index]
        record = self.records[index]
        sample = load_prepared_sample(record.path, dataset_root=record.dataset_root)
        if record.sample_id is not None and sample["sample_id"] != record.sample_id:
            raise ValueError(
                f"Manifest declares sample_id {record.sample_id!r} for {record.path}, "
                f"but the file contains {sample['sample_id']!r}."
            )
        # Embedded-image samples retain the historical single-load cache. Lazy
        # image-path samples are intentionally not retained with their decoded
        # 14-view tensors, which would recreate the dataset-sized RAM/GPU cache.
        if sample.get("prepared_storage_format") != "lazy_image_paths_v1":
            self._cache[index] = sample
        return sample

    def load_static(self, index: int) -> dict[str, Any]:
        """Load tensors and lazy image metadata without decoding image pixels."""
        if index in self._cache:
            return self._cache[index]
        record = self.records[index]
        sample = load_prepared_sample(
            record.path,
            materialize_images=False,
            dataset_root=record.dataset_root,
        )
        if record.sample_id is not None and sample["sample_id"] != record.sample_id:
            raise ValueError(
                f"Manifest declares sample_id {record.sample_id!r} for {record.path}, "
                f"but the file contains {sample['sample_id']!r}."
            )
        if sample.get("prepared_storage_format") != "lazy_image_paths_v1":
            self._cache[index] = sample
        return sample

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        result = []
        for index, record in enumerate(self.records):
            # Avoid decoding every lazy sample's images merely to validate split IDs.
            result.append(record.sample_id or self.load_static(index)["sample_id"])
        return tuple(result)


def validate_disjoint_splits(*datasets: PreparedMeshDataset) -> None:
    """Reject path or sample-ID leakage across manifest-backed splits."""

    seen_paths: dict[Path, str] = {}
    seen_ids: dict[str, str] = {}
    for dataset in datasets:
        split = dataset.records[0].split
        for record in dataset.records:
            previous = seen_paths.get(record.path)
            if previous is not None:
                raise ValueError(
                    f"Prepared sample path {record.path} appears in both {previous!r} and "
                    f"{split!r} splits."
                )
            seen_paths[record.path] = split
        for sample_id in dataset.sample_ids:
            previous = seen_ids.get(sample_id)
            if previous is not None:
                raise ValueError(
                    f"sample_id {sample_id!r} appears in both {previous!r} and {split!r} splits."
                )
            seen_ids[sample_id] = split
