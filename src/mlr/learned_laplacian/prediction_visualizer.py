from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mlr.data import Camera, Mesh
from mlr.io import save_mesh
from mlr.laplacian import compute_laplacian_coordinates
from mlr.refinement import RefinementConfig, refine_mesh_with_laplacian
from mlr.synthetic import create_orbit_cameras

from .dataset import load_prepared_sample
from .target_scaling import (
    EDGE_SCALE_DEFINITION,
    EDGE_SCALE_NORMALIZED_LAPLACIAN,
    RAW_LAPLACIAN,
    denormalize_laplacian_by_edge_scale,
)
from .visualization import render_mesh_comparison_grid


RAW_SUFFIX = "_raw_delta.npy"
TARGET_SUFFIX = "_target_space_delta.npy"


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    raw_path: Path | None = None
    target_space_path: Path | None = None
    record_path: Path | None = None


@dataclass(frozen=True)
class RunMetadata:
    run_dir: Path
    config: dict[str, Any]
    metrics: dict[str, Any]
    manifest_path: Path | None
    manifest: dict[str, Any] | None
    config_source: str | None
    manifest_source: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisualizationOptions:
    output_dir: Path
    camera_index: int = 0
    image_size: int = 256
    columns: int = 3
    operator_type: str | None = None
    lambda_lap: float | None = None
    lambda_anchor: float | None = None
    lambda_edge: float | None = None
    num_iters: int | None = None
    learning_rate: float | None = None
    device: str = "cpu"
    overwrite: bool = False
    skip_render: bool = False
    skip_refinement: bool = False
    progress: bool = True


def discover_run_metadata(
    run_dir: str | Path, manifest_override: str | Path | None = None
) -> RunMetadata:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    warnings: list[str] = []
    config: dict[str, Any] = {}
    config_source = None
    for name in ("config.json", "run_config.json"):
        path = run_dir / name
        if path.is_file():
            payload = _read_json_object(path)
            nested = payload.get("experiment_config")
            config = dict(nested) if isinstance(nested, Mapping) else payload
            for key in ("manifest_path", "manifest", "dataset_manifest"):
                if key in payload:
                    config.setdefault(key, payload[key])
            config_source = str(path)
            break
    if not config:
        checkpoint = run_dir / "best.pt"
        if checkpoint.is_file():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            candidate = payload.get("experiment_config") if isinstance(payload, Mapping) else None
            if isinstance(candidate, Mapping):
                config = dict(candidate)
                config_source = f"{checkpoint}:experiment_config"

    metrics_path = run_dir / "metrics.json"
    metrics = _read_json_object(metrics_path) if metrics_path.is_file() else {}
    manifest_path = (
        Path(manifest_override).resolve()
        if manifest_override is not None
        else _discover_manifest_path(run_dir, config)
    )
    if manifest_override is not None and not manifest_path.is_file():
        raise FileNotFoundError(f"Explicit dataset manifest does not exist: {manifest_path}")
    manifest = None
    manifest_source = None
    if manifest_path is not None:
        manifest = _read_json_object(manifest_path)
        if not isinstance(manifest.get("samples"), list):
            raise ValueError(f"Dataset manifest has no 'samples' list: {manifest_path}")
        manifest_source = str(manifest_path)
    else:
        warnings.append(
            "Dataset manifest was not found; predictions can be listed but samples cannot be "
            "reconstructed safely."
        )
    return RunMetadata(
        run_dir=run_dir,
        config=config,
        metrics=metrics,
        manifest_path=manifest_path,
        manifest=manifest,
        config_source=config_source,
        manifest_source=manifest_source,
        warnings=tuple(warnings),
    )


def discover_predictions(run_dir: str | Path, split: str) -> dict[str, PredictionRecord]:
    prediction_dir = Path(run_dir).resolve() / "predictions" / split
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {prediction_dir}")
    records: dict[str, PredictionRecord] = {}
    for path in sorted(prediction_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.endswith(RAW_SUFFIX):
            sample_id = path.name[: -len(RAW_SUFFIX)]
            current = records.get(sample_id, PredictionRecord(sample_id))
            records[sample_id] = PredictionRecord(
                sample_id, path, current.target_space_path, current.record_path
            )
        elif path.name.endswith(TARGET_SUFFIX):
            sample_id = path.name[: -len(TARGET_SUFFIX)]
            current = records.get(sample_id, PredictionRecord(sample_id))
            records[sample_id] = PredictionRecord(
                sample_id, current.raw_path, path, current.record_path
            )
        elif path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                names = set(archive.files)
                sample_id = (
                    str(archive["sample_id"].item())
                    if "sample_id" in names
                    else re.sub(r"_prediction$", "", path.stem)
                )
                if not ({"raw_delta", "target_space_delta"} & names):
                    continue
            current = records.get(sample_id, PredictionRecord(sample_id))
            records[sample_id] = PredictionRecord(
                sample_id, current.raw_path, current.target_space_path, path
            )
    return dict(sorted(records.items()))


def load_prediction_sample(
    metadata: RunMetadata, split: str, sample_id: str
) -> tuple[dict[str, Any], Path]:
    if metadata.manifest is None or metadata.manifest_path is None:
        raise ValueError(
            "Sample metadata is unavailable. Add dataset_manifest.json to the run directory "
            "or record an explicit manifest path in the run config."
        )
    matches = [
        item
        for item in metadata.manifest["samples"]
        if isinstance(item, Mapping)
        and item.get("split") == split
        and item.get("sample_id") == sample_id
    ]
    if not matches:
        raise KeyError(f"Manifest has no sample_id {sample_id!r} in split {split!r}.")
    if len(matches) > 1:
        raise ValueError(f"Manifest contains duplicate sample_id {sample_id!r} in split {split!r}.")
    path_value = matches[0].get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"Manifest record for {sample_id!r} has no valid path.")
    sample_path = Path(path_value)
    if not sample_path.is_absolute():
        sample_path = metadata.manifest_path.parent / sample_path
    sample_path = sample_path.resolve()
    if not sample_path.is_file():
        raise FileNotFoundError(f"Prepared sample does not exist: {sample_path}")
    sample = load_prepared_sample(sample_path)
    if sample["sample_id"] != sample_id:
        raise ValueError(
            f"Prepared sample ID {sample['sample_id']!r} does not match prediction ID "
            f"{sample_id!r}."
        )
    return sample, sample_path


def resolve_refinement_config(
    run_config: Mapping[str, Any],
    sample: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> tuple[RefinementConfig, list[str]]:
    values = asdict(RefinementConfig())
    reconstruction = run_config.get("reconstruction", {})
    if isinstance(reconstruction, Mapping):
        for name in values:
            if name in reconstruction:
                values[name] = reconstruction[name]
    warnings: list[str] = []
    sample_operator = sample.get("metadata", {}).get("operator_type")
    run_operator = reconstruction.get("operator_type") if isinstance(reconstruction, Mapping) else None
    if sample_operator is not None and run_operator is not None and sample_operator != run_operator:
        raise ValueError(
            f"Operator mismatch: sample records {sample_operator!r}, run config records "
            f"{run_operator!r}."
        )
    if run_operator is None and sample_operator is not None:
        values["operator_type"] = sample_operator
    if run_operator is None and sample_operator is None:
        warnings.append(
            "Operator type was not recorded; falling back to repository default: "
            f"{values['operator_type']}"
        )
    override_operator = (overrides or {}).get("operator_type")
    if (
        override_operator is not None
        and sample_operator is not None
        and override_operator != sample_operator
    ):
        raise ValueError(
            f"Operator override {override_operator!r} is incompatible with the sample target "
            f"operator {sample_operator!r}."
        )
    for name, value in (overrides or {}).items():
        if value is not None and name in values:
            values[name] = value
    config = RefinementConfig(**values)
    if config.num_iters < 1 or config.learning_rate <= 0:
        raise ValueError("Refinement num_iters and learning_rate must be positive.")
    if config.lambda_lap < 0 or config.lambda_anchor < 0 or config.lambda_edge < 0:
        raise ValueError("Refinement lambda values must be non-negative.")
    return config, warnings


def visualize_prediction_sample(
    metadata: RunMetadata,
    split: str,
    record: PredictionRecord,
    options: VisualizationOptions,
) -> dict[str, Any]:
    sample_output = options.output_dir / _safe_sample_id(record.sample_id)
    summary_path = sample_output / "summary.json"
    if summary_path.exists() and not options.overwrite:
        raise FileExistsError(
            f"Visualization already exists for {record.sample_id!r}: {sample_output}. "
            "Use --overwrite to replace it."
        )
    sample, sample_path = load_prediction_sample(metadata, split, record.sample_id)
    vertices = _numpy(sample["vertices"]).astype(np.float64, copy=False)
    faces = _numpy(sample["faces"]).astype(np.int64, copy=False)
    if not np.isfinite(vertices).all():
        raise ValueError("Input mesh vertices contain NaN or infinite values.")
    raw_delta, prediction_path, prediction_format, prediction_space = _load_raw_prediction(
        record, metadata, sample
    )
    _validate_prediction(raw_delta, vertices)
    coarse = Mesh(vertices.copy(), faces.copy()).ensure_normals()
    gt_mesh = _gt_mesh(sample)
    refinement_config, warnings = resolve_refinement_config(
        metadata.config,
        sample,
        {
            "operator_type": options.operator_type,
            "lambda_lap": options.lambda_lap,
            "lambda_anchor": options.lambda_anchor,
            "lambda_edge": options.lambda_edge,
            "num_iters": options.num_iters,
            "learning_rate": options.learning_rate,
        },
    )
    warnings = list(metadata.warnings) + warnings
    target_mode, target_scaling = _target_metadata(metadata, sample)
    norms = np.linalg.norm(raw_delta, axis=1)
    bbox_diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if np.allclose(raw_delta, 0.0):
        warnings.append("Prediction is identically zero.")
    if float(norms.max(initial=0.0)) > max(100.0 * bbox_diagonal, 1e6):
        warnings.append("Prediction norm is extremely large relative to the input mesh.")
    if options.progress:
        print(f"sample ID: {record.sample_id}")
        print(f"input vertex shape: {vertices.shape}")
        print(f"face shape: {faces.shape}")
        print(f"prediction shape: {raw_delta.shape}")
        print(
            "prediction norm mean/median/max: "
            f"{np.mean(norms):.8g} / {np.median(norms):.8g} / {np.max(norms):.8g}"
        )
        print(f"GT shape: {None if gt_mesh is None else gt_mesh.vertices.shape}")
        print(f"operator type: {refinement_config.operator_type}")
        print(f"target mode: {target_mode}")
        print(f"target scaling: {target_scaling}")

    sample_output.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, str] = {}
    save_mesh(coarse, sample_output / "coarse.obj")
    output_files["coarse"] = "coarse.obj"
    refined = coarse
    history: list[dict[str, float]] = []
    if not options.skip_refinement:
        confidence = _numpy(sample["target_confidence"])
        result = refine_mesh_with_laplacian(
            mesh=coarse,
            delta_target=raw_delta,
            confidence=confidence,
            anchors=coarse.vertices,
            config=refinement_config,
        )
        refined = result.mesh
        history = result.history
        if not np.isfinite(refined.vertices).all():
            raise FloatingPointError("Refinement produced NaN or infinite vertices.")
        if not history or not np.isfinite(history[-1]["loss"]):
            raise FloatingPointError("Refinement produced a non-finite final loss.")
        save_mesh(refined, sample_output / "predicted_refined.obj")
        output_files["predicted_refined"] = "predicted_refined.obj"
    if gt_mesh is not None:
        save_mesh(gt_mesh, sample_output / "gt.obj")
        output_files["gt"] = "gt.obj"
    (sample_output / "refinement_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    output_files["refinement_history"] = "refinement_history.json"
    np.save(sample_output / "vertex_displacement.npy", refined.vertices - coarse.vertices)
    output_files["vertex_displacement"] = "vertex_displacement.npy"
    residual = compute_laplacian_coordinates(
        refined.vertices, refined.faces, refinement_config.operator_type
    ) - raw_delta
    np.save(sample_output / "laplacian_residual.npy", residual)
    output_files["laplacian_residual"] = "laplacian_residual.npy"

    camera, camera_source = _resolve_camera(sample, coarse, options.camera_index, options.image_size)
    if not options.skip_render:
        entries = [("Coarse", coarse)]
        if not options.skip_refinement:
            entries.append(("Predicted", refined))
        if gt_mesh is not None:
            entries.append(("GT", gt_mesh))
        render_mesh_comparison_grid(
            entries,
            camera,
            sample_output / "comparison.png",
            image_size=options.image_size,
            columns=options.columns,
        )
        output_files["comparison"] = "comparison.png"

    displacement = np.linalg.norm(refined.vertices - coarse.vertices, axis=1)
    if float(displacement.max(initial=0.0)) > 2.0 * max(bbox_diagonal, 1e-12):
        warnings.append("Maximum vertex displacement exceeds twice the input bounding-box diagonal.")
    if options.progress:
        print(
            "vertex displacement mean/median/max: "
            f"{np.mean(displacement):.8g} / {np.median(displacement):.8g} / "
            f"{np.max(displacement):.8g}"
        )
        print(f"initial loss: {history[0]['loss'] if history else None}")
        print(f"final loss: {history[-1]['loss'] if history else None}")
        print(f"output directory: {sample_output}")
    sample_metadata = sample.get("metadata", {})
    output_files["summary"] = "summary.json"
    summary = {
        "sample_id": record.sample_id,
        "split": split,
        "run_dir": str(metadata.run_dir),
        "prediction_path": str(prediction_path),
        "prediction_format": prediction_format,
        "prediction_space": prediction_space,
        "input_mesh_path": sample_metadata.get("coarse_mesh_path", str(sample_path)),
        "gt_mesh_path": sample_metadata.get("gt_mesh_path"),
        "prepared_sample_path": str(sample_path),
        "gt_available": gt_mesh is not None,
        "camera_source": camera_source,
        "camera_index": options.camera_index,
        "vertex_count": coarse.num_vertices,
        "face_count": coarse.num_faces,
        "target_mode": target_mode,
        "operator_type": refinement_config.operator_type,
        "target_scaling": target_scaling,
        "raw_delta_mean_norm": float(np.mean(norms)),
        "raw_delta_median_norm": float(np.median(norms)),
        "raw_delta_max_norm": float(np.max(norms)),
        "mean_vertex_displacement": float(np.mean(displacement)),
        "median_vertex_displacement": float(np.median(displacement)),
        "max_vertex_displacement": float(np.max(displacement)),
        "initial_refinement_loss": history[0]["loss"] if history else None,
        "final_refinement_loss": history[-1]["loss"] if history else None,
        "refinement_config": asdict(refinement_config),
        "output_files": output_files,
        "warnings": warnings,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def visualize_prediction_split(
    metadata: RunMetadata,
    split: str,
    records: Sequence[PredictionRecord],
    options: VisualizationOptions,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    succeeded = failed = skipped = 0
    for record in records:
        try:
            summary = visualize_prediction_sample(metadata, split, record, options)
            results.append({"sample_id": record.sample_id, "status": "succeeded", "summary": summary})
            succeeded += 1
        except FileExistsError as error:
            results.append(
                {"sample_id": record.sample_id, "status": "skipped", "message": str(error)}
            )
            skipped += 1
        except Exception as error:  # Batch processing must preserve failures and continue.
            results.append(
                {
                    "sample_id": record.sample_id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            failed += 1
    batch = {
        "processed": len(records),
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "sample_results": results,
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    (options.output_dir / "batch_summary.json").write_text(
        json.dumps(batch, indent=2) + "\n", encoding="utf-8"
    )
    return batch


def prediction_listing(
    metadata: RunMetadata,
    split: str,
    records: Mapping[str, PredictionRecord],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    manifest_ids = set()
    if metadata.manifest is not None:
        manifest_ids = {
            str(item.get("sample_id"))
            for item in metadata.manifest["samples"]
            if isinstance(item, Mapping) and item.get("split") == split
        }
    output_dir = Path(output_dir)
    return [
        {
            "sample_id": sample_id,
            "raw_prediction": record.raw_path is not None
            or _npz_has(record.record_path, "raw_delta"),
            "target_space_prediction": record.target_space_path is not None
            or _npz_has(record.record_path, "target_space_delta"),
            "sample_metadata": sample_id in manifest_ids,
            "visualization_exists": (output_dir / _safe_sample_id(sample_id) / "summary.json").is_file(),
        }
        for sample_id, record in records.items()
    ]


def _discover_manifest_path(run_dir: Path, config: Mapping[str, Any]) -> Path | None:
    local = run_dir / "dataset_manifest.json"
    if local.is_file():
        return local.resolve()
    for key in ("manifest_path", "manifest", "dataset_manifest"):
        value = config.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if not path.is_absolute():
                path = run_dir / path
            if path.is_file():
                return path.resolve()
    return None


def _load_raw_prediction(
    record: PredictionRecord,
    metadata: RunMetadata,
    sample: Mapping[str, Any],
) -> tuple[np.ndarray, Path, str, str]:
    if record.raw_path is not None:
        return np.asarray(np.load(record.raw_path)), record.raw_path, "npy", "raw_laplacian"
    if _npz_has(record.record_path, "raw_delta"):
        with np.load(record.record_path, allow_pickle=False) as archive:
            return (
                np.asarray(archive["raw_delta"]),
                record.record_path,
                "npz",
                "raw_laplacian",
            )
    target_path = record.target_space_path or record.record_path
    if target_path is None:
        raise FileNotFoundError(f"No prediction data found for {record.sample_id!r}.")
    if record.target_space_path is not None:
        target = np.asarray(np.load(record.target_space_path))
        prediction_format = "npy"
    else:
        with np.load(record.record_path, allow_pickle=False) as archive:
            target = np.asarray(archive["target_space_delta"])
        prediction_format = "npz"
    target_mode = metadata.config.get("target_mode")
    scaling = metadata.config.get("target_scaling")
    if (
        target_mode != EDGE_SCALE_NORMALIZED_LAPLACIAN
        or not isinstance(scaling, Mapping)
        or scaling.get("method") != EDGE_SCALE_DEFINITION
        or "epsilon" not in scaling
        or "local_edge_length" not in sample
    ):
        raise ValueError(
            "Raw Laplacian prediction is unavailable and target scaling metadata is insufficient."
        )
    raw = denormalize_laplacian_by_edge_scale(
        torch.as_tensor(target), torch.as_tensor(sample["local_edge_length"])
    ).numpy()
    return raw, target_path, prediction_format, "target_space_inverse_scaled"


def _target_metadata(
    metadata: RunMetadata, sample: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    sample_metadata = sample.get("metadata", {})
    configured_mode = metadata.config.get("target_mode") or metadata.metrics.get("target_mode")
    sample_mode = sample_metadata.get("laplacian_target_mode")
    if configured_mode is not None and sample_mode is not None and configured_mode != sample_mode:
        raise ValueError(
            f"Target mode mismatch: run records {configured_mode!r}, sample records "
            f"{sample_mode!r}."
        )
    target_mode = configured_mode or sample_mode or RAW_LAPLACIAN
    scaling = metadata.config.get("target_scaling")
    if isinstance(scaling, Mapping):
        result = dict(scaling)
    else:
        result = {}
    sample_method = sample_metadata.get("edge_scale_definition")
    if result.get("method") is not None and sample_method is not None:
        if result["method"] != sample_method:
            raise ValueError(
                f"Target scaling mismatch: run records {result['method']!r}, sample records "
                f"{sample_method!r}."
            )
    if "method" not in result and sample_metadata.get("edge_scale_definition") is not None:
        result["method"] = sample_metadata["edge_scale_definition"]
    if "epsilon" not in result and sample_metadata.get("edge_scale_epsilon") is not None:
        result["epsilon"] = sample_metadata["edge_scale_epsilon"]
    return str(target_mode), result


def _validate_prediction(prediction: np.ndarray, vertices: np.ndarray) -> None:
    if prediction.ndim != 2 or prediction.shape[1:] != (3,):
        raise ValueError(f"Prediction must have shape [N, 3], got {prediction.shape}.")
    if prediction.shape != vertices.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match input mesh vertex shape "
            f"{vertices.shape}."
        )
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or infinite values.")


def _gt_mesh(sample: Mapping[str, Any]) -> Mesh | None:
    if sample.get("gt_vertices") is None or sample.get("gt_faces") is None:
        return None
    vertices = _numpy(sample["gt_vertices"])
    faces = _numpy(sample["gt_faces"]).astype(np.int64)
    if not np.isfinite(vertices).all():
        raise ValueError("GT mesh vertices contain NaN or infinite values.")
    return Mesh(vertices, faces).ensure_normals()


def _resolve_camera(
    sample: Mapping[str, Any], mesh: Mesh, camera_index: int, image_size: int
) -> tuple[Camera, str]:
    intrinsics = sample.get("intrinsics")
    extrinsics = sample.get("extrinsics")
    if intrinsics is not None and extrinsics is not None and len(intrinsics) > 0:
        if camera_index < 0 or camera_index >= len(intrinsics):
            raise IndexError(
                f"camera_index {camera_index} is outside [0, {len(intrinsics) - 1}]."
            )
        intrinsic = _numpy(intrinsics[camera_index]).astype(np.float64, copy=True)
        extrinsic = _numpy(extrinsics[camera_index])
        height, width = image_size, image_size
        images = sample.get("images")
        if images is not None and getattr(images, "ndim", 0) == 4:
            height, width = int(images.shape[-2]), int(images.shape[-1])
        intrinsic[0, :] *= image_size / width
        intrinsic[1, :] *= image_size / height
        return (
            Camera(
                intrinsic,
                extrinsic[:3, :3],
                extrinsic[:3, 3],
                image_size=(image_size, image_size),
                name=f"sample_camera_{camera_index}",
            ),
            "prepared_sample",
        )
    if camera_index != 0:
        raise IndexError("Fallback camera only supports camera_index 0.")
    return (
        create_orbit_cameras(mesh, 1, (image_size, image_size))[0],
        "deterministic_orbit_fallback",
    )


def _npz_has(path: Path | None, key: str) -> bool:
    if path is None:
        return False
    with np.load(path, allow_pickle=False) as archive:
        return key in archive.files


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
