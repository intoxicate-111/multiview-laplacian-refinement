from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .diagnostics import _amp_settings
from .evaluation import reconstruct_and_evaluate
from .multi_dataset import PreparedMeshDataset
from .multi_trainer import _build_model, _prepare_item_for_use, _prepare_object_static
from .renderer_visibility import VISIBILITY_CONDITIONS
from .target_scaling import denormalize_laplacian_by_edge_scale
from .trainer import load_checkpoint


def run_expanded_renderer_visibility_evaluation(
    run_dir: str | Path,
    expanded_manifest: str | Path,
    output_dir: str | Path,
    *,
    split: str = "validation",
    device: str = "cuda",
    seed: int = 7,
    reconstruction_iters: int = 200,
) -> dict[str, Any]:
    """Evaluate frozen predictions on real expanded queries under four masks.

    The expanded sample's own vertices/faces provide renderer visibility.  The
    schema-required expanded Laplacian target is an identity placeholder, so it
    is deliberately excluded from both prediction metrics and oracle reporting.
    """

    run_dir = Path(run_dir).resolve()
    expanded_manifest = Path(expanded_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_json(run_dir / "config.json")
    dataset = PreparedMeshDataset.from_manifest(expanded_manifest, split)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _build_model(config, None, False).to(resolved_device)
    checkpoint = load_checkpoint(run_dir / "best.pt", model, map_location=resolved_device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, resolved_device)
    reconstruction_config = {
        "operator_type": "uniform",
        "lambda_lap": 1.0,
        "lambda_anchor": 0.01,
        "lambda_edge": 0.0,
        "num_iters": int(reconstruction_iters),
        "learning_rate": 0.01,
        "robust_loss": "huber",
        "huber_delta": 0.01,
        "dense_vertex_limit": 5000,
        "chamfer_samples": 3000,
        "metric_seed": seed,
        "evaluate_oracle": False,
    }

    results: dict[str, Any] = {}
    for condition in VISIBILITY_CONDITIONS:
        print(f"Expanded-query reconstruction: {condition}", flush=True)
        condition_config = copy.deepcopy(config)
        condition_config["renderer_visibility"] = {"condition": condition}
        condition_config.setdefault("query_training", {})["enabled"] = False
        condition_config["query_training"]["zero_initial_laplacian"] = True
        records = []
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            prepared = _prepare_item_for_use(
                _prepare_object_static(static, condition_config),
                condition_config,
                resolved_device,
                cache_on_device=False,
                non_blocking=False,
                decode_images=True,
            )
            sample = dict(prepared.sample)
            sample["query_positions"] = sample["vertices"]
            sample["query_is_exact"] = torch.ones(
                sample["vertices"].shape[0],
                dtype=torch.bool,
                device=resolved_device,
            )
            with torch.no_grad(), torch.autocast(
                device_type=resolved_device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                normalized_prediction = model(sample).predicted_laplacian.float()
            if not torch.isfinite(normalized_prediction).all():
                raise FloatingPointError(
                    f"Non-finite expanded prediction for {sample['sample_id']}."
                )
            raw_prediction = denormalize_laplacian_by_edge_scale(
                normalized_prediction, sample["local_edge_length"]
            )
            sample_id = str(sample["sample_id"])
            mesh_dir = output_dir / condition / sample_id
            metrics = reconstruct_and_evaluate(
                static,
                raw_prediction.detach().cpu(),
                mesh_dir,
                reconstruction_config,
                normalized_prediction=normalized_prediction.detach().cpu(),
                edge_scale_epsilon=float(
                    config.get("target_scaling", {}).get("epsilon", 1e-12)
                ),
            )
            record = {
                "sample_id": sample_id,
                "prediction_mean_magnitude": float(
                    torch.linalg.vector_norm(normalized_prediction, dim=-1).mean().item()
                ),
                "geometry": metrics["geometry"],
                "predicted_improves_over_initial": metrics[
                    "predicted_improves_over_coarse"
                ],
                "reconstruction": metrics["reconstruction"],
            }
            _write_json(mesh_dir / "expanded_metrics.json", record)
            records.append(record)
            print(
                f"  {sample_id}: chamfer initial="
                f"{record['geometry']['coarse']['chamfer']:.6g} predicted="
                f"{record['geometry']['predicted']['chamfer']:.6g}",
                flush=True,
            )
            del prepared, sample, normalized_prediction, raw_prediction
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()
        results[condition] = {
            "aggregate_mesh_mean": _aggregate(records),
            "per_mesh": records,
        }

    summary = {
        "checkpoint": str(run_dir / "best.pt"),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "expanded_manifest": str(expanded_manifest),
        "split": split,
        "visibility_source": (
            "each expanded sample's own vertices/faces rendered by the face-ID pass; "
            "no GT depth, GT visibility, or correspondence"
        ),
        "rgb_source": (
            "existing GT-mesh observation renders; renderer visibility is nevertheless "
            "computed on the expanded query mesh as required for inference"
        ),
        "oracle": {
            "available": False,
            "reason": (
                "The expanded manifest contains only a schema-required identity-placeholder "
                "Laplacian target and no valid expanded-graph oracle correspondence. It was "
                "not reported as an oracle."
            ),
        },
        "reconstruction_iterations": reconstruction_iters,
        "results": results,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _aggregate(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    initial = [record["geometry"]["coarse"] for record in records]
    predicted = [record["geometry"]["predicted"] for record in records]
    fields = (
        "chamfer",
        "point_to_surface_forward_mean",
        "point_to_surface_reverse_mean",
        "point_to_surface_bidirectional_mean",
        "normal_consistency",
        "bbox_diagonal_ratio_to_coarse",
    )
    return {
        "initial": {field: float(np.mean([row[field] for row in initial])) for field in fields},
        "predicted": {
            field: float(np.mean([row[field] for row in predicted])) for field in fields
        },
        "improves_over_initial_meshes": int(
            sum(bool(record["predicted_improves_over_initial"]) for record in records)
        ),
        "collapsed_or_exploded_meshes": int(
            sum(bool(row["collapsed_or_exploded"]) for row in predicted)
        ),
        "all_finite_meshes": int(sum(bool(row["all_finite"]) for row in predicted)),
        "prediction_mean_magnitude": float(
            np.mean([record["prediction_mean_magnitude"] for record in records])
        ),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Expanded-query renderer visibility evaluation",
        "",
        "Visibility was rasterized from each expanded mesh itself. No GT depth, GT "
        "visibility, or correspondence was used.",
        "",
        "| visibility | initial Chamfer | predicted Chamfer | bidirectional P2S | normal consistency | improves | collapse/explosion |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in VISIBILITY_CONDITIONS:
        aggregate = summary["results"][condition]["aggregate_mesh_mean"]
        lines.append(
            f"| {condition} | {aggregate['initial']['chamfer']:.6g} | "
            f"{aggregate['predicted']['chamfer']:.6g} | "
            f"{aggregate['predicted']['point_to_surface_bidirectional_mean']:.6g} | "
            f"{aggregate['predicted']['normal_consistency']:.4f} | "
            f"{aggregate['improves_over_initial_meshes']}/5 | "
            f"{aggregate['collapsed_or_exploded_meshes']}/5 |"
        )
    lines.extend(("", f"Expanded-graph oracle: unavailable — {summary['oracle']['reason']}", ""))
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(value), indent=2) + "\n", encoding="utf-8")


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value
