from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .multi_dataset import PreparedMeshDataset, validate_disjoint_splits
from .multi_trainer import (
    _amp_settings,
    _prepare_item_for_use,
    _prepare_object_static,
    train_multi_object,
)
from .query_training import apply_query_augmentation, query_augmentation_settings
from .vertex_sampling import (
    HIGH_LAPLACIAN_MIXTURE,
    LAPLACIAN_MAGNITUDE_MIXTURE,
    sample_training_vertices,
    vertex_sampling_settings,
)


ARM_ORDER = (
    "canonical_0001",
    "exact_0000",
    "support_0010",
    "support_0030",
    "support_0100",
    "importance_0001",
)
GEOMETRY_AWARE_EXTRA_ARMS = ("strong_importance_0001", "smooth_importance_0001")
ORACLE_RESIDUAL_ARMS = ("oracle_expert_e0", "oracle_expert_e1")
IMAGE_RESOLUTION_ARMS = ("image_resolution_f0", "image_resolution_f1")
ARM_CHOICES = (
    ARM_ORDER + GEOMETRY_AWARE_EXTRA_ARMS + ORACLE_RESIDUAL_ARMS + IMAGE_RESOLUTION_ARMS
)
QUERY_SETS = (
    "exact",
    "near_0001",
    "moderate_0010",
    "expanded_0030",
    "large_0100",
    "expanded_like",
)
FIXED_SUPPORTS = {
    "exact": 0.0,
    "near_0001": 0.001,
    "moderate_0010": 0.01,
    "expanded_0030": 0.03,
    "large_0100": 0.10,
}
GROUPS = ("all", "lowest_10", "smooth_bottom_90", "high_top_10", "high_top_1")


def arm_config(
    base: Mapping[str, Any], arm: str, *, max_optimizer_steps: int
) -> dict[str, Any]:
    if arm not in ARM_CHOICES:
        raise ValueError(f"Unknown screening arm: {arm!r}.")
    result = copy.deepcopy(dict(base))
    result["query_training"]["apply_to_validation"] = False
    result["multi_object_training"]["max_optimizer_steps"] = int(max_optimizer_steps)
    result["multi_object_training"]["epochs"] = max(
        int(result["multi_object_training"].get("epochs", 1)), max_optimizer_steps
    )
    result["multi_object_training"]["checkpoint_every_epochs"] = 0
    result["multi_object_training"]["checkpoint_epochs"] = []
    result["training"]["vertex_sampling"] = {"mode": "full"}
    support_by_arm = {
        "canonical_0001": 0.001,
        "exact_0000": 0.0,
        "support_0010": 0.01,
        "support_0030": 0.03,
        "support_0100": 0.10,
        "importance_0001": 0.001,
        "strong_importance_0001": 0.001,
        "smooth_importance_0001": 0.001,
        "oracle_expert_e0": 0.001,
        "oracle_expert_e1": 0.001,
        "image_resolution_f0": 0.001,
        "image_resolution_f1": 0.001,
    }
    support = support_by_arm[arm]
    result["query_training"]["enabled"] = support > 0
    if support > 0:
        result["query_training"]["normal_std_h"] = 0.3 * support
        result["query_training"]["tangent_std_h"] = 0.3 * support
        result["query_training"]["max_offset_h"] = support
    if arm == "importance_0001":
        result["training"]["vertex_sampling"] = {
            "mode": HIGH_LAPLACIAN_MIXTURE,
            "sample_count_ratio": 1.0,
            "uniform_fraction": 0.5,
            "top_10_fraction": 0.25,
            "top_1_to_10_fraction": 0.25,
        }
    elif arm == "strong_importance_0001":
        result["training"]["vertex_sampling"] = {
            "mode": LAPLACIAN_MAGNITUDE_MIXTURE,
            "sample_count_ratio": 1.0,
            "uniform_fraction": 0.25,
            "top_10_fraction": 0.50,
            "top_1_to_10_fraction": 0.0,
            "top_1_fraction": 0.25,
            "bottom_90_fraction": 0.0,
        }
    elif arm == "smooth_importance_0001":
        result["training"]["vertex_sampling"] = {
            "mode": LAPLACIAN_MAGNITUDE_MIXTURE,
            "sample_count_ratio": 1.0,
            "uniform_fraction": 0.50,
            "top_10_fraction": 0.0,
            "top_1_to_10_fraction": 0.0,
            "top_1_fraction": 0.0,
            "bottom_90_fraction": 0.50,
        }
    if arm == "oracle_expert_e1":
        result.setdefault("model", {})["oracle_residual_expert"] = {
            "enabled": True,
            "hidden_dim": 32,
            "top_fraction": 0.10,
            "gate": "clean_gt_normalized_laplacian_top10_per_mesh",
        }
    else:
        result.setdefault("model", {}).pop("oracle_residual_expert", None)
    if arm in IMAGE_RESOLUTION_ARMS:
        result.setdefault("image_encoder", {})["second_stride"] = (
            2 if arm == "image_resolution_f0" else 1
        )
    result["screening"] = {
        "arm": arm,
        "train_query_support_h": support,
        "short_run": True,
        "shared_seed": int(result.get("seed", 7)),
        "exact_query_validation": True,
        "resume_checkpoint": None,
    }
    return result


def run_screening_arm(
    manifest_path: str | Path,
    base_config_path: str | Path,
    output_root: str | Path,
    arm: str,
    *,
    max_optimizer_steps: int = 1000,
    device: str = "cuda",
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    base_config_path = Path(base_config_path).resolve()
    output_root = Path(output_root).resolve()
    arm_dir = output_root / "arms" / arm
    if arm_dir.exists() and any(arm_dir.iterdir()):
        raise FileExistsError(f"Screening arm directory is not empty: {arm_dir}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    base = _read_json(base_config_path)
    config = arm_config(base, arm, max_optimizer_steps=max_optimizer_steps)
    _write_json(arm_dir / "config.json", config)
    shutil.copy2(manifest_path, arm_dir / "dataset_manifest.json")

    train_dataset = PreparedMeshDataset.from_manifest(manifest_path, "train")
    validation_dataset = PreparedMeshDataset.from_manifest(manifest_path, "validation")
    validate_disjoint_splits(train_dataset, validation_dataset)
    result = train_multi_object(
        train_dataset,
        validation_dataset,
        config,
        output_dir=arm_dir,
        device_override=device,
        resume_checkpoint=None,
    )
    if result.optimizer_steps != max_optimizer_steps:
        raise RuntimeError(
            f"Arm {arm} completed {result.optimizer_steps} steps, expected {max_optimizer_steps}."
        )

    query_distribution = replay_train_query_distribution(
        train_dataset, config, epochs=result.completed_epochs
    )
    _write_json(arm_dir / "train_query_distribution.json", query_distribution["summary"])
    _write_csv(arm_dir / "train_query_histogram.csv", query_distribution["histogram"])
    exposure = replay_training_exposure(
        train_dataset, config, epochs=result.completed_epochs
    )
    _write_json(arm_dir / "training_vertex_exposure.json", exposure)
    evaluation = evaluate_fixed_query_sets(
        result.model,
        validation_dataset,
        config,
        arm_dir,
        device=torch.device(device),
    )
    summary = {
        "arm": arm,
        "optimizer_steps": result.optimizer_steps,
        "completed_epochs": result.completed_epochs,
        "best_epoch": result.best_epoch,
        "best_selection_loss": result.best_selection_loss,
        "final_exact_validation_loss": result.final_validation_loss,
        "train_query_distribution": query_distribution["summary"],
        "training_vertex_exposure": exposure,
        "evaluation": evaluation,
    }
    _write_json(arm_dir / "screening_summary.json", summary)
    return summary


def evaluate_fixed_query_sets(
    model: torch.nn.Module,
    validation_dataset: PreparedMeshDataset,
    config: Mapping[str, Any],
    arm_dir: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    seed = int(config.get("seed", 7))
    per_mesh_rows: list[dict[str, Any]] = []
    aggregate_payload: dict[tuple[str, str], list[tuple[torch.Tensor, torch.Tensor]]] = {}
    predictions_dir = arm_dir / "fixed_query_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for index in range(len(validation_dataset)):
        static = validation_dataset.load_static(index)
        prepared_cpu = _prepare_object_static(static, config)
        prepared_device = _prepare_item_for_use(
            prepared_cpu,
            config,
            device,
            cache_on_device=False,
            decode_images=True,
        )
        sample_id = str(prepared_cpu.sample["sample_id"])
        query_positions = fixed_query_positions(prepared_cpu.sample, seed=seed)
        group_masks = target_group_masks(
            prepared_cpu.training_target, prepared_cpu.sample["valid_scale_mask"]
        )
        for query_name in QUERY_SETS:
            query_sample = dict(prepared_device.sample)
            query_sample["query_positions"] = query_positions[query_name]["positions"].to(
                device
            )
            query_sample["query_is_exact"] = (
                query_positions[query_name]["ratio"] == 0
            ).to(device)
            prepared = type(prepared_device)(
                sample=query_sample,
                training_target=prepared_device.training_target,
                clipped_target_vertices=prepared_device.clipped_target_vertices,
                raw_target=prepared_device.raw_target,
                face_count=prepared_device.face_count,
                image_decode_resize_seconds=prepared_device.image_decode_resize_seconds,
                decoded_image_bytes=prepared_device.decoded_image_bytes,
                used_view_count=prepared_device.used_view_count,
            )
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                prediction = model(prepared.sample).predicted_laplacian.float()
            prediction_cpu = prediction.detach().cpu()
            target_cpu = prepared_cpu.training_target.float().cpu()
            ratio = query_positions[query_name]["ratio"].cpu()
            safe_id = _safe_token(sample_id)
            np.savez_compressed(
                predictions_dir / f"{safe_id}__{query_name}.npz",
                prediction=prediction_cpu.numpy(),
                target=target_cpu.numpy(),
                query_displacement_over_h=ratio.numpy(),
                endpoint_error=torch.linalg.vector_norm(
                    prediction_cpu - target_cpu, dim=-1
                ).numpy(),
                target_magnitude=torch.linalg.vector_norm(target_cpu, dim=-1).numpy(),
                prediction_magnitude=torch.linalg.vector_norm(
                    prediction_cpu, dim=-1
                ).numpy(),
            )
            for group_name, mask in group_masks.items():
                row = _group_metrics(prediction_cpu, target_cpu, mask)
                row.update(
                    {
                        "sample_id": sample_id,
                        "query_set": query_name,
                        "group": group_name,
                        "mean_query_displacement_over_h": float(ratio[mask].mean().item()),
                    }
                )
                per_mesh_rows.append(row)
                aggregate_payload.setdefault((query_name, group_name), []).append(
                    (prediction_cpu[mask], target_cpu[mask])
                )
    aggregate_rows: list[dict[str, Any]] = []
    for query_name in QUERY_SETS:
        for group_name in GROUPS:
            pairs = aggregate_payload[(query_name, group_name)]
            prediction = torch.cat([pair[0] for pair in pairs])
            target = torch.cat([pair[1] for pair in pairs])
            row = _group_metrics(
                prediction, target, torch.ones(len(target), dtype=torch.bool)
            )
            row.update({"query_set": query_name, "group": group_name})
            aggregate_rows.append(row)
    _write_csv(arm_dir / "per_mesh_per_group.csv", per_mesh_rows)
    _write_csv(arm_dir / "aggregate_per_group.csv", aggregate_rows)
    return {
        "per_mesh_csv": str(arm_dir / "per_mesh_per_group.csv"),
        "aggregate_csv": str(arm_dir / "aggregate_per_group.csv"),
        "prediction_directory": str(predictions_dir),
        "aggregate": aggregate_rows,
    }


def fixed_query_positions(
    sample: Mapping[str, Any], *, seed: int
) -> dict[str, dict[str, torch.Tensor]]:
    vertices = sample["vertices"].float().cpu()
    normals = F.normalize(sample["vertex_normals"].float().cpu(), dim=-1, eps=1e-8)
    local_h = sample["local_edge_length"].float().cpu()
    sample_id = str(sample["sample_id"])
    generator = torch.Generator().manual_seed(_stable_seed("eval_query", sample_id, seed))
    tangent = torch.randn(vertices.shape, generator=generator)
    tangent = tangent - (tangent * normals).sum(dim=-1, keepdim=True) * normals
    tangent = F.normalize(tangent, dim=-1, eps=1e-8)
    normal_component = torch.randn((len(vertices), 1), generator=generator)
    tangent_component = torch.randn((len(vertices), 1), generator=generator)
    direction = F.normalize(
        normal_component * normals + tangent_component * tangent, dim=-1, eps=1e-8
    )
    invalid_direction = torch.linalg.vector_norm(direction, dim=-1) < 0.5
    direction[invalid_direction] = normals[invalid_direction]
    result: dict[str, dict[str, torch.Tensor]] = {}
    for name, support in FIXED_SUPPORTS.items():
        ratio = torch.full((len(vertices),), float(support), dtype=torch.float32)
        result[name] = {
            "positions": vertices + direction * (ratio * local_h).unsqueeze(-1),
            "ratio": ratio,
        }

    count = len(vertices)
    quantiles = (torch.arange(count, dtype=torch.float64) + 0.5) / count
    standard_normal = torch.distributions.Normal(0.0, 1.0)
    median = 0.01157
    p95 = 0.08380
    sigma = math.log(p95 / median) / 1.6448536269514722
    ratio = torch.exp(math.log(median) + sigma * standard_normal.icdf(quantiles))
    ratio = ratio.clamp_max(0.15).float()
    permutation = torch.randperm(count, generator=generator)
    ratio = ratio.index_select(0, permutation)
    result["expanded_like"] = {
        "positions": vertices + direction * (ratio * local_h).unsqueeze(-1),
        "ratio": ratio,
    }
    return result


def target_group_masks(
    target: torch.Tensor, valid_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    target = target.float().cpu()
    valid = valid_mask.to(dtype=torch.bool).cpu()
    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    magnitude = torch.linalg.vector_norm(target, dim=-1)
    order = valid_indices[torch.argsort(magnitude[valid_indices])]
    count = int(order.numel())
    low_10_count = max(1, int(round(0.10 * count)))
    top_10_count = max(1, int(round(0.10 * count)))
    top_1_count = max(1, int(round(0.01 * count)))

    def mask(indices: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(len(target), dtype=torch.bool)
        result[indices] = True
        return result

    return {
        "all": valid,
        "lowest_10": mask(order[:low_10_count]),
        "smooth_bottom_90": mask(order[: count - top_10_count]),
        "high_top_10": mask(order[count - top_10_count :]),
        "high_top_1": mask(order[count - top_1_count :]),
    }


def replay_train_query_distribution(
    train_dataset: PreparedMeshDataset,
    config: Mapping[str, Any],
    *,
    epochs: int,
    histogram_bins: int = 1000,
) -> dict[str, Any]:
    settings = query_augmentation_settings(config)
    support = settings.max_offset_h if settings.enabled else 0.001
    histogram_max = support + max(1e-7, support * 2e-4)
    counts = np.zeros(histogram_bins, dtype=np.int64)
    total_count = 0
    total_sum = 0.0
    maximum = 0.0
    samples = [
        _prepare_object_static(train_dataset.load_static(i), config).sample
        for i in range(len(train_dataset))
    ]
    for epoch in range(1, epochs + 1):
        for sample in samples:
            if settings.enabled:
                augmented = apply_query_augmentation(
                    sample,
                    settings,
                    base_seed=int(config.get("seed", 7)),
                    epoch=epoch,
                )
                offset = augmented["query_positions"] - sample["vertices"]
                local_h = sample["local_edge_length"].float()
                valid = sample["valid_scale_mask"] & (local_h > 0)
                ratio = (
                    torch.linalg.vector_norm(offset.float(), dim=-1)[valid] / local_h[valid]
                ).double().numpy()
            else:
                ratio = np.zeros(int(sample["valid_scale_mask"].sum().item()), dtype=np.float64)
            hist, _ = np.histogram(
                ratio, bins=histogram_bins, range=(0.0, histogram_max)
            )
            counts += hist
            total_count += int(ratio.size)
            total_sum += float(ratio.sum())
            maximum = max(maximum, float(ratio.max(initial=0.0)))
    edges = np.linspace(0.0, histogram_max, histogram_bins + 1)
    summary = {
        "sample_count": total_count,
        "epochs": epochs,
        "support_h": 0.0 if not settings.enabled else settings.max_offset_h,
        "mean": total_sum / max(total_count, 1),
        "median": _histogram_quantile(counts, edges, 0.50, maximum),
        "p90": _histogram_quantile(counts, edges, 0.90, maximum),
        "p95": _histogram_quantile(counts, edges, 0.95, maximum),
        "p99": _histogram_quantile(counts, edges, 0.99, maximum),
        "max": maximum,
    }
    histogram = [
        {
            "bin": index,
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": int(value),
        }
        for index, value in enumerate(counts)
    ]
    return {"summary": summary, "histogram": histogram}


def replay_training_exposure(
    train_dataset: PreparedMeshDataset,
    config: Mapping[str, Any],
    *,
    epochs: int,
) -> dict[str, Any]:
    settings = vertex_sampling_settings(config)
    prepared = [
        _prepare_object_static(train_dataset.load_static(i), config)
        for i in range(len(train_dataset))
    ]
    counts = {
        "all": 0,
        "lowest_10": 0,
        "smooth_bottom_90": 0,
        "high_top_10": 0,
        "high_top_1_to_10": 0,
        "high_top_1": 0,
    }
    selected_rows = 0
    for epoch in range(1, epochs + 1):
        for item in prepared:
            selection = sample_training_vertices(
                item.training_target,
                item.sample["valid_scale_mask"],
                settings,
                sample_id=str(item.sample["sample_id"]),
                base_seed=int(config.get("seed", 7)),
                epoch=epoch,
            )
            indices = selection.indices
            if indices is None:
                indices = torch.nonzero(
                    item.sample["valid_scale_mask"], as_tuple=False
                ).squeeze(-1)
            groups = target_group_masks(
                item.training_target, item.sample["valid_scale_mask"]
            )
            groups["high_top_1_to_10"] = (
                groups["high_top_10"] & ~groups["high_top_1"]
            )
            selected_rows += int(indices.numel())
            counts["all"] += int(indices.numel())
            for name in counts:
                if name != "all":
                    counts[name] += int(groups[name][indices.cpu()].sum().item())
    fractions = {
        name: value / max(selected_rows, 1) for name, value in counts.items()
    }
    expected_uniform = {
        "all": 1.0,
        "lowest_10": 0.10,
        "smooth_bottom_90": 0.90,
        "high_top_10": 0.10,
        "high_top_1_to_10": 0.09,
        "high_top_1": 0.01,
    }
    return {
        "mode": settings.mode,
        "epochs": epochs,
        "selected_rows": selected_rows,
        "group_draw_counts": counts,
        "group_draw_fractions": fractions,
        "exposure_multiplier_vs_uniform": {
            name: fractions[name] / expected_uniform[name] for name in counts
        },
    }


def analyze_screening(output_root: str | Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    summaries = {
        arm: _read_json(output_root / "arms" / arm / "screening_summary.json")
        for arm in ARM_ORDER
    }
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    a_rows = paired_exact_query_differences(output_root, analysis_dir)
    b_rows = _collect_matrix_rows(
        output_root,
        (
            "exact_0000",
            "canonical_0001",
            "support_0010",
            "support_0030",
            "support_0100",
        ),
    )
    c_rows = _collect_matrix_rows(
        output_root, ("canonical_0001", "importance_0001"), query_sets=("exact",)
    )
    d_rows = [
        row
        for row in b_rows
        if row["group"] in {"smooth_bottom_90", "high_top_10", "high_top_1"}
    ]
    _write_csv(analysis_dir / "B_train_support_x_eval_query.csv", b_rows)
    _write_csv(analysis_dir / "C_importance_groups.csv", c_rows)
    _write_csv(analysis_dir / "D_displacement_x_laplacian_group.csv", d_rows)

    a_summary = _hypothesis_a(output_root, a_rows)
    b_summary = _hypothesis_b(b_rows)
    c_summary = _hypothesis_c(c_rows, summaries)
    final = {
        "experiment": "sofa50_controlled_screening_1000step",
        "shared_optimizer_steps": summaries["canonical_0001"]["optimizer_steps"],
        "contract_audit": _contract_audit(output_root),
        "arms": {arm: summaries[arm]["train_query_distribution"] for arm in ARM_ORDER},
        "evaluation_query_distributions": _evaluation_query_distributions(output_root),
        "H1_tiny_query_perturbation": a_summary,
        "H2_query_support_mismatch": b_summary,
        "H3_high_laplacian_exposure": c_summary,
        "artifacts": {
            "A_paired_per_vertex_csv": str(analysis_dir / "A_paired_per_vertex.csv"),
            "B_matrix_csv": str(analysis_dir / "B_train_support_x_eval_query.csv"),
            "C_groups_csv": str(analysis_dir / "C_importance_groups.csv"),
            "D_cross_table_csv": str(analysis_dir / "D_displacement_x_laplacian_group.csv"),
        },
    }
    _write_json(analysis_dir / "summary.json", final)
    (analysis_dir / "REPORT.md").write_text(_report_text(final), encoding="utf-8")
    return final


def _contract_audit(output_root: Path) -> dict[str, Any]:
    configs = {arm: _read_json(output_root / "arms" / arm / "config.json") for arm in ARM_ORDER}
    reference = configs["canonical_0001"]
    invariant_fields = (
        "seed",
        "input_mode",
        "target_mode",
        "target_semantics",
        "target_scaling",
        "renderer_visibility",
        "image_encoder",
        "model",
        "confidence",
        "data_loading",
        "recovery",
    )
    invariants_equal = all(
        config.get(field) == reference.get(field)
        for config in configs.values()
        for field in invariant_fields
    )
    histories = {
        arm: _read_json_list(output_root / "arms" / arm / "training_history.json")
        for arm in ARM_ORDER
    }
    summaries = {
        arm: _read_json(output_root / "arms" / arm / "screening_summary.json")
        for arm in ARM_ORDER
    }
    return {
        "invariant_config_fields_equal": invariants_equal,
        "invariant_config_fields": list(invariant_fields),
        "all_seed_7": all(int(config["seed"]) == 7 for config in configs.values()),
        "all_exact_validation": all(
            config["query_training"]["apply_to_validation"] is False
            for config in configs.values()
        ),
        "all_fresh_start": all(
            config["screening"]["resume_checkpoint"] is None
            for config in configs.values()
        ),
        "all_1000_optimizer_steps": all(
            int(summary["optimizer_steps"]) == 1000 for summary in summaries.values()
        ),
        "history_records_per_arm": {arm: len(history) for arm, history in histories.items()},
        "fixed_prediction_files_per_arm": {
            arm: len(list((output_root / "arms" / arm / "fixed_query_predictions").glob("*.npz")))
            for arm in ARM_ORDER
        },
    }


def _evaluation_query_distributions(output_root: Path) -> dict[str, dict[str, float]]:
    prediction_dir = output_root / "arms" / "canonical_0001" / "fixed_query_predictions"
    result: dict[str, dict[str, float]] = {}
    for query_name in QUERY_SETS:
        values = np.concatenate(
            [
                np.load(path)["query_displacement_over_h"]
                for path in sorted(prediction_dir.glob(f"*__{query_name}.npz"))
            ]
        )
        result[query_name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max(initial=0.0)),
        }
    return result


def paired_exact_query_differences(
    output_root: Path, analysis_dir: Path
) -> list[dict[str, Any]]:
    on_dir = output_root / "arms" / "canonical_0001" / "fixed_query_predictions"
    off_dir = output_root / "arms" / "exact_0000" / "fixed_query_predictions"
    rows: list[dict[str, Any]] = []
    for on_path in sorted(on_dir.glob("*__exact.npz")):
        off_path = off_dir / on_path.name
        on = np.load(on_path)
        off = np.load(off_path)
        if not np.array_equal(on["target"], off["target"]):
            raise ValueError(f"A-arm targets differ for {on_path.name}.")
        sample_id = on_path.name.removesuffix("__exact.npz")
        error_on = np.linalg.norm(on["prediction"] - on["target"], axis=1)
        error_off = np.linalg.norm(off["prediction"] - off["target"], axis=1)
        pred_mag_on = np.linalg.norm(on["prediction"], axis=1)
        pred_mag_off = np.linalg.norm(off["prediction"], axis=1)
        target_mag = np.linalg.norm(on["target"], axis=1)
        np.savez_compressed(
            analysis_dir / f"A_paired_{sample_id}.npz",
            error_on=error_on,
            error_off=error_off,
            error_difference_on_minus_off=error_on - error_off,
            prediction_magnitude_on=pred_mag_on,
            prediction_magnitude_off=pred_mag_off,
            target_magnitude=target_mag,
        )
        rows.extend(
            {
                "sample_id": sample_id,
                "vertex_index": index,
                "target_magnitude": float(target_mag[index]),
                "error_on": float(error_on[index]),
                "error_off": float(error_off[index]),
                "error_difference_on_minus_off": float(error_on[index] - error_off[index]),
                "prediction_magnitude_on": float(pred_mag_on[index]),
                "prediction_magnitude_off": float(pred_mag_off[index]),
            }
            for index in range(len(error_on))
        )
    _write_csv(analysis_dir / "A_paired_per_vertex.csv", rows)
    return rows


def _collect_matrix_rows(
    output_root: Path,
    arms: Sequence[str],
    *,
    query_sets: Sequence[str] = QUERY_SETS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        source = _read_csv(output_root / "arms" / arm / "aggregate_per_group.csv")
        config = _read_json(output_root / "arms" / arm / "config.json")
        support = float(config["screening"]["train_query_support_h"])
        for row in source:
            if row["query_set"] in query_sets:
                rows.append({"arm": arm, "train_support_h": support, **row})
    return rows


def _hypothesis_a(output_root: Path, paired: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    on = _aggregate_row(output_root, "canonical_0001", "exact", "all")
    off = _aggregate_row(output_root, "exact_0000", "exact", "all")
    on_low = _aggregate_row(output_root, "canonical_0001", "exact", "lowest_10")
    off_low = _aggregate_row(output_root, "exact_0000", "exact", "lowest_10")
    vee_change = _relative_change(on["endpoint_error"], off["endpoint_error"])
    low_change = _relative_change(
        on_low["mean_prediction_magnitude"], off_low["mean_prediction_magnitude"]
    )
    if vee_change > 0.05 and low_change > 0.05:
        verdict = "Supported"
    elif abs(vee_change) < 0.02 and abs(low_change) < 0.02:
        verdict = "Not supported"
    else:
        verdict = "Inconclusive"
    paired_difference = float(
        np.mean([float(row["error_difference_on_minus_off"]) for row in paired])
    )
    return {
        "verdict": verdict,
        "exact_endpoint_error_on": on["endpoint_error"],
        "exact_endpoint_error_off": off["endpoint_error"],
        "relative_endpoint_change_on_vs_off": vee_change,
        "lowest10_predicted_magnitude_on": on_low["mean_prediction_magnitude"],
        "lowest10_predicted_magnitude_off": off_low["mean_prediction_magnitude"],
        "relative_lowest10_prediction_magnitude_change": low_change,
        "paired_mean_error_difference_on_minus_off": paired_difference,
        "global_cosine_on": on["global_cosine"],
        "global_cosine_off": off["global_cosine"],
        "prediction_to_gt_norm_ratio_on": on["prediction_to_gt_norm_ratio"],
        "prediction_to_gt_norm_ratio_off": off["prediction_to_gt_norm_ratio"],
    }


def _hypothesis_b(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def get(arm: str, query: str, group: str = "all") -> Mapping[str, Any]:
        return next(
            row
            for row in rows
            if row["arm"] == arm and row["query_set"] == query and row["group"] == group
        )

    current_exact = get("canonical_0001", "exact")
    current_large = get("canonical_0001", "large_0100")
    current_like = get("canonical_0001", "expanded_like")
    candidates = [
        get(arm, "expanded_like")
        for arm in ("exact_0000", "canonical_0001", "support_0010", "support_0030", "support_0100")
    ]
    best = min(candidates, key=lambda row: float(row["endpoint_error"]))
    best_exact = get(str(best["arm"]), "exact")
    improvement = -_relative_change(best["endpoint_error"], current_like["endpoint_error"])
    exact_tradeoff = _relative_change(best_exact["endpoint_error"], current_exact["endpoint_error"])
    displacement_failure = _relative_change(
        current_large["endpoint_error"], current_exact["endpoint_error"]
    )
    verdict = (
        "Supported"
        if displacement_failure > 0.05 and improvement > 0.05
        else "Not supported"
        if abs(displacement_failure) < 0.02 and improvement < 0.02
        else "Inconclusive"
    )
    return {
        "verdict": verdict,
        "current_exact_endpoint_error": current_exact["endpoint_error"],
        "current_large_endpoint_error": current_large["endpoint_error"],
        "current_large_relative_degradation": displacement_failure,
        "current_expanded_like_endpoint_error": current_like["endpoint_error"],
        "best_expanded_like_arm": best["arm"],
        "best_expanded_like_endpoint_error": best["endpoint_error"],
        "relative_expanded_like_improvement": improvement,
        "best_arm_exact_relative_tradeoff": exact_tradeoff,
    }


def _hypothesis_c(
    rows: Sequence[Mapping[str, Any]], summaries: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    def get(arm: str, group: str) -> Mapping[str, Any]:
        return next(row for row in rows if row["arm"] == arm and row["group"] == group)

    baseline_top10 = get("canonical_0001", "high_top_10")
    importance_top10 = get("importance_0001", "high_top_10")
    baseline_top1 = get("canonical_0001", "high_top_1")
    importance_top1 = get("importance_0001", "high_top_1")
    baseline_smooth = get("canonical_0001", "smooth_bottom_90")
    importance_smooth = get("importance_0001", "smooth_bottom_90")
    top10_improvement = -_relative_change(
        importance_top10["endpoint_error"], baseline_top10["endpoint_error"]
    )
    top1_improvement = -_relative_change(
        importance_top1["endpoint_error"], baseline_top1["endpoint_error"]
    )
    smooth_change = _relative_change(
        importance_smooth["endpoint_error"], baseline_smooth["endpoint_error"]
    )
    verdict = (
        "Supported"
        if top10_improvement > 0.05 and top1_improvement > 0.05 and smooth_change < 0.05
        else "Not supported"
        if top10_improvement < 0.02 and top1_improvement < 0.02
        else "Inconclusive"
    )
    return {
        "verdict": verdict,
        "baseline_top10_endpoint_error": baseline_top10["endpoint_error"],
        "importance_top10_endpoint_error": importance_top10["endpoint_error"],
        "top10_relative_improvement": top10_improvement,
        "baseline_top1_endpoint_error": baseline_top1["endpoint_error"],
        "importance_top1_endpoint_error": importance_top1["endpoint_error"],
        "top1_relative_improvement": top1_improvement,
        "baseline_top10_relative_error": baseline_top10["relative_error_of_means"],
        "importance_top10_relative_error": importance_top10["relative_error_of_means"],
        "baseline_top1_relative_error": baseline_top1["relative_error_of_means"],
        "importance_top1_relative_error": importance_top1["relative_error_of_means"],
        "smooth_relative_endpoint_change": smooth_change,
        "baseline_exposure": summaries["canonical_0001"]["training_vertex_exposure"],
        "importance_exposure": summaries["importance_0001"]["training_vertex_exposure"],
    }


def _group_metrics(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, Any]:
    prediction = prediction[mask]
    target = target[mask]
    residual = prediction - target
    endpoint = torch.linalg.vector_norm(residual, dim=-1)
    prediction_magnitude = torch.linalg.vector_norm(prediction, dim=-1)
    target_magnitude = torch.linalg.vector_norm(target, dim=-1)
    per_vertex_cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    target_norm = torch.linalg.vector_norm(target)
    return {
        "vertex_count": int(len(target)),
        "training_huber_loss": float(
            F.huber_loss(prediction, target, delta=0.01, reduction="mean").item()
        ),
        "endpoint_error": float(endpoint.mean().item()),
        "global_cosine": float(
            F.cosine_similarity(
                prediction.reshape(1, -1), target.reshape(1, -1), dim=-1, eps=1e-8
            ).item()
        ),
        "mean_per_vertex_cosine": float(per_vertex_cosine.mean().item()),
        "prediction_to_gt_norm_ratio": float(
            (torch.linalg.vector_norm(prediction) / target_norm.clamp_min(1e-12)).item()
        ),
        "mean_prediction_magnitude": float(prediction_magnitude.mean().item()),
        "mean_target_magnitude": float(target_magnitude.mean().item()),
        "mean_magnitude_ratio": float(
            (prediction_magnitude.mean() / target_magnitude.mean().clamp_min(1e-12)).item()
        ),
        "relative_error_of_means": float(
            (endpoint.mean() / target_magnitude.mean().clamp_min(1e-12)).item()
        ),
        "mean_per_vertex_relative_error": float(
            (endpoint / target_magnitude.clamp_min(1e-8)).mean().item()
        ),
    }


def _aggregate_row(output_root: Path, arm: str, query: str, group: str) -> dict[str, Any]:
    rows = _read_csv(output_root / "arms" / arm / "aggregate_per_group.csv")
    return next(row for row in rows if row["query_set"] == query and row["group"] == group)


def _relative_change(value: Any, reference: Any) -> float:
    return (float(value) - float(reference)) / max(abs(float(reference)), 1e-12)


def _histogram_quantile(
    counts: np.ndarray, edges: np.ndarray, quantile: float, maximum: float
) -> float:
    total = int(counts.sum())
    if total == 0 or maximum == 0:
        return 0.0
    target = quantile * total
    cumulative = np.cumsum(counts)
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(counts) - 1)
    before = int(cumulative[index - 1]) if index > 0 else 0
    within = (target - before) / max(int(counts[index]), 1)
    return float(edges[index] + within * (edges[index + 1] - edges[index]))


def _stable_seed(namespace: str, sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{sample_id}".encode("utf-8")).digest()
    return int((int.from_bytes(digest[:8], "little") + int(seed)) % (2**63 - 1))


def _safe_token(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in value)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _read_json_list(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "train_support_h",
        "vertex_count",
        "training_huber_loss",
        "endpoint_error",
        "global_cosine",
        "mean_per_vertex_cosine",
        "prediction_to_gt_norm_ratio",
        "mean_prediction_magnitude",
        "mean_target_magnitude",
        "mean_magnitude_ratio",
        "relative_error_of_means",
        "mean_per_vertex_relative_error",
        "mean_query_displacement_over_h",
    }
    for row in rows:
        for key in numeric.intersection(row):
            row[key] = float(row[key])
    return rows


def _report_text(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Sofa50 controlled screening",
        "",
        f"Shared optimizer-step budget: {summary['shared_optimizer_steps']}",
        "",
    ]
    for key in (
        "H1_tiny_query_perturbation",
        "H2_query_support_mismatch",
        "H3_high_laplacian_exposure",
    ):
        payload = summary[key]
        lines.extend((f"## {key}", "", f"Verdict: **{payload['verdict']}**", "", "```json", json.dumps(payload, indent=2), "```", ""))
    return "\n".join(lines)
