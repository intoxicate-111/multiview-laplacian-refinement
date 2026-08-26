#!/usr/bin/env python3
from __future__ import annotations

"""Blackwell runtime preflight for old-domain native-1920 Arm B and E."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import (
    _build_model,
    _prepare_item_for_use,
    _prepare_object_static,
    _recovery_aware_geometry_settings,
    _recovery_refine_loss_with_audit,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value.get("experiment_config", value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_gradient(model: torch.nn.Module) -> dict[str, Any]:
    named = list(model.named_parameters())
    present = [(name, value.grad) for name, value in named if value.grad is not None]
    norm = math.sqrt(
        sum(float(gradient.detach().float().square().sum().cpu()) for _, gradient in present)
    )
    return {
        "parameter_tensors": len(named),
        "gradient_tensors": len(present),
        "all_present": len(present) == len(named),
        "all_finite": all(bool(torch.isfinite(gradient).all()) for _, gradient in present),
        "norm": norm,
        "nonzero": sum(int(torch.count_nonzero(gradient).cpu()) for _, gradient in present),
    }


def choose_largest_training_sample(manifest: Path) -> tuple[dict[str, Any], int]:
    dataset = PreparedMeshDataset.from_manifest(manifest, "train")
    selected: dict[str, Any] | None = None
    selected_vertices = -1
    for index in range(len(dataset)):
        sample = dataset.load_static(index)
        count = int(sample["vertices"].shape[0])
        if count > selected_vertices:
            selected = sample
            selected_vertices = count
    if selected is None:
        raise RuntimeError("Training split is empty")
    return selected, selected_vertices


def audit_config(config: dict[str, Any], arm: str) -> dict[str, bool]:
    metadata = config["experiment_metadata"]
    common = {
        "dataset": config["dataset"]["expected_split_counts"]
        == {"train": 200, "validation": 25, "test": 25},
        "native_1920": metadata["input_resolution"] == 1920,
        "views_28": metadata["views"] == 28,
        "no_confidence": config["confidence"]["enabled"] is False,
        "view_chunk_4": config["image_encoder"]["view_chunk_size"] == 4,
        "gradient_checkpointing": config["image_encoder"]["gradient_checkpointing"] is True,
        "world_size_4": metadata["distributed_world_size"] == 4,
        "accumulation_2": config["multi_object_training"]["gradient_accumulation_meshes"] == 2,
        "effective_batch_8": metadata["effective_global_batch_meshes"] == 8,
        "steps_20000": config["multi_object_training"]["max_optimizer_steps"] == 20000,
        "from_scratch": metadata["initialization"] == "from_scratch",
        "raw_target": config["target_mode"] == "raw_laplacian",
    }
    if arm == "B":
        recovery = config["training"]["recovery_aware_geometry_loss"]
        common |= {
            "huber": config["training"]["loss"] == "huber"
            and config["training"]["huber_delta"] == 0.01,
            "beta_001": recovery["enabled"] is True and recovery["beta"] == 0.01,
            "lambda_b_001": recovery["lambda"] == 0.01,
            "pcg_exact_recipe": recovery["maximum_iterations"] == 256
            and recovery["tolerance"] == 1e-4
            and recovery["compute_dtype"] == "float32",
            "uniform_operator": config["recovery"]["operator_type"]
            == "uniform_random_walk_current_graph",
        }
    else:
        common |= {
            "direct_semantics": config["prediction_semantics"]
            == "direct_vertex_displacement",
            "mse_only": config["training"]["loss"] == "mse",
            "no_recovery": config["training"]["recovery_aware_geometry_loss"]["enabled"]
            is False,
            "direct_addition": config["recovery"]["mode"] == "direct_vertex_addition",
        }
    return common


def run_arm(
    arm: str,
    config: dict[str, Any],
    static: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = _build_model(config, None, False).to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    prepared = _prepare_item_for_use(
        _prepare_object_static(static, config),
        config,
        device,
        cache_on_device=False,
        decode_images=True,
    )
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        output = model(prepared.sample)
    prediction = output.predicted_laplacian.float()
    if arm == "B":
        lap_loss = F.huber_loss(
            prediction,
            prepared.training_target.float(),
            reduction="mean",
            delta=0.01,
        )
        settings = _recovery_aware_geometry_settings(config)
        vertex_loss, recovered, solve = _recovery_refine_loss_with_audit(
            prediction, prepared, settings
        )
        objective = lap_loss + 0.01 * vertex_loss
        solve_payload: dict[str, Any] | None = {
            "iterations": int(solve.iterations),
            "converged": bool(solve.converged),
            "relative_residual": float(solve.relative_residual),
        }
    else:
        target = prepared.training_target.float()
        objective = (prediction - target).square().sum(dim=-1).mean()
        recovered = prepared.sample["vertices"].float() + prediction
        solve_payload = None
    objective.backward()
    gradients = parameter_gradient(model)
    clean = prepared.clean_vertices
    if clean is None and arm == "E":
        clean = prepared.sample["vertices"].float() + prepared.training_target.float()
    if clean is None:
        raise RuntimeError("Preflight requires loss-side clean vertices")
    result = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "sample_id": str(prepared.sample["sample_id"]),
        "vertices": int(prepared.sample["vertices"].shape[0]),
        "views": int(prepared.sample["num_views"]),
        "objective": float(objective.detach().cpu()),
        "recovered_vertex_rms": float(
            torch.sqrt((recovered.detach() - clean).square().sum(dim=-1).mean()).cpu()
        ),
        "gradients": gradients,
        "solve": solve_payload,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "model_input_has_gt_or_clean": any(
            "clean" in key.lower() or key.startswith("gt_") or key == "target_positions"
            for key in prepared.sample
        ),
    }
    del model, prepared, output, prediction, objective, recovered
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-b-config", required=True, type=Path)
    parser.add_argument("--arm-e-config", required=True, type=Path)
    parser.add_argument("--data-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    data_audit = json.loads(args.data_audit.read_text(encoding="utf-8"))
    if data_audit.get("contract_audit") is not True:
        raise RuntimeError("Data preflight did not pass")
    configs = {
        "B": read_json(args.arm_b_config.resolve()),
        "E": read_json(args.arm_e_config.resolve()),
    }
    config_checks = {arm: audit_config(config, arm) for arm, config in configs.items()}
    static, largest_count = choose_largest_training_sample(args.manifest.resolve())
    runtime = {
        arm: run_arm(arm, config, static, device) for arm, config in configs.items()
    }
    passed = bool(
        all(all(checks.values()) for checks in config_checks.values())
        and all(item["parameter_count"] == 826115 for item in runtime.values())
        and all(item["gradients"]["all_present"] for item in runtime.values())
        and all(item["gradients"]["all_finite"] for item in runtime.values())
        and all(item["gradients"]["norm"] > 0 for item in runtime.values())
        and all(not item["model_input_has_gt_or_clean"] for item in runtime.values())
        and runtime["B"]["solve"] is not None
        and runtime["B"]["solve"]["converged"]
    )
    payload = {
        "contract_audit": passed,
        "scope": "largest_training_mesh_native1920_forward_backward_no_optimizer_step",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "data_audit_sha256": sha256_file(args.data_audit.resolve()),
        "largest_training_vertex_count": largest_count,
        "gpu": torch.cuda.get_device_name(device),
        "config_checks": config_checks,
        "runtime": runtime,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
