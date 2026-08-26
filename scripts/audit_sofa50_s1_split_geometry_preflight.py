#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed real-model preflight for the S1 split-geometry hybrid."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from audit_sofa50_end_to_end_hybrid_preflight import _gradient_audit, _solver_audit
from mlr.learned_laplacian.canonical_experiment import (
    _exact_query_sample,
    _load_device_item,
)
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery_with_audit,
)
from mlr.learned_laplacian.model import LearnedLaplacianModel
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model


LAMBDA = 3e-2
TOLERANCE = 1e-8
MAXIMUM_ITERATIONS = 2048


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _nested(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        current = current[part]
    return current


def _norm(parameters: Iterable[torch.nn.Parameter]) -> dict[str, Any]:
    parameters = tuple(parameters)
    gradients = [parameter.grad for parameter in parameters]
    present = [gradient for gradient in gradients if gradient is not None]
    squared = sum(
        float(gradient.detach().float().square().sum().cpu())
        for gradient in present
    )
    return {
        "parameter_tensors": len(parameters),
        "gradient_tensors": len(present),
        "all_finite": bool(present)
        and all(bool(torch.isfinite(gradient).all()) for gradient in present),
        "norm": math.sqrt(squared),
        "nonzero_entries": sum(
            int(torch.count_nonzero(gradient).detach().cpu()) for gradient in present
        ),
    }


def _tensor_gradient(value: torch.Tensor) -> dict[str, Any]:
    gradient = value.grad
    if gradient is None:
        return {
            "present": False,
            "all_finite": False,
            "norm": 0.0,
            "nonzero_entries": 0,
        }
    return {
        "present": True,
        "all_finite": bool(torch.isfinite(gradient).all()),
        "norm": float(torch.linalg.vector_norm(gradient).detach().cpu()),
        "nonzero_entries": int(torch.count_nonzero(gradient).detach().cpu()),
    }


def _same_state(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def _storage_ids(parameters: Iterable[torch.nn.Parameter]) -> set[int]:
    return {
        int(parameter.untyped_storage().data_ptr()) for parameter in parameters
    }


def _critical_config_audit(
    s0: Mapping[str, Any], s1: Mapping[str, Any]
) -> dict[str, bool]:
    identical_paths = (
        "confidence",
        "data_loading",
        "dataset",
        "image_encoder",
        "input_mode",
        "local_query_jitter",
        "model.dropout",
        "model.geometry_mode",
        "model.hidden_dim",
        "model.num_graph_layers",
        "model.position_encoding",
        "model.recovery_lambda_head",
        "query_training",
        "recovery.V_direct",
        "recovery.adam_vertex_optimization",
        "recovery.additional_V_input_anchor",
        "recovery.confidence_weighting",
        "recovery.lambda",
        "recovery.objective",
        "recovery.operator",
        "recovery.robust_loss",
        "recovery.solver",
        "recovery.visibility_gate",
        "renderer_visibility",
        "seed",
        "target_definition",
        "target_mode",
        "target_scaling",
        "target_semantics",
        "training",
        "multi_object_training.early_stopping",
        "multi_object_training.epochs",
        "multi_object_training.max_optimizer_steps",
        "multi_object_training.profile_training",
        "multi_object_training.report_every_optimizer_steps",
        "multi_object_training.shuffle",
        "multi_object_training.validation_every_epochs",
    )
    return {
        path: _nested(s0, path) == _nested(s1, path) for path in identical_paths
    }


def _real_gradient_audit(
    model: LearnedLaplacianModel,
    dataset: PreparedMeshDataset,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    prepared = _load_device_item(dataset, 0, config, device)
    conditioned = _exact_query_sample(prepared.sample, device)
    forbidden_input_keys = sorted(
        key
        for key in conditioned
        if key
        in {
            "clean_reference_vertices",
            "clean_vertices",
            "target",
            "raw_laplacian_target",
            "delta_target",
            "V_clean",
        }
    )
    model.zero_grad(set_to_none=True)
    amp_enabled, amp_dtype = _amp_settings(config, device)
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        output = model(conditioned)
    delta = output.predicted_laplacian.float()
    displacement = output.direct_vertex_displacement_prediction
    if displacement is None:
        raise RuntimeError("S1 Direct tower emitted no displacement.")
    displacement = displacement.float()
    delta.retain_grad()
    displacement.retain_grad()
    v_direct = prepared.sample["vertices"].double() + displacement.double()
    v_direct.retain_grad()
    recovered, solver = differentiable_regularized_sparse_recovery_with_audit(
        delta.double(),
        v_direct,
        prepared.sample["edge_index"],
        prepared.sample["vertex_degree"].double(),
        regularization=LAMBDA,
        maximum_iterations=MAXIMUM_ITERATIONS,
        tolerance=TOLERANCE,
    )
    clean = prepared.clean_vertices
    if clean is None:
        raise RuntimeError("Loss-side clean vertices are unavailable.")
    loss = (recovered - clean.double()).square().sum(dim=-1).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("S1 preflight loss is non-finite.")
    loss.backward()

    if model.direct_predictor is None:
        raise RuntimeError("S1 model has no Direct graph tower.")
    groups = model.split_geometry_parameter_groups()
    lap_graph = tuple(model.predictor.input_mlp.parameters()) + tuple(
        model.predictor.blocks.parameters()
    )
    direct_graph = tuple(model.direct_predictor.input_mlp.parameters()) + tuple(
        model.direct_predictor.blocks.parameters()
    )
    result = {
        "sample_id": str(prepared.sample["sample_id"]),
        "model_input_keys": sorted(conditioned),
        "forbidden_gt_input_keys": forbidden_input_keys,
        "prediction_shapes": {
            "delta_hat": list(delta.shape),
            "DeltaV_direct": list(displacement.shape),
            "V_H": list(recovered.shape),
        },
        "loss": float(loss.detach().cpu()),
        "solver": {
            "converged": bool(solver.converged),
            "iterations": int(solver.iterations),
            "relative_residual": float(solver.relative_residual),
        },
        "delta_hat": _tensor_gradient(delta),
        "DeltaV_direct": _tensor_gradient(displacement),
        "V_direct": _tensor_gradient(v_direct),
        "shared_visual_frontend": _norm(model.image_encoder.parameters()),
        "lap_graph_tower": _norm(lap_graph),
        "lap_head": _norm(groups["lap_head"]),
        "direct_graph_tower": _norm(direct_graph),
        "direct_head": _norm(groups["direct_head"]),
    }
    required = (
        result["delta_hat"],
        result["DeltaV_direct"],
        result["V_direct"],
        result["shared_visual_frontend"],
        result["lap_graph_tower"],
        result["lap_head"],
        result["direct_graph_tower"],
        result["direct_head"],
    )
    result["passed"] = (
        solver.converged
        and not forbidden_input_keys
        and all(
            bool(item["all_finite"])
            and float(item["norm"]) > 0.0
            and int(item["nonzero_entries"]) > 0
            for item in required
        )
    )
    if not result["passed"]:
        raise RuntimeError(f"S1 real-gradient preflight failed: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--s0-config", required=True, type=Path)
    parser.add_argument("--s1-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--representative-samples", type=int, default=3)
    args = parser.parse_args()

    s0_config = _read(args.s0_config.resolve())
    s1_config = _read(args.s1_config.resolve())
    torch.manual_seed(7)
    s0_model = _build_model(s0_config, None, False)
    torch.manual_seed(7)
    s1_model = _build_model(s1_config, None, False)
    if not isinstance(s1_model, LearnedLaplacianModel):
        raise RuntimeError("S1 did not instantiate LearnedLaplacianModel.")
    if not s1_model.split_geometry_towers_enabled:
        raise RuntimeError("S1 split geometry towers are disabled.")
    if s1_model.direct_predictor is None:
        raise RuntimeError("S1 Direct graph tower is absent.")

    groups = s1_model.split_geometry_parameter_groups()
    group_storage = {
        name: _storage_ids(parameters)
        for name, parameters in groups.items()
        if name in {"shared_frontend", "lap_tower", "direct_tower"}
    }
    storage_disjoint = (
        group_storage["shared_frontend"].isdisjoint(group_storage["lap_tower"])
        and group_storage["shared_frontend"].isdisjoint(
            group_storage["direct_tower"]
        )
        and group_storage["lap_tower"].isdisjoint(group_storage["direct_tower"])
    )
    all_parameter_ids = {id(parameter) for parameter in s1_model.parameters()}
    grouped_parameter_ids = {
        id(parameter)
        for name in ("shared_frontend", "lap_tower", "direct_tower")
        for parameter in groups[name]
    }
    config_audit = _critical_config_audit(s0_config, s1_config)
    contract_checks = {
        "critical_S0_fields_identical": all(config_audit.values()),
        "dataset_400_50_50": s1_config["dataset"]["expected_split_counts"]
        == {"train": 400, "validation": 50, "test": 50},
        "seed_7": s1_config["seed"] == 7,
        "world_size_4": s1_config["experiment_metadata"][
            "distributed_world_size"
        ]
        == 4,
        "accumulation_2": s1_config["multi_object_training"][
            "gradient_accumulation_meshes"
        ]
        == 2,
        "effective_global_batch_8": s1_config["experiment_metadata"][
            "effective_global_batch_meshes"
        ]
        == 8,
        "steps_20000": s1_config["multi_object_training"][
            "max_optimizer_steps"
        ]
        == 20000,
        "final_geometry_loss_only": (
            s1_config["training"]["hybrid_single_geometry_loss"]["enabled"]
            and not s1_config["training"]["hybrid_single_geometry_loss"][
                "auxiliary_laplacian_loss"
            ]
            and not s1_config["training"]["hybrid_single_geometry_loss"][
                "auxiliary_direct_vertex_loss"
            ]
            and not s1_config["training"]["recovery_aware_geometry_loss"][
                "enabled"
            ]
        ),
        "lambda_3e_2": s1_config["training"]["hybrid_single_geometry_loss"][
            "lambda"
        ]
        == LAMBDA,
        "float64_pcg_tol1e_8_max2048": (
            s1_config["training"]["hybrid_single_geometry_loss"][
                "compute_dtype"
            ]
            == "float64"
            and s1_config["training"]["hybrid_single_geometry_loss"][
                "tolerance"
            ]
            == TOLERANCE
            and s1_config["training"]["hybrid_single_geometry_loss"][
                "maximum_iterations"
            ]
            == MAXIMUM_ITERATIONS
        ),
        "validation_hybrid_chamfer_selection": s1_config[
            "experiment_metadata"
        ]["checkpoint_selector"]
        == "validation_final_hybrid_unified_v2_chamfer_only",
        "split_before_graph_processing": s1_model.split_geometry_towers_enabled,
        "all_graph_parameters_independent": storage_disjoint,
        "parameter_partition_complete": all_parameter_ids == grouped_parameter_ids,
        "shared_frontend_initialization_matches_S0": _same_state(
            s0_model.image_encoder, s1_model.image_encoder
        ),
        "lap_tower_initialization_matches_S0": _same_state(
            s0_model.predictor, s1_model.predictor
        ),
        "from_scratch_no_checkpoint": "initialization_checkpoint"
        not in s1_config.get("multi_object_training", {}),
    }
    device = torch.device(args.device)
    validation = PreparedMeshDataset.from_manifest(
        args.manifest.resolve(), "validation"
    )
    s1_model = s1_model.to(device)
    real_gradient = _real_gradient_audit(
        s1_model, validation, s1_config, device
    )

    parameter_counts = {
        "shared_frontend": sum(
            parameter.numel() for parameter in groups["shared_frontend"]
        ),
        "lap_tower": sum(parameter.numel() for parameter in groups["lap_tower"]),
        "direct_tower": sum(
            parameter.numel() for parameter in groups["direct_tower"]
        ),
        "total_S1": sum(parameter.numel() for parameter in groups["shared_frontend"])
        + sum(parameter.numel() for parameter in groups["lap_tower"])
        + sum(parameter.numel() for parameter in groups["direct_tower"]),
        "total_S0": sum(parameter.numel() for parameter in s0_model.parameters()),
    }
    parameter_counts["increase_vs_S0"] = (
        parameter_counts["total_S1"] - parameter_counts["total_S0"]
    )
    payload = {
        "contract_audit": all(contract_checks.values()) and real_gradient["passed"],
        "contract_checks": contract_checks,
        "critical_config_field_audit": config_audit,
        "architecture": {
            "fork_location": "shared vertex_features before both predictor input MLPs and all graph/message-passing layers",
            "shared_frontend": "image encoder + HF construction + camera projection/sampling + multiview aggregation + geometry/Fourier construction",
            "lap_tower": "independent input MLP + 3 graph blocks + Lap output MLP",
            "direct_tower": "independent input MLP + 3 graph blocks + Direct output MLP",
            "post_fork_feature_exchange": False,
            "parameter_counts": parameter_counts,
            "storage_disjoint": storage_disjoint,
        },
        "real_model_gradient_preflight": real_gradient,
        "analytic_gradient_preflight": _gradient_audit(),
        "solver_audit": _solver_audit(
            args.manifest.resolve(), args.representative_samples
        ),
        "solver_settings": {
            "operator": "uniform random-walk L=I-D^-1A",
            "lambda": LAMBDA,
            "dtype": "float64",
            "tolerance": TOLERANCE,
            "maximum_iterations": MAXIMUM_ITERATIONS,
        },
    }
    if not payload["contract_audit"]:
        raise RuntimeError(f"S1 contract failed: {payload}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
