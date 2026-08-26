#!/usr/bin/env python3
from __future__ import annotations

"""Validation-only step-0 and gradient gate for old-domain B+E continuation."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery_with_audit,
    recovery_forward_audit,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.two_branch_hybrid import TwoBranchPretrainedHybridModel


TOLERANCE = 1e-8
MAXIMUM_ITERATIONS = 2048
EXPECTED_BRANCH_PARAMETERS = 826115


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> dict[str, Any]:
    tensors = tuple(parameters)
    gradients = [parameter.grad for parameter in tensors if parameter.grad is not None]
    finite = bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)
    squared = sum(float(value.detach().float().square().sum().cpu()) for value in gradients)
    return {
        "parameter_tensors": len(tensors),
        "gradient_tensors": len(gradients),
        "all_finite": finite,
        "norm": math.sqrt(squared),
        "nonzero_entries": sum(int(torch.count_nonzero(value).detach().cpu()) for value in gradients),
    }


def tensor_gradient(value: torch.Tensor) -> dict[str, Any]:
    gradient = value.grad
    if gradient is None:
        return {"present": False, "all_finite": False, "norm": 0.0, "nonzero_entries": 0}
    return {
        "present": True,
        "all_finite": bool(torch.isfinite(gradient).all()),
        "norm": float(torch.linalg.vector_norm(gradient).detach().cpu()),
        "nonzero_entries": int(torch.count_nonzero(gradient).detach().cpu()),
    }


def gradient_audit(
    model: TwoBranchPretrainedHybridModel,
    dataset: PreparedMeshDataset,
    config: dict[str, Any],
    device: torch.device,
    regularization: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    prepared = _load_device_item(dataset, 0, config, device)
    conditioned = _exact_query_sample(prepared.sample, device)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        output = model(conditioned)
    direct_displacement = output.direct_vertex_displacement_prediction
    if direct_displacement is None:
        raise RuntimeError("Arm E did not emit a direct displacement")
    delta_b = output.predicted_laplacian.float()
    delta_v_e = direct_displacement.float()
    delta_b.retain_grad()
    delta_v_e.retain_grad()
    v_direct = prepared.sample["vertices"].double() + delta_v_e.double()
    v_direct.retain_grad()
    recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
        delta_b.double(),
        v_direct,
        prepared.sample["edge_index"],
        prepared.sample["vertex_degree"].double(),
        regularization=regularization,
        maximum_iterations=MAXIMUM_ITERATIONS,
        tolerance=TOLERANCE,
    )
    if not audit.converged or float(audit.relative_residual) > TOLERANCE:
        raise RuntimeError(f"Gradient-audit PCG failed: {audit}")
    if prepared.clean_vertices is None:
        raise RuntimeError("Validation sample lacks loss-side clean vertices")
    loss = (recovered - prepared.clean_vertices.double()).square().sum(dim=-1).mean()
    loss.backward()
    groups = model.branch_parameter_groups()
    result = {
        "sample_id": str(prepared.sample["sample_id"]),
        "loss": float(loss.detach().cpu()),
        "pcg_iterations": int(audit.iterations),
        "pcg_relative_residual": float(audit.relative_residual),
        "delta_B": tensor_gradient(delta_b),
        "V_direct": tensor_gradient(v_direct),
        "DeltaV_E": tensor_gradient(delta_v_e),
        "theta_B_head": gradient_norm(groups["b_head"]),
        "theta_B_backbone": gradient_norm(groups["b_backbone"]),
        "theta_E_head": gradient_norm(groups["e_head"]),
        "theta_E_backbone": gradient_norm(groups["e_backbone"]),
    }
    required = (
        result["delta_B"],
        result["V_direct"],
        result["DeltaV_E"],
        result["theta_B_head"],
        result["theta_B_backbone"],
        result["theta_E_head"],
        result["theta_E_backbone"],
    )
    result["passed"] = all(
        bool(item["all_finite"]) and float(item["norm"]) > 0.0 for item in required
    )
    return result


def selected_reference_rows(path: Path, selected_lambda: float) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if float(row["lambda"]) == selected_lambda
        ]
    if len(rows) != 25:
        raise RuntimeError(f"Expected 25 selected-lambda validation rows, found {len(rows)}")
    return {row["sample_id"]: row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--specialist-summary", required=True, type=Path)
    parser.add_argument("--specialist-predictions", required=True, type=Path)
    parser.add_argument("--lambda-selection", required=True, type=Path)
    parser.add_argument("--lambda-per-sample", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-sample-output", type=Path)
    parser.add_argument("--maximum-per-sample-cd-difference", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = read_object(args.config.resolve())
    selection = read_object(args.lambda_selection.resolve())
    specialist = read_object(args.specialist_summary.resolve())
    if selection.get("selection_split") != "validation" or selection.get("test_accessed") is not False:
        raise RuntimeError("Step-0 gate requires a validation-only lambda selection")
    if specialist.get("split") != "validation" or specialist.get("test_opened") is not False:
        raise RuntimeError("Step-0 gate accepts validation specialist predictions only")
    if not selection.get("contract_audit") or not specialist.get("contract_audit"):
        raise RuntimeError("A prerequisite validation audit failed")
    regularization = float(selection["selected_lambda"])
    configured_lambda = float(config["training"]["hybrid_single_geometry_loss"]["lambda"])
    if configured_lambda != regularization or float(config["recovery"]["lambda"]) != regularization:
        raise RuntimeError("Continuation config does not use the validation-selected lambda")

    device = torch.device(args.device)
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "validation")
    if len(dataset) != 25:
        raise RuntimeError(f"Expected 25 validation meshes, found {len(dataset)}")
    model = _build_model(config, None, False).to(device)
    if not isinstance(model, TwoBranchPretrainedHybridModel):
        raise RuntimeError("Config did not instantiate two complete independent branches")
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    branch_counts = {
        "B": sum(parameter.numel() for parameter in model.arm_b.parameters()),
        "E": sum(parameter.numel() for parameter in model.arm_e.parameters()),
    }
    shared_storage = bool(
        {parameter.data_ptr() for parameter in model.arm_b.parameters()}
        & {parameter.data_ptr() for parameter in model.arm_e.parameters()}
    )

    archive = np.load(args.specialist_predictions.resolve())
    archived_ids = archive["sample_ids"].tolist()
    offsets = archive["offsets"].astype(np.int64)
    if archived_ids != list(dataset.sample_ids):
        raise RuntimeError("Archived validation specialist order differs from the manifest")
    reference = selected_reference_rows(args.lambda_per_sample.resolve(), regularization)

    rows: list[dict[str, Any]] = []
    internal_repeat: dict[str, float] | None = None
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        if sample_id not in reference:
            raise RuntimeError(f"Missing selected-lambda validation reference: {sample_id}")
        prepared = _load_device_item(dataset, index, config, device)
        conditioned = _exact_query_sample(prepared.sample, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            output = model(conditioned)
            if index == 0:
                repeated = model(conditioned)
                if repeated.direct_vertex_displacement_prediction is None:
                    raise RuntimeError("Repeated Arm-E output is missing")
                internal_repeat = {
                    "B_max_coordinate_difference": float(
                        torch.max(torch.abs(output.predicted_laplacian - repeated.predicted_laplacian)).cpu()
                    ),
                    "E_max_coordinate_difference": float(
                        torch.max(
                            torch.abs(
                                output.direct_vertex_displacement_prediction
                                - repeated.direct_vertex_displacement_prediction
                            )
                        ).cpu()
                    ),
                }
        displacement = output.direct_vertex_displacement_prediction
        if displacement is None:
            raise RuntimeError(f"{sample_id}: Arm E output is missing")
        delta = output.predicted_laplacian.float().detach().double()
        displacement = displacement.float().detach().double()
        start, stop = int(offsets[index]), int(offsets[index + 1])
        b_archived = archive["b_prediction"][start:stop].astype(np.float64)
        e_archived = archive["e_displacement"][start:stop].astype(np.float64)
        b_difference = delta.cpu().numpy() - b_archived
        e_difference = displacement.cpu().numpy() - e_archived
        direct = prepared.sample["vertices"].double() + displacement
        with torch.no_grad():
            recovered, audit = recovery_forward_audit(
                delta,
                direct,
                prepared.sample["edge_index"],
                prepared.sample["vertex_degree"].double(),
                regularization=regularization,
                maximum_iterations=MAXIMUM_ITERATIONS,
                tolerance=TOLERANCE,
            )
        vertices = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        initial = Mesh(vertices, faces).ensure_normals()
        clean = _clean_mesh(static)
        recovered_np = recovered.detach().cpu().numpy()
        geometry = _geometry_row(
            "validation",
            sample_id,
            "continuous_step0",
            Mesh(recovered_np, faces.copy()).ensure_normals(),
            clean,
            initial,
        )
        refined_cd = float(geometry["chamfer"])
        reference_cd = float(reference[sample_id]["refined_chamfer"])
        initial_cd = float(reference[sample_id]["initial_chamfer"])
        rows.append(
            {
                "sample_id": sample_id,
                "vertices": len(vertices),
                "refined_chamfer": refined_cd,
                "reference_chamfer": reference_cd,
                "reference_cd_difference": refined_cd - reference_cd,
                "initial_chamfer": initial_cd,
                "p2s_p95": float(geometry["p2s_p95"]),
                "fscore": float(geometry["fscore"]),
                "normal_consistency": float(geometry["normal_consistency"]),
                "introduced_flipped_faces": int(geometry["introduced_flipped_faces"]),
                "new_degenerate_faces": int(geometry["new_degenerate_faces"]),
                "same_index_recovered_vertex_rms": float(
                    np.sqrt(np.mean(np.sum((recovered_np - clean.vertices) ** 2, axis=1)))
                ),
                "B_archived_rms_difference": float(np.sqrt(np.mean(b_difference**2))),
                "B_archived_max_difference": float(np.max(np.abs(b_difference))),
                "E_archived_rms_difference": float(np.sqrt(np.mean(e_difference**2))),
                "E_archived_max_difference": float(np.max(np.abs(e_difference))),
                "pcg_iterations": int(audit.iterations),
                "pcg_converged": bool(audit.converged),
                "pcg_relative_residual": float(audit.relative_residual),
                "improved": refined_cd < initial_cd,
                "worsened": refined_cd > initial_cd,
            }
        )
        print(f"validation {index + 1}/{len(dataset)} {sample_id}", flush=True)
        torch.cuda.empty_cache()

    mean_cd = float(np.mean([row["refined_chamfer"] for row in rows]))
    reference_mean_cd = float(np.mean([row["reference_chamfer"] for row in rows]))
    relative_difference = abs(mean_cd - reference_mean_cd) / reference_mean_cd
    maximum_cd_difference = max(abs(float(row["reference_cd_difference"])) for row in rows)
    nan_inf_count = sum(
        not math.isfinite(float(row[key]))
        for row in rows
        for key in (
            "refined_chamfer",
            "p2s_p95",
            "fscore",
            "normal_consistency",
            "same_index_recovered_vertex_rms",
            "pcg_relative_residual",
        )
    )
    actual_sha = {
        "B": sha256_file(Path(model.arm_b_checkpoint)),
        "E": sha256_file(Path(model.arm_e_checkpoint)),
    }
    expected_sha = {
        "B": str(specialist["arm_b_checkpoint_sha256"]),
        "E": str(specialist["arm_e_checkpoint_sha256"]),
    }
    gradients = gradient_audit(
        model, dataset, config, device, regularization, amp_enabled, amp_dtype
    )
    checks = {
        "validation_only": selection.get("test_accessed") is False
        and specialist.get("test_opened") is False,
        "selected_lambda_exact": configured_lambda == regularization,
        "two_complete_networks": branch_counts
        == {"B": EXPECTED_BRANCH_PARAMETERS, "E": EXPECTED_BRANCH_PARAMETERS}
        and parameter_count == 2 * EXPECTED_BRANCH_PARAMETERS,
        "no_shared_parameter_storage": not shared_storage,
        "checkpoint_sha_identity": actual_sha == expected_sha,
        "all_solves_converged": all(row["pcg_converged"] for row in rows)
        and max(float(row["pcg_relative_residual"]) for row in rows) <= TOLERANCE,
        "aggregate_cd_relative_difference_le_0p1_percent": relative_difference <= 1e-3,
        "maximum_per_sample_cd_difference_within_gate": maximum_cd_difference
        <= args.maximum_per_sample_cd_difference,
        "nan_inf_zero": nan_inf_count == 0,
        "all_required_gradients_finite_nonzero": bool(gradients["passed"]),
    }
    summary = {
        "contract_audit": all(checks.values()),
        "contract_checks": checks,
        "split": "validation",
        "test_accessed": False,
        "samples": len(rows),
        "parameter_count": parameter_count,
        "branch_parameter_counts": branch_counts,
        "shared_parameter_storage": shared_storage,
        "checkpoint_identity": {
            "B": {"path": model.arm_b_checkpoint, "sha256": actual_sha["B"]},
            "E": {"path": model.arm_e_checkpoint, "sha256": actual_sha["E"]},
        },
        "solver": {
            "operator": "uniform_random_walk_I_minus_DinvA",
            "lambda": regularization,
            "dtype": "float64",
            "tolerance": TOLERANCE,
            "maximum_iterations": MAXIMUM_ITERATIONS,
            "maximum_relative_residual": max(float(row["pcg_relative_residual"]) for row in rows),
        },
        "reproduction": {
            "reference": "frozen validation selected-lambda realization",
            "mean_chamfer": mean_cd,
            "reference_mean_chamfer": reference_mean_cd,
            "aggregate_relative_cd_difference": relative_difference,
            "maximum_per_sample_cd_difference": maximum_cd_difference,
            "maximum_per_sample_cd_difference_gate": args.maximum_per_sample_cd_difference,
            "maximum_B_archived_coordinate_difference": max(
                float(row["B_archived_max_difference"]) for row in rows
            ),
            "maximum_E_archived_coordinate_difference": max(
                float(row["E_archived_max_difference"]) for row in rows
            ),
            "latent_differences_are_diagnostic_only": True,
            "internal_repeat": internal_repeat,
        },
        "geometry": {
            "mean_chamfer": mean_cd,
            "p2s_p95": float(np.mean([row["p2s_p95"] for row in rows])),
            "fscore": float(np.mean([row["fscore"] for row in rows])),
            "normal_consistency": float(np.mean([row["normal_consistency"] for row in rows])),
            "same_index_recovered_vertex_rms": float(
                np.mean([row["same_index_recovered_vertex_rms"] for row in rows])
            ),
            "introduced_flipped_faces": int(sum(row["introduced_flipped_faces"] for row in rows)),
            "new_degenerate_faces": int(sum(row["new_degenerate_faces"] for row in rows)),
            "improved_worsened": [
                int(sum(row["improved"] for row in rows)),
                int(sum(row["worsened"] for row in rows)),
            ],
            "nan_inf_count": nan_inf_count,
        },
        "gradient_audit": gradients,
        "metric_protocol": METRIC_PROTOCOL,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.per_sample_output is not None:
        args.per_sample_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_sample_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    if not summary["contract_audit"]:
        raise RuntimeError(f"Step-0 validation gate failed: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
