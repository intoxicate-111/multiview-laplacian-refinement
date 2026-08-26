#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed step-0 and real-gradient audit for continuous B/E training."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import _clean_mesh, _geometry_row
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import (
    _exact_query_sample,
    _load_device_item,
)
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery_with_audit,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.two_branch_hybrid import TwoBranchPretrainedHybridModel


LAMBDA = 3e-2
TOLERANCE = 1e-8
MAXIMUM_ITERATIONS = 2048
ARM_B = "B_lap_plus_refine"
ARM_E = "E_direct_vertex_residual"
ARM_H = "Hybrid_B_laplacian_E_anchor"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archived_rows(report: Path, arm: str, split: str) -> list[dict[str, Any]]:
    payload = _read(report / "shards" / f"{arm}.json")
    if payload.get("arm") != arm:
        raise RuntimeError(f"Archived arm mismatch for {arm}.")
    return [dict(row) for row in payload["rows"] if row["split"] == split]


def _archived_array(report: Path, arm: str, split: str) -> np.ndarray:
    return np.load(report / "shards" / f"{arm}_prediction_arrays.npz")[
        f"{split}_prediction"
    ].astype(np.float64)


def _starts(rows: list[dict[str, Any]], array: np.ndarray) -> list[int]:
    counts = [int(row["vertices"]) for row in rows]
    if sum(counts) != len(array):
        raise RuntimeError("Archived prediction length does not match row metadata.")
    return list(np.cumsum([0, *counts[:-1]]))


def _frozen_rows(path: Path, split: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row["split"] == split and row["arm"] == ARM_H
        }


def _norm(parameters: Iterable[torch.nn.Parameter]) -> dict[str, Any]:
    tensors = [parameter.grad for parameter in parameters]
    present = [value for value in tensors if value is not None]
    finite = bool(present) and all(bool(torch.isfinite(value).all()) for value in present)
    squared = sum(float(value.detach().float().square().sum().cpu()) for value in present)
    return {
        "parameter_tensors": len(tensors),
        "gradient_tensors": len(present),
        "all_finite": finite,
        "norm": math.sqrt(squared),
        "nonzero_entries": sum(int(torch.count_nonzero(value).detach().cpu()) for value in present),
    }


def _tensor_gradient(value: torch.Tensor) -> dict[str, Any]:
    gradient = value.grad
    if gradient is None:
        return {"present": False, "all_finite": False, "norm": 0.0, "nonzero_entries": 0}
    return {
        "present": True,
        "all_finite": bool(torch.isfinite(gradient).all()),
        "norm": float(torch.linalg.vector_norm(gradient).detach().cpu()),
        "nonzero_entries": int(torch.count_nonzero(gradient).detach().cpu()),
    }


def _gradient_audit(
    model: TwoBranchPretrainedHybridModel,
    dataset: PreparedMeshDataset,
    config: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    prepared = _load_device_item(dataset, 0, config, device)
    conditioned = _exact_query_sample(prepared.sample, device)
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        output = model(conditioned)
    delta_b = output.predicted_laplacian.float()
    delta_v_e = output.direct_vertex_displacement_prediction
    if delta_v_e is None:
        raise RuntimeError("Arm E did not emit its direct displacement.")
    delta_v_e = delta_v_e.float()
    delta_b.retain_grad()
    delta_v_e.retain_grad()
    v_direct = prepared.sample["vertices"].double() + delta_v_e.double()
    v_direct.retain_grad()
    recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
        delta_b.double(),
        v_direct,
        prepared.sample["edge_index"],
        prepared.sample["vertex_degree"].double(),
        regularization=LAMBDA,
        maximum_iterations=MAXIMUM_ITERATIONS,
        tolerance=TOLERANCE,
    )
    if not audit.converged:
        raise RuntimeError(f"Real gradient audit PCG failed: {audit}")
    clean = prepared.clean_vertices
    if clean is None:
        raise RuntimeError("The loss side lacks clean vertices.")
    loss = (recovered - clean.double()).square().sum(dim=-1).mean()
    loss.backward()
    groups = model.branch_parameter_groups()
    result = {
        "sample_id": str(prepared.sample["sample_id"]),
        "loss": float(loss.detach().cpu()),
        "pcg_iterations": int(audit.iterations),
        "pcg_relative_residual": float(audit.relative_residual),
        "delta_B": _tensor_gradient(delta_b),
        "V_direct": _tensor_gradient(v_direct),
        "DeltaV_E": _tensor_gradient(delta_v_e),
        "theta_B_head": _norm(groups["b_head"]),
        "theta_B_backbone": _norm(groups["b_backbone"]),
        "theta_E_head": _norm(groups["e_head"]),
        "theta_E_backbone": _norm(groups["e_backbone"]),
    }
    required = (
        result["delta_B"], result["V_direct"], result["DeltaV_E"],
        result["theta_B_head"], result["theta_B_backbone"],
        result["theta_E_head"], result["theta_E_backbone"],
    )
    result["passed"] = all(
        bool(item["all_finite"]) and float(item["norm"]) > 0.0 for item in required
    )
    if not result["passed"]:
        raise RuntimeError(f"A required pretrained branch gradient failed: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--arm-b-report", required=True, type=Path)
    parser.add_argument("--arm-e-report", required=True, type=Path)
    parser.add_argument("--frozen-report", required=True, type=Path)
    parser.add_argument("--tight-reference", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-audit", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = _read(args.config.resolve())
    device = torch.device(args.device)
    model = _build_model(config, None, False).to(device)
    if not isinstance(model, TwoBranchPretrainedHybridModel):
        raise RuntimeError("Config did not instantiate two complete pretrained branches.")
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), args.split)

    b_rows = _archived_rows(args.arm_b_report.resolve(), ARM_B, args.split)
    e_rows = _archived_rows(args.arm_e_report.resolve(), ARM_E, args.split)
    expected_ids = list(dataset.sample_ids)
    if [row["sample_id"] for row in b_rows] != expected_ids:
        raise RuntimeError("Arm-B archive order differs from the manifest.")
    if [row["sample_id"] for row in e_rows] != expected_ids:
        raise RuntimeError("Arm-E archive order differs from the manifest.")
    b_archive = _archived_array(args.arm_b_report.resolve(), ARM_B, args.split)
    e_archive = _archived_array(args.arm_e_report.resolve(), ARM_E, args.split)
    b_starts, e_starts = _starts(b_rows, b_archive), _starts(e_rows, e_archive)
    frozen = _frozen_rows(
        args.frozen_report.resolve() / "matched_per_sample.csv", args.split
    )
    tight_reference = _read(args.tight_reference.resolve())
    tight_rows = {row["sample_id"]: row for row in tight_reference["rows"]}

    rows: list[dict[str, Any]] = []
    internal_reproduction: dict[str, float] | None = None
    count_to_run = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    for index in range(count_to_run):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        prepared = _load_device_item(dataset, index, config, device)
        conditioned = _exact_query_sample(prepared.sample, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            output = model(conditioned)
            if index == 0:
                b_alone = model.arm_b(conditioned).predicted_laplacian.float()
                e_alone = model.arm_e(conditioned).predicted_laplacian.float()
                repeated = model(conditioned)
                assert repeated.direct_vertex_displacement_prediction is not None
                internal_reproduction = {
                    "wrapper_vs_separate_b_max": float(
                        torch.max(torch.abs(output.predicted_laplacian.float() - b_alone)).cpu()
                    ),
                    "wrapper_vs_separate_e_max": float(
                        torch.max(torch.abs(output.direct_vertex_displacement_prediction.float() - e_alone)).cpu()
                    ),
                    "repeated_wrapper_b_max": float(
                        torch.max(torch.abs(output.predicted_laplacian.float() - repeated.predicted_laplacian.float())).cpu()
                    ),
                    "repeated_wrapper_e_max": float(
                        torch.max(torch.abs(output.direct_vertex_displacement_prediction.float() - repeated.direct_vertex_displacement_prediction.float())).cpu()
                    ),
                }
        direct_prediction = output.direct_vertex_displacement_prediction
        if direct_prediction is None:
            raise RuntimeError(f"{sample_id}: missing Arm-E output.")
        delta = output.predicted_laplacian.float().detach().double()
        displacement = direct_prediction.float().detach().double()
        count = int(delta.shape[0])
        b_expected = b_archive[b_starts[index] : b_starts[index] + count]
        e_expected = e_archive[e_starts[index] : e_starts[index] + count]
        b_difference = delta.cpu().numpy() - b_expected
        e_difference = displacement.cpu().numpy() - e_expected
        v_direct = prepared.sample["vertices"].double() + displacement
        recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
            delta,
            v_direct,
            prepared.sample["edge_index"],
            prepared.sample["vertex_degree"].double(),
            regularization=LAMBDA,
            maximum_iterations=MAXIMUM_ITERATIONS,
            tolerance=TOLERANCE,
        )
        if not audit.converged:
            raise RuntimeError(f"{sample_id}: step-0 PCG failed: {audit}")
        vertices = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        initial = Mesh(vertices, faces).ensure_normals()
        clean = _clean_mesh(static)
        geometry = _geometry_row(
            args.split,
            sample_id,
            "continuous_step0",
            Mesh(recovered.detach().cpu().numpy(), faces.copy()).ensure_normals(),
            clean,
            initial,
        )
        refined_cd = float(geometry["chamfer"])
        tight_cd = float(tight_rows[sample_id]["tight_chamfer"])
        rows.append(
            {
                "sample_id": sample_id,
                "vertices": count,
                "b_output_rms_difference": float(np.sqrt(np.mean(np.square(b_difference)))),
                "b_output_max_difference": float(np.max(np.abs(b_difference))),
                "e_output_rms_difference": float(np.sqrt(np.mean(np.square(e_difference)))),
                "e_output_max_difference": float(np.max(np.abs(e_difference))),
                "pcg_iterations": int(audit.iterations),
                "pcg_relative_residual": float(audit.relative_residual),
                "refined_chamfer": refined_cd,
                "tight_reference_chamfer": tight_cd,
                "tight_reference_cd_difference": refined_cd - tight_cd,
                "frozen_loose_cd_difference": refined_cd - float(frozen[sample_id]["refined_chamfer"]),
                "p2s_p95": float(geometry["p2s_p95"]),
                "fscore": float(geometry["fscore"]),
                "normal_consistency": float(geometry["normal_consistency"]),
                "introduced_flipped_faces": int(geometry["introduced_flipped_faces"]),
                "new_degenerate_faces": int(geometry["new_degenerate_faces"]),
                "vertex_rms": float(
                    np.sqrt(np.mean(np.sum(np.square(recovered.detach().cpu().numpy() - clean.vertices), axis=1)))
                ),
                "improved": refined_cd < float(frozen[sample_id]["initial_chamfer"]),
            }
        )
        print(f"{args.split} {index + 1}/{count_to_run} {sample_id}", flush=True)

    maximum_b = max(float(row["b_output_max_difference"]) for row in rows)
    maximum_e = max(float(row["e_output_max_difference"]) for row in rows)
    maximum_tight_cd = max(abs(float(row["tight_reference_cd_difference"])) for row in rows)
    mean_chamfer = float(np.mean([float(row["refined_chamfer"]) for row in rows]))
    tight_mean_chamfer = float(
        np.mean([float(row["tight_reference_chamfer"]) for row in rows])
    )
    aggregate_relative_difference = abs(mean_chamfer - tight_mean_chamfer) / tight_mean_chamfer
    nan_inf_count = sum(
        not math.isfinite(float(row[key]))
        for row in rows
        for key in (
            "refined_chamfer", "p2s_p95", "fscore", "normal_consistency",
            "vertex_rms", "pcg_relative_residual",
        )
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    summary: dict[str, Any] = {
        "split": args.split,
        "samples": len(rows),
        "parameter_count": parameter_count,
        "shared_parameter_storage": bool(
            {parameter.data_ptr() for parameter in model.arm_b.parameters()}
            & {parameter.data_ptr() for parameter in model.arm_e.parameters()}
        ),
        "checkpoint_identity": {
            "arm_b": {"path": model.arm_b_checkpoint, "sha256": _sha256(Path(model.arm_b_checkpoint))},
            "arm_e": {"path": model.arm_e_checkpoint, "sha256": _sha256(Path(model.arm_e_checkpoint))},
        },
        "solver": {
            "lambda": LAMBDA,
            "tolerance": TOLERANCE,
            "maximum_iterations": MAXIMUM_ITERATIONS,
            "maximum_relative_residual": max(float(row["pcg_relative_residual"]) for row in rows),
        },
        "output_reproduction": {
            "maximum_arm_b_coordinate_difference": maximum_b,
            "maximum_arm_e_coordinate_difference": maximum_e,
            "maximum_tight_reference_cd_difference": maximum_tight_cd,
            "aggregate_tight_reference_cd": tight_mean_chamfer,
            "aggregate_relative_cd_difference": aggregate_relative_difference,
            "latent_difference_is_diagnostic_only": True,
        },
        "internal_reproduction": internal_reproduction,
        "geometry": {
            "mean_chamfer": mean_chamfer,
            "mean_p2s_p95": float(np.mean([float(row["p2s_p95"]) for row in rows])),
            "mean_fscore": float(np.mean([float(row["fscore"]) for row in rows])),
            "mean_normal": float(np.mean([float(row["normal_consistency"]) for row in rows])),
            "introduced_flips": int(sum(int(row["introduced_flipped_faces"]) for row in rows)),
            "new_degenerates": int(sum(int(row["new_degenerate_faces"]) for row in rows)),
            "mean_vertex_rms": float(np.mean([float(row["vertex_rms"]) for row in rows])),
            "improved_worsened": [sum(bool(row["improved"]) for row in rows), sum(not bool(row["improved"]) for row in rows)],
            "nan_inf_count": nan_inf_count,
        },
        "rows": rows,
    }
    summary["gradient_audit"] = (
        _gradient_audit(model, dataset, config, device, amp_enabled, amp_dtype)
        if args.gradient_audit
        else None
    )
    expected_sha = {
        "arm_b": "a483e2212f568e771873594cf1e37d13d62cbd2e1e72244baded7dd15573970c",
        "arm_e": "6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1",
    }
    summary["contract_checks"] = {
        "two_complete_networks": parameter_count == 2 * 826115,
        "no_shared_parameter_storage": not summary["shared_parameter_storage"],
        "checkpoint_sha": all(
            summary["checkpoint_identity"][arm]["sha256"] == expected_sha[arm]
            for arm in expected_sha
        ),
        "all_solves_converged": summary["solver"]["maximum_relative_residual"] <= TOLERANCE,
        "latent_difference_diagnostic_recorded": math.isfinite(maximum_b) and math.isfinite(maximum_e),
        "aggregate_cd_relative_difference_le_0p1_percent": aggregate_relative_difference <= 1e-3,
        "maximum_per_sample_cd_difference_le_5e-5": maximum_tight_cd <= 5e-5,
        "test_improved_worsened_49_1": args.split != "test" or summary["geometry"]["improved_worsened"] == [49, 1],
        "nan_inf_zero": nan_inf_count == 0,
        "gradient_audit": summary["gradient_audit"] is None or summary["gradient_audit"]["passed"],
    }
    summary["contract_audit"] = all(summary["contract_checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    if not summary["contract_audit"]:
        raise RuntimeError(f"Step-0 reproduction failed: {summary['contract_checks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
