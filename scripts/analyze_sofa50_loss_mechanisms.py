#!/usr/bin/env python3
from __future__ import annotations

"""Read-only output-space loss-mechanism audit for Sofa50 v2 A/B/E/S0.

The script deliberately differentiates the historical scalar objectives with
respect to stored/inferred output fields.  It never updates model parameters.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy import stats

from diagnose_sofa50_exact_solve_visibility_sweep import uniform_sparse_laplacian
from diagnose_sofa50_exact_target_oracle import _clean_mesh
from diagnose_sofa50_representation_b_vs_e import (
    SPECTRAL_BANDS,
    SPECTRAL_PROTOCOL,
    _starts,
    spectral_band_components,
)
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery,
    uniform_laplacian_apply,
    uniform_laplacian_transpose_apply,
)
from mlr.learned_laplacian.losses import weighted_robust_laplacian_loss
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint


LAMBDA_B = 1e-2
BETA_B = 1e-2
LAMBDA_S0 = 3e-2
B_MAXITER = 256
B_TOLERANCE = 1e-4
S0_MAXITER = 2048
S0_TOLERANCE = 1e-8
EXPECTED_S0_SHA = "9af46b5c3203415aa06c3967fe2f5d36bd1cab389f036c481e147e874e5dab62"
EPS = 1e-30
PATHS = (
    "g_A_delta",
    "g_B_lap_delta",
    "g_B_vertex_delta",
    "g_B_total_delta",
    "g_E_V",
    "g_S0_lap_delta",
    "g_S0_direct_V",
)

# Predeclared before observing this audit's measurements.
DECISION_THRESHOLDS = {
    "material_gradient_cosine": 0.80,
    "material_band_fraction_points": 0.10,
    "strong_cancellation_cosine": -0.50,
    "strong_cancellation_ratio": 0.50,
    "strong_path_norm_ratio": 10.0,
    "geometry_correlation_advantage": 0.10,
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive(report: Path, arm: str, split: str) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[int]]:
    payload = _read(report / "shards" / f"{arm}.json")
    rows = [dict(row) for row in payload["rows"] if row["split"] == split]
    arrays = np.load(report / "shards" / f"{arm}_prediction_arrays.npz")
    prediction = arrays[f"{split}_prediction"].astype(np.float64)
    target = arrays[f"{split}_target"].astype(np.float64)
    return rows, prediction, target, _starts(rows, prediction)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS))


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def _recover(
    delta: torch.Tensor,
    anchor: torch.Tensor,
    edge_index: torch.Tensor,
    degree: torch.Tensor,
    *,
    regularization: float,
    maximum_iterations: int,
    tolerance: float,
) -> torch.Tensor:
    return differentiable_regularized_sparse_recovery(
        delta,
        anchor,
        edge_index,
        degree,
        regularization=regularization,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )


def _huber(delta: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
    return weighted_robust_laplacian_loss(
        delta,
        target,
        confidence,
        loss_type="huber",
        huber_delta=0.01,
        charbonnier_epsilon=1e-3,
        target_magnitude_weight_lambda=0.0,
    )


def _mse_vertices(vertices: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    return (vertices - clean).square().sum(dim=-1).mean()


def _gradient_rows(
    split: str,
    sample_id: str,
    faces: np.ndarray,
    gradients: Mapping[str, np.ndarray],
    order: int,
) -> list[dict[str, Any]]:
    names = list(gradients)
    stacked = np.concatenate([gradients[name] for name in names], axis=1)
    filtered, _ = spectral_band_components(stacked, faces, order=order)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        column = slice(3 * index, 3 * index + 3)
        signal = stacked[:, column]
        total = float(np.square(signal).sum())
        energies = {
            band: max(0.0, float(np.einsum("ij,ij->", signal, filtered[band][:, column])))
            for band in SPECTRAL_BANDS
        }
        rows.append(
            {
                "split": split,
                "sample_id": sample_id,
                "path": name,
                "gradient_norm": math.sqrt(total),
                "total_energy": total,
                **{f"{band}_energy": value for band, value in energies.items()},
                **{
                    f"{band}_fraction": value / max(total, EPS)
                    for band, value in energies.items()
                },
            }
        )
    return rows


def _correlation_rows(
    split: str,
    sample_id: str,
    gradients: Mapping[str, np.ndarray],
    features: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, gradient in gradients.items():
        g = np.linalg.norm(gradient, axis=1)
        for feature, values_by_path in features.items():
            x = np.asarray(values_by_path[path], dtype=np.float64).reshape(-1)
            pearson = stats.pearsonr(g, x).statistic if len(g) > 2 else float("nan")
            spearman = stats.spearmanr(g, x).statistic if len(g) > 2 else float("nan")
            order = np.argsort(x, kind="stable")
            top10 = order[-max(1, math.ceil(0.10 * len(order))):]
            top1 = order[-max(1, math.ceil(0.01 * len(order))):]
            gradient_order = np.argsort(g, kind="stable")
            gradient_top10 = gradient_order[-max(1, math.ceil(0.10 * len(gradient_order))):]
            gradient_top1 = gradient_order[-max(1, math.ceil(0.01 * len(gradient_order))):]
            energy = np.square(g)
            rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "path": path,
                    "feature": feature,
                    "pearson": float(pearson),
                    "spearman": float(spearman),
                    "top10_gradient_energy_fraction": float(energy[top10].sum() / max(energy.sum(), EPS)),
                    "top1_gradient_energy_fraction": float(energy[top1].sum() / max(energy.sum(), EPS)),
                    "feature_mean_all_vertices": float(x.mean()),
                    "feature_mean_on_top10_gradient_vertices": float(x[gradient_top10].mean()),
                    "feature_mean_on_top1_gradient_vertices": float(x[gradient_top1].mean()),
                }
            )
    return rows


def _same_state_rows(
    split: str,
    sample_id: str,
    faces: np.ndarray,
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    order: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state, (direct, recovery) in pairs.items():
        spectral = _gradient_rows(
            split, sample_id, faces,
            {"direct": direct, "recovery": recovery}, order,
        )
        by_name = {row["path"]: row for row in spectral}
        rows.append(
            {
                "split": split,
                "sample_id": sample_id,
                "state": state,
                "cosine": _cosine(direct, recovery),
                "direct_norm": _norm(direct),
                "recovery_norm": _norm(recovery),
                "norm_ratio_recovery_over_direct": _norm(recovery) / max(_norm(direct), EPS),
                **{
                    f"{band}_fraction_direct": by_name["direct"][f"{band}_fraction"]
                    for band in SPECTRAL_BANDS
                },
                **{
                    f"{band}_fraction_recovery": by_name["recovery"][f"{band}_fraction"]
                    for band in SPECTRAL_BANDS
                },
            }
        )
    return rows


def _rhs_row(
    split: str,
    sample_id: str,
    state: str,
    delta: torch.Tensor,
    delta_gt: torch.Tensor,
    v_direct: torch.Tensor,
    clean: torch.Tensor,
    edge_index: torch.Tensor,
    degree: torch.Tensor,
) -> dict[str, Any]:
    e_l = uniform_laplacian_transpose_apply(delta - delta_gt, edge_index, degree)
    e_d = LAMBDA_S0 * (v_direct - clean)
    a = e_l.detach().cpu().numpy()
    b = e_d.detach().cpu().numpy()
    return {
        "split": split,
        "sample_id": sample_id,
        "state": state,
        "lap_rhs_norm": _norm(a),
        "direct_rhs_norm": _norm(b),
        "combined_rhs_norm": _norm(a + b),
        "rhs_cosine": _cosine(a, b),
        "cancellation_ratio": _norm(a + b) / max(_norm(a) + _norm(b), EPS),
    }


def _positional_same_state_row(
    split: str,
    sample_id: str,
    state: str,
    delta: torch.Tensor,
    v_direct_value: torch.Tensor,
    clean: torch.Tensor,
    edge_index: torch.Tensor,
    degree: torch.Tensor,
) -> dict[str, Any]:
    v_direct = v_direct_value.detach().clone().requires_grad_(True)
    direct_loss = _mse_vertices(v_direct, clean)
    direct_gradient = torch.autograd.grad(direct_loss, v_direct)[0]
    v_direct_hybrid = v_direct_value.detach().clone().requires_grad_(True)
    recovered = _recover(
        delta.detach(), v_direct_hybrid, edge_index, degree,
        regularization=LAMBDA_S0, maximum_iterations=S0_MAXITER,
        tolerance=S0_TOLERANCE,
    )
    hybrid_loss = _mse_vertices(recovered, clean)
    hybrid_gradient = torch.autograd.grad(hybrid_loss, v_direct_hybrid)[0]
    direct_np = direct_gradient.detach().cpu().numpy()
    hybrid_np = hybrid_gradient.detach().cpu().numpy()
    direct_vertex = np.linalg.norm(direct_np, axis=1)
    hybrid_vertex = np.linalg.norm(hybrid_np, axis=1)
    return {
        "split": split, "sample_id": sample_id, "state": state,
        "cosine": _cosine(direct_np, hybrid_np),
        "direct_norm": _norm(direct_np), "hybrid_norm": _norm(hybrid_np),
        "norm_ratio_hybrid_over_direct": _norm(hybrid_np) / max(_norm(direct_np), EPS),
        "direct_vertex_gradient_mean": float(direct_vertex.mean()),
        "direct_vertex_gradient_median": float(np.median(direct_vertex)),
        "direct_vertex_gradient_p95": float(np.quantile(direct_vertex, 0.95)),
        "hybrid_vertex_gradient_mean": float(hybrid_vertex.mean()),
        "hybrid_vertex_gradient_median": float(np.median(hybrid_vertex)),
        "hybrid_vertex_gradient_p95": float(np.quantile(hybrid_vertex, 0.95)),
    }


def _finite_difference(
    loss_function: Any,
    variable: torch.Tensor,
    analytic: torch.Tensor,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator(device=variable.device).manual_seed(seed)
    direction = torch.randn(variable.shape, dtype=variable.dtype, device=variable.device, generator=generator)
    direction = direction / torch.linalg.vector_norm(direction)
    epsilon = 2e-5
    with torch.no_grad():
        plus = float(loss_function(variable + epsilon * direction).detach().cpu())
        minus = float(loss_function(variable - epsilon * direction).detach().cpu())
    numeric = (plus - minus) / (2.0 * epsilon)
    exact = float((analytic * direction).sum().detach().cpu())
    return {
        "analytic": exact,
        "finite_difference": numeric,
        "relative_error": abs(exact - numeric) / max(abs(exact), abs(numeric), 1e-12),
    }


def shard(args: argparse.Namespace) -> None:
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), args.split)
    archive_specs = (
        ("A_lap_only", args.arm_ab_report),
        ("B_lap_plus_refine", args.arm_ab_report),
        ("E_direct_vertex_residual", args.arm_e_report),
    )
    archives: dict[str, tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[int]]] = {
        arm: _archive(Path(report).resolve(), arm, args.split) for arm, report in archive_specs
    }
    expected = list(dataset.sample_ids)
    for arm, (rows, _, _, _) in archives.items():
        if [row["sample_id"] for row in rows] != expected:
            raise RuntimeError(f"{arm} archive IDs do not match the manifest")
    if not np.array_equal(archives["A_lap_only"][2], archives["B_lap_plus_refine"][2]):
        raise RuntimeError("Archived A/B Laplacian targets differ")
    checkpoint_identities = []
    for arm, report in archive_specs:
        archived = _read(Path(report).resolve() / "shards" / f"{arm}.json")
        checkpoint_identities.append({
            "method": arm,
            "checkpoint": archived["checkpoint"],
            "checkpoint_sha256": archived["checkpoint_sha256"],
        })

    device = torch.device(args.device)
    run = args.s0_run.resolve()
    config_payload = _read(run / "run_config.json")
    config = config_payload.get("experiment_config", config_payload)
    checkpoint = run / args.checkpoint
    checkpoint_sha = _sha256(checkpoint)
    if args.checkpoint == "checkpoint_best.pt" and checkpoint_sha != EXPECTED_S0_SHA:
        raise RuntimeError(f"S0 checkpoint SHA mismatch: {checkpoint_sha}")
    model = _build_model(config, None, False).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)

    output: dict[str, list[dict[str, Any]]] = {
        "gradient_spectral_rows": [],
        "same_state_rows": [],
        "positional_same_state_rows": [],
        "correlation_rows": [],
        "rhs_rows": [],
        "finite_difference_rows": [],
    }
    indices = [i for i in range(len(dataset)) if i % args.shard_count == args.shard_index]
    for progress, index in enumerate(indices, 1):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        vertices_np = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        clean_np = _clean_mesh(static).vertices.astype(np.float64)
        lap, _ = uniform_sparse_laplacian(faces, len(vertices_np))
        delta_gt_np = np.asarray(lap @ clean_np, dtype=np.float64)
        count = len(vertices_np)
        fields: dict[str, np.ndarray] = {}
        for arm, (_, array, _, starts) in archives.items():
            fields[arm] = array[starts[index]: starts[index] + count]
        _, _, stored_laplacian_target, target_starts = archives["B_lap_plus_refine"]
        target_stored_np = stored_laplacian_target[target_starts[index]: target_starts[index] + count]

        prepared = _load_device_item(dataset, index, config, device)
        conditioned = _exact_query_sample(prepared.sample, device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            prediction = model(conditioned)
        if prediction.direct_vertex_displacement_prediction is None:
            raise RuntimeError("S0 did not return a direct branch")
        delta_s0_np = prediction.predicted_laplacian.detach().double().cpu().numpy()
        vdirect_s0_np = vertices_np + prediction.direct_vertex_displacement_prediction.detach().double().cpu().numpy()

        edge = torch.as_tensor(static["edge_index"], dtype=torch.long, device=device)
        # Historical A/B/E output-space losses use float32; S0 recovery uses float64.
        degree32 = torch.as_tensor(static["vertex_degree"], dtype=torch.float32, device=device)
        degree64 = degree32.double()
        vertices32 = torch.as_tensor(vertices_np, dtype=torch.float32, device=device)
        clean32 = torch.as_tensor(clean_np, dtype=torch.float32, device=device)
        # Use the archived float32 target tensor bit pattern for the historical
        # A/B Huber objectives.  Recomputed float64 L@V is reserved for exact
        # operator/RHS diagnostics below.
        target32 = torch.as_tensor(target_stored_np, dtype=torch.float32, device=device)
        confidence32 = torch.ones((count,), dtype=torch.float32, device=device)

        delta_a = torch.tensor(fields["A_lap_only"], dtype=torch.float32, device=device, requires_grad=True)
        loss_a = _huber(delta_a, target32, confidence32)
        g_a = torch.autograd.grad(loss_a, delta_a)[0]

        delta_b = torch.tensor(fields["B_lap_plus_refine"], dtype=torch.float32, device=device, requires_grad=True)
        loss_b_lap = _huber(delta_b, target32, confidence32)
        recovered_b = _recover(delta_b, vertices32, edge, degree32, regularization=LAMBDA_B, maximum_iterations=B_MAXITER, tolerance=B_TOLERANCE)
        loss_b_vertex = _mse_vertices(recovered_b, clean32)
        g_b_lap = torch.autograd.grad(loss_b_lap, delta_b, retain_graph=True)[0]
        g_b_vertex_unweighted = torch.autograd.grad(loss_b_vertex, delta_b, retain_graph=True)[0]
        g_b_total = torch.autograd.grad(loss_b_lap + BETA_B * loss_b_vertex, delta_b)[0]

        v_e = torch.tensor(vertices_np + fields["E_direct_vertex_residual"], dtype=torch.float32, device=device, requires_grad=True)
        loss_e = _mse_vertices(v_e, clean32)
        g_e = torch.autograd.grad(loss_e, v_e)[0]

        delta_s0 = torch.tensor(delta_s0_np, dtype=torch.float64, device=device, requires_grad=True)
        vdirect_s0 = torch.tensor(vdirect_s0_np, dtype=torch.float64, device=device, requires_grad=True)
        clean64 = torch.as_tensor(clean_np, dtype=torch.float64, device=device)
        vertices64 = torch.as_tensor(vertices_np, dtype=torch.float64, device=device)
        target64 = torch.as_tensor(delta_gt_np, dtype=torch.float64, device=device)
        recovered_s0 = _recover(delta_s0, vdirect_s0, edge, degree64, regularization=LAMBDA_S0, maximum_iterations=S0_MAXITER, tolerance=S0_TOLERANCE)
        loss_s0 = _mse_vertices(recovered_s0, clean64)
        g_s0_lap, g_s0_direct = torch.autograd.grad(loss_s0, (delta_s0, vdirect_s0))

        gradients = {
            "g_A_delta": g_a.detach().double().cpu().numpy(),
            "g_B_lap_delta": g_b_lap.detach().double().cpu().numpy(),
            # This is the exact contribution d(beta*L_vertex)/ddelta requested
            # by the historical B decomposition.
            "g_B_vertex_delta": (BETA_B * g_b_vertex_unweighted).detach().double().cpu().numpy(),
            "g_B_total_delta": g_b_total.detach().double().cpu().numpy(),
            "g_E_V": g_e.detach().double().cpu().numpy(),
            "g_S0_lap_delta": g_s0_lap.detach().cpu().numpy(),
            "g_S0_direct_V": g_s0_direct.detach().cpu().numpy(),
        }
        output["gradient_spectral_rows"].extend(_gradient_rows(args.split, sample_id, faces, gradients, args.chebyshev_order))

        # Same-state direct Huber versus unweighted recovery gradient.
        same_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for state, state_np in (
            ("A_state", fields["A_lap_only"]),
            ("B_state", fields["B_lap_plus_refine"]),
            ("S0_state", delta_s0_np),
        ):
            state_delta = torch.tensor(state_np, dtype=torch.float32, device=device, requires_grad=True)
            state_huber = _huber(state_delta, target32, confidence32)
            state_recovered = _recover(
                state_delta, vertices32, edge, degree32,
                regularization=LAMBDA_B, maximum_iterations=B_MAXITER,
                tolerance=B_TOLERANCE,
            )
            state_vertex = _mse_vertices(state_recovered, clean32)
            direct_grad = torch.autograd.grad(state_huber, state_delta, retain_graph=True)[0]
            recovery_grad = torch.autograd.grad(state_vertex, state_delta)[0]
            same_pairs[state] = (direct_grad.detach().cpu().numpy(), recovery_grad.detach().cpu().numpy())
        output["same_state_rows"].extend(_same_state_rows(args.split, sample_id, faces, same_pairs, args.chebyshev_order))

        # Same positional tensor, different scalar supervision: direct E-style
        # vertex MSE versus the solver-mediated positional pathway.  The Lap
        # field is held fixed at B or S0 respectively.
        output["positional_same_state_rows"].append(
            _positional_same_state_row(
                args.split, sample_id, "Frozen_B_plus_E",
                torch.as_tensor(fields["B_lap_plus_refine"], dtype=torch.float64, device=device),
                torch.as_tensor(vertices_np + fields["E_direct_vertex_residual"], dtype=torch.float64, device=device),
                clean64, edge, degree64,
            )
        )
        output["positional_same_state_rows"].append(
            _positional_same_state_row(
                args.split, sample_id, "S0", delta_s0.detach(),
                vdirect_s0.detach(), clean64, edge, degree64,
            )
        )

        # Error-localization features, always evaluated at the same sample/state.
        recovered_a = _recover(
            torch.as_tensor(fields["A_lap_only"], dtype=torch.float64, device=device), vertices64,
            edge, degree64, regularization=LAMBDA_B, maximum_iterations=S0_MAXITER, tolerance=S0_TOLERANCE,
        ).detach().cpu().numpy()
        recovered_b_np = recovered_b.detach().double().cpu().numpy()
        v_e_np = v_e.detach().double().cpu().numpy()
        recovered_s0_np = recovered_s0.detach().cpu().numpy()
        delta_for_path = {
            "g_A_delta": fields["A_lap_only"],
            "g_B_lap_delta": fields["B_lap_plus_refine"],
            "g_B_vertex_delta": fields["B_lap_plus_refine"],
            "g_B_total_delta": fields["B_lap_plus_refine"],
            "g_E_V": np.asarray(lap @ v_e_np),
            "g_S0_lap_delta": delta_s0_np,
            "g_S0_direct_V": np.asarray(lap @ vdirect_s0_np),
        }
        final_for_path = {
            "g_A_delta": recovered_a,
            "g_B_lap_delta": recovered_b_np,
            "g_B_vertex_delta": recovered_b_np,
            "g_B_total_delta": recovered_b_np,
            "g_E_V": v_e_np,
            "g_S0_lap_delta": recovered_s0_np,
            "g_S0_direct_V": recovered_s0_np,
        }
        same_index_for_path = dict(final_for_path)
        same_index_for_path["g_S0_direct_V"] = vdirect_s0_np
        features = {
            "raw_laplacian_error": {name: np.linalg.norm(value - delta_gt_np, axis=1) for name, value in delta_for_path.items()},
            "same_index_vertex_error": {name: np.linalg.norm(value - clean_np, axis=1) for name, value in same_index_for_path.items()},
            "gt_differential_magnitude": {name: np.linalg.norm(delta_gt_np, axis=1) for name in PATHS},
            "final_recovered_geometry_error": {name: np.linalg.norm(value - clean_np, axis=1) for name, value in final_for_path.items()},
        }
        output["correlation_rows"].extend(_correlation_rows(args.split, sample_id, gradients, features))

        delta_b64 = torch.as_tensor(fields["B_lap_plus_refine"], dtype=torch.float64, device=device)
        v_e64 = torch.as_tensor(v_e_np, dtype=torch.float64, device=device)
        output["rhs_rows"].append(_rhs_row(args.split, sample_id, "Frozen_B_plus_E", delta_b64, target64, v_e64, clean64, edge, degree64))
        output["rhs_rows"].append(_rhs_row(args.split, sample_id, "S0", delta_s0.detach(), target64, vdirect_s0.detach(), clean64, edge, degree64))

        if args.split == "validation" and index in (0, 24, 49):
            def lap_loss(value: torch.Tensor) -> torch.Tensor:
                recovered = _recover(value, vdirect_s0.detach(), edge, degree64, regularization=LAMBDA_S0, maximum_iterations=S0_MAXITER, tolerance=S0_TOLERANCE)
                return _mse_vertices(recovered, clean64)
            def direct_loss(value: torch.Tensor) -> torch.Tensor:
                recovered = _recover(delta_s0.detach(), value, edge, degree64, regularization=LAMBDA_S0, maximum_iterations=S0_MAXITER, tolerance=S0_TOLERANCE)
                return _mse_vertices(recovered, clean64)
            for branch, variable, analytic, function in (
                ("S0_lap", delta_s0.detach(), g_s0_lap, lap_loss),
                ("S0_direct", vdirect_s0.detach(), g_s0_direct, direct_loss),
            ):
                row = _finite_difference(function, variable, analytic, 7000 + index)
                row.update({"split": args.split, "sample_id": sample_id, "branch": branch})
                output["finite_difference_rows"].append(row)

        print(f"{args.split} shard={args.shard_index} {progress}/{len(indices)} {sample_id}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_json(
        args.output_dir / "shards" / f"loss_{args.split}_{args.shard_index:02d}.json",
        {
            "read_only": True,
            "split": args.split,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_identities": checkpoint_identities + [{
                "method": "S0", "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
            }],
            "spectral_protocol": SPECTRAL_PROTOCOL,
            "decision_thresholds": DECISION_THRESHOLDS,
            **output,
        },
    )


def _bootstrap(values: np.ndarray, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(10000, dtype=np.float64)
    for i in range(len(means)):
        means[i] = values[rng.integers(0, len(values), len(values))].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_difference(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    group_field: str,
    left: str,
    right: str,
    value_field: str,
) -> dict[str, Any]:
    a = {row["sample_id"]: float(row[value_field]) for row in rows if row["split"] == split and row[group_field] == left}
    b = {row["sample_id"]: float(row[value_field]) for row in rows if row["split"] == split and row[group_field] == right}
    if sorted(a) != sorted(b):
        raise RuntimeError(f"Paired IDs differ for {left} versus {right}")
    difference = np.asarray([a[key] - b[key] for key in sorted(a)], dtype=np.float64)
    low, high = _bootstrap(difference)
    return {
        "split": split,
        "left": left,
        "right": right,
        "field": value_field,
        "samples": len(difference),
        "mean_left_minus_right": float(difference.mean()),
        "median_left_minus_right": float(np.median(difference)),
        "left_lower": int(np.sum(difference < 0)),
        "right_lower": int(np.sum(difference > 0)),
        "ties": int(np.sum(difference == 0)),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
    }


def merge(args: argparse.Namespace) -> None:
    payloads = []
    for split in ("validation", "test"):
        for index in range(args.shard_count):
            payloads.append(_read(args.output_dir / "shards" / f"loss_{split}_{index:02d}.json"))
    categories = ("gradient_spectral_rows", "same_state_rows", "positional_same_state_rows", "correlation_rows", "rhs_rows", "finite_difference_rows")
    merged = {name: [row for payload in payloads for row in payload[name]] for name in categories}
    gradient_aggregate: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for path in PATHS:
            selected = [row for row in merged["gradient_spectral_rows"] if row["split"] == split and row["path"] == path]
            row: dict[str, Any] = {"split": split, "path": path, "samples": len(selected)}
            for field in (
                "gradient_norm", "total_energy",
                *(f"{band}_energy" for band in SPECTRAL_BANDS),
                *(f"{band}_fraction" for band in SPECTRAL_BANDS),
            ):
                values = np.asarray([item[field] for item in selected], dtype=np.float64)
                low, high = _bootstrap(values)
                row.update({field: float(values.mean()), f"{field}_ci_low": low, f"{field}_ci_high": high})
            gradient_aggregate.append(row)
    same_aggregate: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for state in ("A_state", "B_state", "S0_state"):
            selected = [row for row in merged["same_state_rows"] if row["split"] == split and row["state"] == state]
            row = {"split": split, "state": state, "samples": len(selected)}
            for field in ("cosine", "direct_norm", "recovery_norm", "norm_ratio_recovery_over_direct"):
                values = np.asarray([item[field] for item in selected])
                row.update({f"mean_{field}": float(values.mean()), f"median_{field}": float(np.median(values))})
            for band in SPECTRAL_BANDS:
                for kind in ("direct", "recovery"):
                    field = f"{band}_fraction_{kind}"
                    row[field] = float(np.mean([item[field] for item in selected]))
            same_aggregate.append(row)
    positional_same_aggregate: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for state in ("Frozen_B_plus_E", "S0"):
            selected = [row for row in merged["positional_same_state_rows"] if row["split"] == split and row["state"] == state]
            row = {"split": split, "state": state, "samples": len(selected)}
            for field in (
                "cosine", "direct_norm", "hybrid_norm", "norm_ratio_hybrid_over_direct",
                "direct_vertex_gradient_mean", "direct_vertex_gradient_median", "direct_vertex_gradient_p95",
                "hybrid_vertex_gradient_mean", "hybrid_vertex_gradient_median", "hybrid_vertex_gradient_p95",
            ):
                values = np.asarray([item[field] for item in selected], dtype=np.float64)
                row.update({f"mean_{field}": float(values.mean()), f"median_{field}": float(np.median(values))})
            positional_same_aggregate.append(row)
    correlation_aggregate: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for path in PATHS:
            for feature in sorted({row["feature"] for row in merged["correlation_rows"]}):
                selected = [row for row in merged["correlation_rows"] if row["split"] == split and row["path"] == path and row["feature"] == feature]
                correlation_aggregate.append({
                    "split": split, "path": path, "feature": feature, "samples": len(selected),
                    "mean_pearson": float(np.nanmean([row["pearson"] for row in selected])),
                    "mean_spearman": float(np.nanmean([row["spearman"] for row in selected])),
                    "mean_top10_gradient_energy_fraction": float(np.mean([row["top10_gradient_energy_fraction"] for row in selected])),
                    "mean_top1_gradient_energy_fraction": float(np.mean([row["top1_gradient_energy_fraction"] for row in selected])),
                    "mean_feature_all_vertices": float(np.mean([row["feature_mean_all_vertices"] for row in selected])),
                    "mean_feature_on_top10_gradient_vertices": float(np.mean([row["feature_mean_on_top10_gradient_vertices"] for row in selected])),
                    "mean_feature_on_top1_gradient_vertices": float(np.mean([row["feature_mean_on_top1_gradient_vertices"] for row in selected])),
                })
    rhs_aggregate: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for state in ("Frozen_B_plus_E", "S0"):
            selected = [row for row in merged["rhs_rows"] if row["split"] == split and row["state"] == state]
            rhs_aggregate.append({
                "split": split, "state": state, "samples": len(selected),
                **{f"mean_{field}": float(np.mean([row[field] for row in selected])) for field in ("lap_rhs_norm", "direct_rhs_norm", "combined_rhs_norm", "rhs_cosine", "cancellation_ratio")},
                **{f"median_{field}": float(np.median([row[field] for row in selected])) for field in ("lap_rhs_norm", "direct_rhs_norm", "combined_rhs_norm")},
                "median_rhs_cosine": float(np.median([row["rhs_cosine"] for row in selected])),
                "median_cancellation_ratio": float(np.median([row["cancellation_ratio"] for row in selected])),
                "p10_cancellation_ratio": float(np.quantile([row["cancellation_ratio"] for row in selected], 0.10)),
                "p90_cancellation_ratio": float(np.quantile([row["cancellation_ratio"] for row in selected], 0.90)),
            })
    paired_statistics: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for left, right in (
            ("g_A_delta", "g_B_lap_delta"),
            ("g_B_lap_delta", "g_B_vertex_delta"),
            ("g_B_lap_delta", "g_B_total_delta"),
            ("g_S0_lap_delta", "g_S0_direct_V"),
            ("g_E_V", "g_S0_direct_V"),
        ):
            paired_statistics.append(
                _paired_difference(
                    merged["gradient_spectral_rows"], split=split,
                    group_field="path", left=left, right=right,
                    value_field="gradient_norm",
                )
            )
        for field in ("lap_rhs_norm", "direct_rhs_norm", "combined_rhs_norm", "rhs_cosine", "cancellation_ratio"):
            paired_statistics.append(
                _paired_difference(
                    merged["rhs_rows"], split=split, group_field="state",
                    left="Frozen_B_plus_E", right="S0", value_field=field,
                )
            )
    checks = {
        "all_read_only": all(payload["read_only"] for payload in payloads),
        "all_shards_present": len(payloads) == 2 * args.shard_count,
        "validation_test_50": all(sum(row["split"] == split and row["path"] == "g_A_delta" for row in merged["gradient_spectral_rows"]) == 50 for split in ("validation", "test")),
        "s0_best_identity": len({payload["checkpoint_sha256"] for payload in payloads}) == 1 and payloads[0]["checkpoint_sha256"] == EXPECTED_S0_SHA,
        "finite_gradients": all(np.isfinite(row["gradient_norm"]) for row in merged["gradient_spectral_rows"]),
        "finite_positional_counterfactual": all(
            np.isfinite(float(row[field]))
            for row in merged["positional_same_state_rows"]
            for field in (
                "cosine", "direct_norm", "hybrid_norm",
                "norm_ratio_hybrid_over_direct",
            )
        ),
        "finite_rhs": all(
            np.isfinite(float(row[field]))
            for row in merged["rhs_rows"]
            for field in (
                "lap_rhs_norm", "direct_rhs_norm", "combined_rhs_norm",
                "rhs_cosine", "cancellation_ratio",
            )
        ),
        "finite_difference": all(row["relative_error"] < 5e-4 for row in merged["finite_difference_rows"]),
    }
    summary = {
        "contract_audit": all(checks.values()),
        "contract_checks": checks,
        "read_only": True,
        "checkpoint_identities": payloads[0]["checkpoint_identities"],
        "decision_thresholds": DECISION_THRESHOLDS,
        "spectral_protocol": SPECTRAL_PROTOCOL,
        "gradient_aggregate": gradient_aggregate,
        "same_state_aggregate": same_aggregate,
        "positional_same_state_aggregate": positional_same_aggregate,
        "correlation_aggregate": correlation_aggregate,
        "rhs_aggregate": rhs_aggregate,
        "paired_gradient_statistics": paired_statistics,
        "maximum_finite_difference_relative_error": max(row["relative_error"] for row in merged["finite_difference_rows"]),
    }
    _write_json(args.output_dir / "loss_mechanism_summary.json", summary)
    for name, rows in merged.items():
        _write_csv(args.output_dir / f"{name}.csv", rows)
    for name, rows in (
        ("gradient_aggregate", gradient_aggregate),
        ("same_state_aggregate", same_aggregate),
        ("positional_same_state_aggregate", positional_same_aggregate),
        ("correlation_aggregate", correlation_aggregate),
        ("rhs_aggregate", rhs_aggregate),
        ("paired_gradient_statistics", paired_statistics),
    ):
        _write_csv(args.output_dir / f"{name}.csv", rows)
    print(json.dumps({"contract_audit": summary["contract_audit"], "output": str(args.output_dir)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("shard", "merge"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--arm-ab-report", type=Path)
    parser.add_argument("--arm-e-report", type=Path)
    parser.add_argument("--s0-run", type=Path)
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--chebyshev-order", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.phase == "shard":
        for field in ("manifest", "arm_ab_report", "arm_e_report", "s0_run"):
            if getattr(args, field) is None:
                parser.error(f"--{field.replace('_', '-')} is required for shard")
        shard(args)
    else:
        merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
