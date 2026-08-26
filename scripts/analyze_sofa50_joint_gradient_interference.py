#!/usr/bin/env python3
from __future__ import annotations

"""Exact independent branch VJPs for the frozen shared-backbone Uniform hybrid."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery_with_audit,
    uniform_laplacian_apply,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint


LAMBDA = 3e-2
TOLERANCE = 1e-8
MAXIMUM_ITERATIONS = 2048
LAYERS = ("image_encoder", "graph_backbone", "all_shared_parameters", "projected_image_field", "shared_feature_Phi")


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flatten(grads: Iterable[torch.Tensor | None], params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    values = []
    for grad, parameter in zip(grads, params):
        values.append(torch.zeros_like(parameter).reshape(-1) if grad is None else grad.reshape(-1))
    return torch.cat(values) if values else torch.zeros(0)


def _metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left = left.detach().double().reshape(-1)
    right = right.detach().double().reshape(-1)
    left_norm = float(torch.linalg.vector_norm(left).cpu())
    right_norm = float(torch.linalg.vector_norm(right).cpu())
    denominator = max(left_norm * right_norm, 1e-300)
    cosine = float(torch.dot(left, right).cpu()) / denominator
    sum_norm = float(torch.linalg.vector_norm(left + right).cpu())
    return {
        "cosine": cosine,
        "lap_norm": left_norm,
        "direct_norm": right_norm,
        "magnitude_ratio": max(left_norm, right_norm) / max(min(left_norm, right_norm), 1e-300),
        "alignment_ratio": sum_norm / max(left_norm + right_norm, 1e-300),
    }


def _parameter_list(modules: Iterable[torch.nn.Module]) -> list[torch.nn.Parameter]:
    result: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def _one(
    model: torch.nn.Module,
    dataset: PreparedMeshDataset,
    index: int,
    config: Mapping[str, Any],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    repeat: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = _load_device_item(dataset, index, config, device)
    conditioned = _exact_query_sample(prepared.sample, device)
    holder: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        holder.append(inputs[0])

    hook = model.predictor.output_mlp.register_forward_pre_hook(capture)
    model.zero_grad(set_to_none=True)
    try:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(conditioned)
        if len(holder) != 1:
            raise RuntimeError(f"Expected one shared feature capture, got {len(holder)}")
        phi = holder[0]
        delta = output.predicted_laplacian.float()
        displacement = output.direct_vertex_displacement_prediction
        if displacement is None:
            raise RuntimeError("The joint model has no direct branch.")
        displacement = displacement.float()
        direct_vertices = prepared.sample["vertices"].double() + displacement.double()
        recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
            delta.double(),
            direct_vertices,
            prepared.sample["edge_index"],
            prepared.sample["vertex_degree"].double(),
            regularization=LAMBDA,
            maximum_iterations=MAXIMUM_ITERATIONS,
            tolerance=TOLERANCE,
        )
        if not audit.converged:
            raise RuntimeError(f"PCG failed: {audit}")
        clean = prepared.clean_vertices
        if clean is None:
            raise RuntimeError("Validation sample has no loss-side clean vertices.")
        loss = (recovered - clean.double()).square().sum(dim=-1).mean()
        g_delta, g_direct = torch.autograd.grad(
            loss, (delta, displacement), retain_graph=True, create_graph=False
        )
        image_params = _parameter_list((model.image_encoder,))
        graph_params = _parameter_list((model.predictor.input_mlp, *model.predictor.blocks))
        shared_params = image_params + graph_params
        lap_param_grads = torch.autograd.grad(
            delta, shared_params, grad_outputs=g_delta.detach(), retain_graph=True, allow_unused=True
        )
        direct_param_grads = torch.autograd.grad(
            displacement, shared_params, grad_outputs=g_direct.detach(), retain_graph=True, allow_unused=True
        )
        image_count = len(image_params)
        vectors = {
            "image_encoder": (
                _flatten(lap_param_grads[:image_count], image_params),
                _flatten(direct_param_grads[:image_count], image_params),
            ),
            "graph_backbone": (
                _flatten(lap_param_grads[image_count:], graph_params),
                _flatten(direct_param_grads[image_count:], graph_params),
            ),
            "all_shared_parameters": (
                _flatten(lap_param_grads, shared_params),
                _flatten(direct_param_grads, shared_params),
            ),
        }
        projected = output.aggregated_image_features
        lap_projected = torch.autograd.grad(
            delta, projected, grad_outputs=g_delta.detach(), retain_graph=True, allow_unused=False
        )[0]
        direct_projected = torch.autograd.grad(
            displacement, projected, grad_outputs=g_direct.detach(), retain_graph=True, allow_unused=False
        )[0]
        lap_phi = torch.autograd.grad(
            delta, phi, grad_outputs=g_delta.detach(), retain_graph=True, allow_unused=False
        )[0]
        direct_phi = torch.autograd.grad(
            displacement, phi, grad_outputs=g_direct.detach(), retain_graph=False, allow_unused=False
        )[0]
        vectors["projected_image_field"] = (lap_projected, direct_projected)
        vectors["shared_feature_Phi"] = (lap_phi, direct_phi)
        rows = []
        sample_id = str(prepared.sample["sample_id"])
        for layer, pair in vectors.items():
            rows.append(
                {
                    "sample_id": sample_id,
                    "sample_index": index,
                    "repeat": repeat,
                    "layer": layer,
                    **_metrics(*pair),
                }
            )
        z = g_direct.detach().double() / LAMBDA
        analytic_lap = uniform_laplacian_apply(
            z, prepared.sample["edge_index"], prepared.sample["vertex_degree"].double()
        )
        analytic_relative_error = float(
            torch.linalg.vector_norm(g_delta.detach().double() - analytic_lap).cpu()
            / torch.linalg.vector_norm(g_delta.detach().double()).clamp_min(1e-300).cpu()
        )
        audit_row = {
            "sample_id": sample_id,
            "sample_index": index,
            "repeat": repeat,
            "loss": float(loss.detach().cpu()),
            "latent_lap_gradient_norm": float(torch.linalg.vector_norm(g_delta).cpu()),
            "latent_direct_gradient_norm": float(torch.linalg.vector_norm(g_direct).cpu()),
            "analytic_gradient_relative_error": analytic_relative_error,
            "pcg_iterations": int(audit.iterations),
            "pcg_relative_residual": float(audit.relative_residual),
            "all_finite": bool(
                torch.isfinite(g_delta).all()
                and torch.isfinite(g_direct).all()
                and all(torch.isfinite(value).all() for pair in vectors.values() for value in pair)
            ),
        }
        return rows, audit_row
    finally:
        hook.remove()


def _summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary = [row for row in rows if int(row["repeat"]) == 0]
    result: list[dict[str, Any]] = []
    for layer in LAYERS:
        selected = [row for row in primary if row["layer"] == layer]
        cosine = np.asarray([row["cosine"] for row in selected], dtype=np.float64)
        ratio = np.asarray([row["magnitude_ratio"] for row in selected], dtype=np.float64)
        align = np.asarray([row["alignment_ratio"] for row in selected], dtype=np.float64)
        result.append(
            {
                "layer": layer,
                "samples": len(selected),
                "cosine_mean": float(cosine.mean()),
                "cosine_median": float(np.median(cosine)),
                "cosine_p10": float(np.quantile(cosine, 0.10)),
                "cosine_p25": float(np.quantile(cosine, 0.25)),
                "cosine_p75": float(np.quantile(cosine, 0.75)),
                "cosine_p90": float(np.quantile(cosine, 0.90)),
                "cosine_minimum": float(cosine.min()),
                "cosine_maximum": float(cosine.max()),
                "fraction_cosine_negative": float(np.mean(cosine < 0)),
                "fraction_cosine_below_minus_0p25": float(np.mean(cosine < -0.25)),
                "fraction_cosine_above_plus_0p25": float(np.mean(cosine > 0.25)),
                "lap_norm_mean": float(np.mean([row["lap_norm"] for row in selected])),
                "direct_norm_mean": float(np.mean([row["direct_norm"] for row in selected])),
                "magnitude_ratio_mean": float(ratio.mean()),
                "magnitude_ratio_median": float(np.median(ratio)),
                "alignment_ratio_mean": float(align.mean()),
                "alignment_ratio_median": float(np.median(align)),
            }
        )
    return result


def _noise(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    repeated = [row for row in rows if int(row["sample_index"]) % 5 == 0]
    result: list[dict[str, Any]] = []
    for layer in LAYERS:
        for field in ("cosine", "lap_norm", "direct_norm", "alignment_ratio"):
            per_sample = []
            for sample_id in sorted({row["sample_id"] for row in repeated}):
                values = np.asarray([
                    row[field] for row in repeated if row["layer"] == layer and row["sample_id"] == sample_id
                ], dtype=np.float64)
                if len(values) != 3:
                    continue
                per_sample.append((float(values.mean()), float(values.std()), float(values.max() - values.min())))
            result.append(
                {
                    "layer": layer,
                    "metric": field,
                    "samples": len(per_sample),
                    "mean_of_repeat_means": float(np.mean([item[0] for item in per_sample])),
                    "mean_repeat_std": float(np.mean([item[1] for item in per_sample])),
                    "maximum_repeat_range": float(max(item[2] for item in per_sample)),
                }
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run_config = _read(args.run.resolve() / "run_config.json")
    config = run_config.get("experiment_config", run_config)
    device = torch.device(args.device)
    model = _build_model(config, None, False).to(device)
    load_checkpoint(args.checkpoint.resolve(), model, map_location=device)
    model.eval()
    if not model.hybrid_direct_head_enabled:
        raise RuntimeError("Expected a shared-backbone hybrid direct head.")
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "validation")
    amp_enabled, amp_dtype = _amp_settings(config, device)
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    repeated_indices = {index for index in range(count) if index % 5 == 0}
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index in range(count):
        repeats = 3 if index in repeated_indices else 1
        for repeat in range(repeats):
            sample_rows, audit = _one(
                model, dataset, index, config, device, amp_enabled, amp_dtype, repeat
            )
            rows.extend(sample_rows)
            audit_rows.append(audit)
            print(f"gradient {index + 1}/{count} repeat={repeat + 1}/{repeats} {audit['sample_id']}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    aggregate = _summary(rows)
    noise = _noise(rows)
    checks = {
        "read_only": True,
        "samples": count == 50,
        "ten_samples_repeated_three_times": sum(row["repeat"] == 2 for row in audit_rows) == 10,
        "all_finite": all(row["all_finite"] for row in audit_rows),
        "all_pcg_converged_to_tolerance": all(row["pcg_relative_residual"] <= TOLERANCE * 1.05 for row in audit_rows),
        "analytic_gradient_matches": max(row["analytic_gradient_relative_error"] for row in audit_rows) <= 1e-5,
    }
    payload = {
        "contract_audit": all(checks.values()),
        "contract_checks": checks,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "gradient_method": "independent exact VJPs: J_delta(Phi)^T(Lz) and J_direct(Phi)^T(lambda z); output heads excluded from shared-parameter cosine",
        "rows": rows,
        "solver_audit_rows": audit_rows,
        "aggregate": aggregate,
        "fp16_repeat_noise": noise,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gradient_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "gradient_per_sample.csv", rows)
    _write_csv(args.output_dir / "gradient_aggregate.csv", aggregate)
    _write_csv(args.output_dir / "gradient_fp16_repeat_noise.csv", noise)
    _write_csv(args.output_dir / "gradient_solver_audit.csv", audit_rows)
    print(json.dumps({"contract_audit": payload["contract_audit"], "output": str(args.output_dir)}, indent=2))
    if not payload["contract_audit"] and args.limit is None:
        raise RuntimeError(f"Gradient mechanism audit failed: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
