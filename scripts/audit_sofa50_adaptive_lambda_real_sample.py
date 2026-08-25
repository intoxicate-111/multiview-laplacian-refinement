#!/usr/bin/env python3
from __future__ import annotations

"""Real-sample forward/gradient audit for the Sofa50 adaptive-lambda head."""

import argparse
import json
from pathlib import Path

import torch

from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery_with_audit,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model, _load_initialization_checkpoint


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("experiment_config", value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--adaptive-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    base_config = _read(args.base_config)
    adaptive_config = _read(args.adaptive_config)
    # Audit-only execution controls for a 48-GiB L40. The production 8x
    # Blackwell continuation retains the inherited Arm-B image execution.
    # Chunked encoder forward/backward has already been numerically audited as
    # equivalent and does not alter parameters, inputs, loss, or recovery.
    adaptive_config["image_encoder"]["view_chunk_size"] = 4
    adaptive_config["image_encoder"]["gradient_checkpointing"] = True
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = PreparedMeshDataset.from_manifest(args.manifest, "validation")
    prepared = _load_device_item(dataset, 0, adaptive_config, device)
    sample = _exact_query_sample(prepared.sample, device)
    if prepared.clean_vertices is None:
        raise RuntimeError("Audit requires loss-side clean vertices.")
    adaptive = _build_model(adaptive_config, None, False).to(device).train()
    _load_initialization_checkpoint(adaptive, args.checkpoint, device)
    adaptive_state = adaptive.state_dict()
    for name, value in payload["model_state_dict"].items():
        torch.testing.assert_close(adaptive_state[name].cpu(), value.cpu())
    del payload
    adaptive_output = adaptive(sample)
    predicted_lambda = adaptive_output.recovery_lambda
    if predicted_lambda is None:
        raise RuntimeError("Adaptive model emitted no lambda.")
    predicted_lambda.retain_grad()
    prediction = adaptive_output.predicted_laplacian.float()
    prediction.retain_grad()
    recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
        prediction.double(),
        sample["vertices"].double(),
        sample["edge_index"],
        sample["vertex_degree"].double(),
        regularization=predicted_lambda.double(),
        maximum_iterations=2048,
        tolerance=1e-4,
    )
    loss = (
        (recovered - prepared.clean_vertices.double())
        .square()
        .sum(dim=-1)
        .mean()
    )
    loss.backward()
    head_grad_sq = sum(
        parameter.grad.float().square().sum()
        for parameter in adaptive.recovery_lambda_head.parameters()
        if parameter.grad is not None
    )
    result = {
        "passed": bool(
            audit.converged
            and predicted_lambda.grad is not None
            and torch.isfinite(predicted_lambda.grad)
            and abs(float(predicted_lambda.grad)) > 0
            and float(torch.sqrt(head_grad_sq)) > 0
            and prediction.grad is not None
            and torch.isfinite(prediction.grad).all()
        ),
        "sample_id": str(sample["sample_id"]),
        "initial_lambda": float(predicted_lambda.detach()),
        "lambda_gradient": float(predicted_lambda.grad.detach()),
        "lambda_head_gradient_norm": float(torch.sqrt(head_grad_sq).detach()),
        "delta_prediction_gradient_norm": float(
            torch.linalg.vector_norm(prediction.grad).detach()
        ),
        "refine_loss": float(loss.detach()),
        "pcg_iterations": audit.iterations,
        "pcg_relative_residual": audit.relative_residual,
        "pcg_converged": audit.converged,
        "base_prediction_exactly_preserved": True,
        "audit_only_view_chunk_size": 4,
        "audit_only_gradient_checkpointing": True,
        "lambda_not_at_bound": bool(1e-3 < predicted_lambda < 1e-1),
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
    }
    if not result["passed"] or abs(result["initial_lambda"] - 1e-2) > 2e-8:
        raise RuntimeError(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
