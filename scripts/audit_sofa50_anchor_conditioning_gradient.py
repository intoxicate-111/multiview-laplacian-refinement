#!/usr/bin/env python3
"""Pre-training gradient and isolation audit for the Arm-B_P ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.frozen_anchor_cache import FrozenAnchorCache, FrozenAnchorDataset
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import (
    _build_model,
    _recovery_aware_geometry_settings,
    _recovery_refine_loss_with_audit,
)
from mlr.learned_laplacian.trainer import load_checkpoint


EXPECTED_E_SHA256 = "6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1"


def gradient_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total = total + parameter.grad.float().square().sum()
    return float(torch.sqrt(total).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--bp-config", required=True, type=Path)
    parser.add_argument("--arm-e-config", required=True, type=Path)
    parser.add_argument("--arm-e-checkpoint", required=True, type=Path)
    parser.add_argument("--anchor-cache-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()

    device = torch.device(args.device)
    bp_config = json.loads(args.bp_config.read_text(encoding="utf-8"))
    e_config = json.loads(args.arm_e_config.read_text(encoding="utf-8"))
    cache = FrozenAnchorCache(
        args.anchor_cache_metadata,
        expected_checkpoint_sha256=EXPECTED_E_SHA256,
    )
    base_dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "train")
    bp_dataset = FrozenAnchorDataset(base_dataset, cache)
    prepared = _load_device_item(bp_dataset, 0, bp_config, device)
    if "recovery_anchor_vertices" in prepared.sample:
        raise RuntimeError("Frozen E anchor leaked into the Arm-B predictor mapping")
    if prepared.recovery_anchor_vertices is None:
        raise RuntimeError("Prepared Arm-B_P item has no loss-side frozen anchor")
    if prepared.recovery_anchor_vertices.requires_grad:
        raise RuntimeError("Frozen E anchor unexpectedly requires gradients")

    torch.manual_seed(int(bp_config.get("seed", 7)))
    bp_model = _build_model(bp_config, None, False).to(device)
    bp_model.train()
    conditioned = _exact_query_sample(prepared.sample, device)
    bp_amp_enabled, bp_amp_dtype = _amp_settings(bp_config, device)
    with torch.autocast(
        device_type=device.type,
        dtype=bp_amp_dtype,
        enabled=bp_amp_enabled,
    ):
        bp_output = bp_model(conditioned)
    delta_prediction = bp_output.predicted_laplacian.float()
    delta_prediction.retain_grad()
    settings = _recovery_aware_geometry_settings(bp_config)
    refine_loss, recovered, solve_audit = _recovery_refine_loss_with_audit(
        delta_prediction, prepared, settings
    )
    refine_loss.backward()
    prediction_gradient_norm = float(
        torch.linalg.vector_norm(delta_prediction.grad).item()
    ) if delta_prediction.grad is not None else 0.0
    model_gradient_norm = gradient_norm(bp_model)

    e_model = _build_model(e_config, None, False).to(device)
    load_checkpoint(args.arm_e_checkpoint, e_model, map_location=device)
    e_model.eval()
    for parameter in e_model.parameters():
        parameter.requires_grad_(False)
    e_prepared = _load_device_item(base_dataset, 0, e_config, device)
    e_conditioned = _exact_query_sample(e_prepared.sample, device)
    e_amp_enabled, e_amp_dtype = _amp_settings(e_config, device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=e_amp_dtype,
        enabled=e_amp_enabled,
    ):
        e_output = e_model(e_conditioned)
    fresh_anchor = e_prepared.sample["vertices"] + e_output.predicted_laplacian.float()
    anchor_difference = float(
        torch.max(torch.abs(fresh_anchor - prepared.recovery_anchor_vertices)).item()
    )
    e_gradients_present = any(parameter.grad is not None for parameter in e_model.parameters())
    passed = (
        prediction_gradient_norm > 0
        and model_gradient_norm > 0
        and torch.isfinite(refine_loss)
        and torch.isfinite(recovered).all()
        and not e_gradients_present
        and bool(solve_audit.converged)
    )
    audit = {
        "contract_audit": bool(passed),
        "sample_id": str(prepared.sample["sample_id"]),
        "bp_predictor_keys": sorted(prepared.sample),
        "frozen_anchor_present_in_predictor_mapping": False,
        "frozen_anchor_requires_grad": prepared.recovery_anchor_vertices.requires_grad,
        "delta_prediction_gradient_norm_from_recovery_loss": prediction_gradient_norm,
        "bp_model_gradient_norm_from_recovery_loss": model_gradient_norm,
        "recovery_refine_loss": float(refine_loss.detach().item()),
        "recovery_converged": bool(solve_audit.converged),
        "recovery_iterations": int(solve_audit.iterations),
        "recovery_relative_residual": float(solve_audit.relative_residual),
        "arm_e_training_mode": e_model.training,
        "arm_e_parameters_require_grad": any(
            parameter.requires_grad for parameter in e_model.parameters()
        ),
        "arm_e_gradients_present": e_gradients_present,
        "fresh_reinference_vs_canonical_cached_anchor_max_abs": anchor_difference,
        "canonical_training_anchor_policy": (
            "the audited cached file is reused unchanged throughout B_P training"
        ),
        "reinference_warning": (
            "CUDA/AMP Arm-E reinference is not assumed bitwise deterministic; "
            "this observed difference is provenance only and does not replace "
            "the canonical cached anchor"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Arm-B_P pre-training gradient/isolation audit failed")


if __name__ == "__main__":
    main()
