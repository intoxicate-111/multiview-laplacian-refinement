#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate old-domain Arm B/E on one explicitly authorized split."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from evaluate_sofa50_recovery_aware_ablation import _infer_recovery_arm, _load_spec
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.controlled_displacement import displacement_target
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import recovery_forward_audit
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint(run: Path) -> Path:
    for name in ("checkpoint_best.pt", "best.pt"):
        path = run / name
        if path.is_file():
            return path
    raise FileNotFoundError(run)


def load_e(run: Path, device: torch.device) -> dict[str, Any]:
    payload = read_json(run / "run_config.json")
    config = payload.get("experiment_config", payload)
    selected = checkpoint(run)
    model = _build_model(config, None, False).to(device)
    load_checkpoint(selected, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    return {
        "config": config,
        "checkpoint": str(selected),
        "checkpoint_sha256": sha256_file(selected),
        "model": model,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
        "parameter_count": sum(value.numel() for value in model.parameters()),
    }


def infer_e(
    dataset: PreparedMeshDataset,
    index: int,
    spec: Mapping[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    prepared = _load_device_item(dataset, index, spec["config"], device)
    conditioned = _exact_query_sample(prepared.sample, device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=spec["amp_dtype"],
        enabled=bool(spec["amp_enabled"]),
    ):
        output = spec["model"](conditioned)
    if output.confidence_prediction is not None:
        raise RuntimeError("Arm E unexpectedly has confidence output")
    prediction = output.predicted_laplacian.float().detach().cpu().numpy().astype(np.float64)
    target = displacement_target(dataset.load_static(index)).cpu().numpy().astype(np.float64)
    return prediction, target


def pcg(
    delta: np.ndarray,
    anchor: np.ndarray,
    static: Mapping[str, Any],
    regularization: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    with torch.no_grad():
        vertices, audit = recovery_forward_audit(
            torch.as_tensor(delta, dtype=torch.float64, device=device),
            torch.as_tensor(anchor, dtype=torch.float64, device=device),
            torch.as_tensor(static["edge_index"], dtype=torch.long, device=device),
            torch.as_tensor(static["vertex_degree"], dtype=torch.float64, device=device),
            regularization=regularization,
            maximum_iterations=2048,
            tolerance=1e-8,
        )
    return vertices.cpu().numpy(), {
        "pcg_iterations": int(audit.iterations),
        "pcg_converged": bool(audit.converged),
        "pcg_relative_residual": float(audit.relative_residual),
    }


def geometry_row(
    arm: str,
    split: str,
    sample_id: str,
    vertices: np.ndarray,
    initial: Mesh,
    clean: Mesh,
    solver: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metric = _geometry_row(
        split,
        sample_id,
        arm,
        Mesh(vertices, initial.faces.copy()).ensure_normals(),
        clean,
        initial,
    )
    initial_metric = _geometry_row(split, sample_id, "initial", initial, clean, initial)
    initial_cd = float(initial_metric["chamfer"])
    refined_cd = float(metric["chamfer"])
    result = {
        "arm": arm,
        "split": split,
        "sample_id": sample_id,
        "vertices": initial.num_vertices,
        "faces": initial.num_faces,
        "initial_chamfer": initial_cd,
        "refined_chamfer": refined_cd,
        "relative_chamfer_gain": (initial_cd - refined_cd) / initial_cd,
        "p2s": float(metric["p2s"]),
        "p2s_p95": float(metric["p2s_p95"]),
        "fscore": float(metric["fscore"]),
        "normal_consistency": float(metric["normal_consistency"]),
        "introduced_flipped_faces": int(metric["introduced_flipped_faces"]),
        "new_degenerate_faces": int(metric["new_degenerate_faces"]),
        "same_index_recovered_vertex_rms": float(
            np.sqrt(np.mean(np.sum((vertices - clean.vertices) ** 2, axis=1)))
        ),
        "improved": refined_cd < initial_cd,
        "worsened": refined_cd > initial_cd,
    }
    if solver is not None:
        result.update(solver)
    return result


def aggregate(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    chosen = [row for row in rows if row["arm"] == arm]
    return {
        "arm": arm,
        "samples": len(chosen),
        "initial_chamfer": float(np.mean([row["initial_chamfer"] for row in chosen])),
        "refined_chamfer": float(np.mean([row["refined_chamfer"] for row in chosen])),
        "p2s_p95": float(np.mean([row["p2s_p95"] for row in chosen])),
        "fscore": float(np.mean([row["fscore"] for row in chosen])),
        "normal_consistency": float(np.mean([row["normal_consistency"] for row in chosen])),
        "same_index_recovered_vertex_rms": float(
            np.mean([row["same_index_recovered_vertex_rms"] for row in chosen])
        ),
        "introduced_flipped_faces": int(
            sum(row["introduced_flipped_faces"] for row in chosen)
        ),
        "new_degenerate_faces": int(sum(row["new_degenerate_faces"] for row in chosen)),
        "improved": int(sum(row["improved"] for row in chosen)),
        "worsened": int(sum(row["worsened"] for row in chosen)),
    }


def authorize_split(split: str, authorization: Path | None) -> None:
    if split == "validation":
        return
    if authorization is None or not authorization.is_file():
        raise RuntimeError("Test split is sealed; final-selection authorization is required")
    payload = read_json(authorization)
    required = (
        payload.get("final_selection_locked") is True
        and payload.get("validation_only_selection") is True
        and payload.get("authorize_single_test_open") is True
    )
    if not required:
        raise RuntimeError("Test authorization contract is incomplete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-b-run", required=True, type=Path)
    parser.add_argument("--arm-e-run", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--test-authorization", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    authorize_split(args.split, args.test_authorization)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), args.split)
    expected = 25
    if len(dataset) != expected:
        raise RuntimeError(f"Expected {expected} {args.split} samples, found {len(dataset)}")
    b_spec = _load_spec(args.arm_b_run.resolve(), device)
    e_spec = load_e(args.arm_e_run.resolve(), device)
    if b_spec["parameter_count"] != 826115 or e_spec["parameter_count"] != 826115:
        raise RuntimeError("Specialist parameter count mismatch")
    if b_spec["config"]["dataset"] != e_spec["config"]["dataset"]:
        raise RuntimeError("B/E dataset configs differ")

    ids: list[str] = []
    offsets = [0]
    b_predictions: list[np.ndarray] = []
    e_predictions: list[np.ndarray] = []
    b_targets: list[np.ndarray] = []
    e_targets: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        b_values = _infer_recovery_arm(dataset, index, b_spec, device)
        b_prediction = b_values["prediction_raw"].numpy().astype(np.float64)
        b_target = b_values["target_raw"].numpy().astype(np.float64)
        e_prediction, e_target = infer_e(dataset, index, e_spec, device)
        initial = Mesh(
            np.asarray(static["vertices"], dtype=np.float64),
            np.asarray(static["faces"], dtype=np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        b_vertices, b_solver = pcg(
            b_prediction, initial.vertices, static, 0.01, device
        )
        if not b_solver["pcg_converged"]:
            raise RuntimeError(f"{sample_id}: Arm-B PCG failed")
        e_vertices = initial.vertices + e_prediction
        rows.append(
            geometry_row("B_recovery_aware", args.split, sample_id, b_vertices, initial, clean, b_solver)
        )
        rows.append(
            geometry_row("E_direct_vertex", args.split, sample_id, e_vertices, initial, clean, None)
        )
        ids.append(sample_id)
        offsets.append(offsets[-1] + initial.num_vertices)
        b_predictions.append(b_prediction)
        e_predictions.append(e_prediction)
        b_targets.append(b_target)
        e_targets.append(e_target)
        print(f"{args.split} {index + 1}/{len(dataset)} {sample_id}", flush=True)
        torch.cuda.empty_cache()

    arrays_path = output / f"{args.split}_specialist_predictions.npz"
    np.savez_compressed(
        arrays_path,
        sample_ids=np.asarray(ids),
        offsets=np.asarray(offsets, dtype=np.int64),
        b_prediction=np.concatenate(b_predictions),
        e_displacement=np.concatenate(e_predictions),
        b_target=np.concatenate(b_targets),
        e_target=np.concatenate(e_targets),
    )
    with (output / f"{args.split}_specialist_per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "contract_audit": True,
        "split": args.split,
        "test_opened": args.split == "test",
        "samples": len(dataset),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "arm_b_checkpoint": b_spec["checkpoint"],
        "arm_b_checkpoint_sha256": b_spec["checkpoint_sha256"],
        "arm_e_checkpoint": e_spec["checkpoint"],
        "arm_e_checkpoint_sha256": e_spec["checkpoint_sha256"],
        "parameter_counts": {"B": b_spec["parameter_count"], "E": e_spec["parameter_count"]},
        "arrays": str(arrays_path),
        "arrays_sha256": sha256_file(arrays_path),
        "metric_protocol": METRIC_PROTOCOL,
        "aggregate": [aggregate(rows, arm) for arm in ("B_recovery_aware", "E_direct_vertex")],
    }
    (output / f"{args.split}_specialist_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
