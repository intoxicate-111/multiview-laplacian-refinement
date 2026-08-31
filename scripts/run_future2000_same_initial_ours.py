#!/usr/bin/env python3
from __future__ import annotations

"""Run one frozen learned-Laplacian arm on selected Future2000 test samples."""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_future2000_external_baseline import _audit_source_identity, _evaluate
from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.learned_laplacian.canonical_experiment import (
    _exact_query_sample,
    _load_device_item,
)
from mlr.learned_laplacian.canonical_pipeline import (
    canonical_current_graph_recovery_inputs,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.synthetic_current_h2_ablation import (
    _recover_raw_one,
)
from mlr.learned_laplacian.target_scaling import (
    EDGE_SCALE_NORMALIZED_LAPLACIAN,
    RAW_LAPLACIAN,
    normalize_laplacian_by_edge_scale,
    prediction_to_raw_laplacian,
)
from run_sofa50_same_initial_ours import spec


def _infer_laplacian_one(
    dataset: PreparedMeshDataset,
    index: int,
    model_spec: dict[str, Any],
    device: torch.device,
    *,
    current_faces: torch.Tensor | np.ndarray,
) -> dict[str, torch.Tensor | float]:
    """Infer a canonical Laplacian arm with or without a confidence head."""

    config = model_spec["config"]
    prepared = _load_device_item(dataset, index, config, device)
    conditioned = _exact_query_sample(prepared.sample, device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=model_spec["amp_dtype"],
        enabled=bool(model_spec["amp_enabled"]),
    ):
        model_output = model_spec["model"](conditioned)
    prediction_output = model_output.predicted_laplacian.float().detach().cpu()
    h = prepared.sample["local_edge_length"].float().detach().cpu()
    valid = prepared.sample["valid_scale_mask"].bool().detach().cpu()
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))
    target_raw = prepared.raw_target.float().detach().cpu()
    target_normalized = normalize_laplacian_by_edge_scale(
        target_raw, h, eps=epsilon, valid_scale_mask=valid
    )
    target_mode = str(config.get("target_mode"))
    prediction_raw = prediction_to_raw_laplacian(
        prediction_output,
        h,
        input_representation=target_mode,
        eps=epsilon,
    )
    if target_mode == EDGE_SCALE_NORMALIZED_LAPLACIAN:
        prediction_normalized = prediction_output
    elif target_mode == RAW_LAPLACIAN:
        prediction_normalized = normalize_laplacian_by_edge_scale(
            prediction_raw, h, eps=epsilon, valid_scale_mask=valid
        )
    else:
        raise ValueError(f"Unsupported target_mode {target_mode!r}")

    confidence_prediction = model_output.confidence_prediction
    confidence = (
        torch.ones(len(prediction_raw), dtype=torch.float32)
        if confidence_prediction is None
        else confidence_prediction.float().detach().cpu()
    )
    visibility = prepared.sample["visibility"].detach().cpu()
    canonical = canonical_current_graph_recovery_inputs(
        prepared.sample["vertices"].detach().cpu(),
        current_faces,
        prediction_normalized,
        visibility,
        None if confidence_prediction is None else confidence,
        epsilon=epsilon,
    )
    roundtrip_error = torch.max(
        torch.abs(canonical.delta_pred_raw.cpu() - prediction_raw)
    ).item()
    return {
        "prediction_output": prediction_output,
        "prediction_raw": prediction_raw,
        "prediction_normalized": prediction_normalized,
        "target_raw": target_raw,
        "target_normalized": target_normalized,
        "confidence": confidence,
        "h": h,
        "valid": valid,
        "visibility_count": visibility.to(torch.int64).sum(dim=0),
        "recovery_weight": canonical.weight.detach().cpu(),
        "target_confidence": prepared.sample["target_confidence"]
        .float()
        .detach()
        .cpu(),
        "roundtrip_error": float(roundtrip_error),
    }


def _recover_all_vertices_one(
    static: dict[str, Any],
    prediction_raw: torch.Tensor,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the formal all-vertex Arm-B standalone sparse recovery."""

    recovery = config.get("recovery", {})
    if not (
        recovery.get("operator_type") == "uniform_random_walk_current_graph"
        and recovery.get("anchor") == "lambda_times_input_vertex_l2"
        and recovery.get("laplacian_equations") == "all_vertices"
        and recovery.get("visibility_gate") is False
        and recovery.get("confidence_weighting") is False
        and recovery.get("standalone_implementation")
        == "scipy_lsmr_augmented_system"
    ):
        raise RuntimeError("Unsupported formal standalone recovery contract")
    regularization = float(recovery["lambda"])
    initial_vertices = (
        torch.as_tensor(static["vertices"]).detach().cpu().numpy().astype(np.float64)
    )
    faces = (
        torch.as_tensor(static["faces"]).detach().cpu().numpy().astype(np.int64)
    )
    laplacian, laplacian_data = uniform_sparse_laplacian(
        faces, len(initial_vertices)
    )
    component_count, labels = component_labels(laplacian_data)
    recovered, audit = regularized_sparse_solve(
        laplacian,
        prediction_raw.detach().cpu().numpy().astype(np.float64),
        initial_vertices,
        labels,
        component_count,
        regularization,
        atol=1e-12,
        btol=1e-12,
        maxiter=100_000,
    )
    if not bool(audit["all_converged"]):
        raise RuntimeError("Arm-B standalone LSMR recovery did not converge")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_mesh(
        Mesh(recovered, faces.copy()).ensure_normals(),
        output_dir / "predicted_refined.obj",
    )
    np.save(output_dir / "predicted_vertices.npy", recovered)
    (output_dir / "solver_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    dataset = PreparedMeshDataset.from_manifest(args.manifest, "test")
    if len(dataset) != args.expected_test_samples:
        raise ValueError(
            f"Expected {args.expected_test_samples} test samples, found {len(dataset)}"
        )
    if args.selection is None:
        selected_ids = [str(value) for value in dataset.sample_ids]
    else:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        selected_ids = [str(value) for value in selection["sample_ids"]]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Selection contains duplicate sample IDs")
    unknown = sorted(set(selected_ids) - set(dataset.sample_ids))
    if unknown:
        raise ValueError(f"Selected sample IDs are absent from test split: {unknown}")
    if args.sample_id is None:
        assigned_ids = selected_ids[args.shard_index :: args.shard_count]
    else:
        if args.sample_id not in selected_ids:
            raise ValueError("--sample-id is not part of the frozen selection")
        assigned_ids = [args.sample_id]

    provenance_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    provenance = {
        str(row["sample_id"]): dict(row) for row in provenance_payload["samples"]
    }
    index_by_id = {str(value): index for index, value in enumerate(dataset.sample_ids)}
    model_spec = spec(
        args.run_dir.resolve(),
        device,
        view_chunk_size=args.view_chunk_size,
        checkpoint_name=args.checkpoint_name,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    output = args.output_dir.resolve() / "ours"
    rows: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    for sample_id in assigned_ids:
        index = index_by_id[sample_id]
        static = dataset.load_static(index)
        source_identity = _audit_source_identity(static, provenance[sample_id])
        sample_dir = output / "samples" / sample_id
        recovery_dir = sample_dir / "recovery"
        sample_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        values = _infer_laplacian_one(
            dataset,
            index,
            model_spec,
            device,
            current_faces=static["faces"],
        )
        recovery_config = model_spec["config"].get("recovery", {})
        formal_all_vertex_recovery = bool(
            recovery_config.get("laplacian_equations") == "all_vertices"
            and recovery_config.get("visibility_gate") is False
            and recovery_config.get("confidence_weighting") is False
            and recovery_config.get("standalone_implementation")
            == "scipy_lsmr_augmented_system"
        )
        if formal_all_vertex_recovery:
            recovery_audit = _recover_all_vertices_one(
                static,
                values["prediction_raw"],
                recovery_dir,
                model_spec["config"],
            )
        else:
            recovery_audit, _ = _recover_raw_one(
                static,
                values["prediction_raw"],
                values["prediction_normalized"],
                values["confidence"],
                recovery_dir,
                model_spec["config"],
            )
        final_mesh = sample_dir / "refined.obj"
        shutil.copy2(recovery_dir / "predicted_refined.obj", final_mesh)
        current_mesh = Mesh(
            static["vertices"].detach().cpu().numpy(),
            static["faces"].detach().cpu().numpy(),
        ).ensure_normals()
        save_mesh(current_mesh, sample_dir / "initial.obj")
        from mlr.io import load_mesh

        refined = load_mesh(final_mesh).ensure_normals()
        metrics = _evaluate(static, refined, args)
        runtime = time.perf_counter() - started
        peak = (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else None
        )
        np.save(sample_dir / "predicted_raw_laplacian.npy", values["prediction_raw"].numpy())
        np.save(sample_dir / "predicted_confidence.npy", values["confidence"].numpy())
        row = {
            "sample_id": sample_id,
            "method": "ours",
            "status": "completed",
            "failure_stage": "",
            "failure_reason": "",
            "runtime_seconds": runtime,
            "peak_gpu_memory_mb": peak,
            "vertex_count": refined.num_vertices,
            "face_count": refined.num_faces,
            "final_mesh": str(final_mesh),
            "coordinate_transform_to_gt": "identity",
            "method_config_path": str(args.run_dir.resolve() / "run_config.json"),
            "checkpoint": str(model_spec["checkpoint"]),
            "checkpoint_sha256": model_spec["checkpoint_sha256"],
            "checkpoint_epoch": model_spec["checkpoint_epoch"],
            "checkpoint_optimizer_steps": model_spec["optimizer_steps"],
            "inference_view_chunk_size": model_spec["inference_view_chunk_size"],
            "formal_all_vertex_recovery": formal_all_vertex_recovery,
            "recovery_solver_all_converged": bool(
                recovery_audit.get("all_converged", True)
            ),
            "recovery_solver_runtime_seconds": recovery_audit.get(
                "runtime_seconds"
            ),
            **source_identity,
            "adapter_initial_mesh_sha256": source_identity["common_initial_mesh_sha256"],
            "adapter_initial_vertex_count": source_identity["initial_vertex_count"],
            "adapter_initial_face_count": source_identity["initial_face_count"],
            "adapter_initial_max_abs_vertex_error": 0.0,
            "adapter_initial_faces_exact": True,
            "common_initial_identity_audit": True,
            **metrics,
        }
        (sample_dir / "status.json").write_text(
            json.dumps({"status": "completed", "row": row}, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(row)
        print(
            f"ours shard={args.shard_index} sample={sample_id} "
            f"chamfer={row['refined_chamfer']:.9g}",
            flush=True,
        )

    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    csv_path = shard_dir / f"per_sample_shard_{args.shard_index:03d}.csv"
    if not rows:
        raise ValueError("Ours shard has no assigned samples")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "method": "ours",
        "status": "completed",
        "pinned_commit": model_spec["checkpoint_sha256"],
        "repository": "intoxicate-111/multiview-laplacian-refinement",
        "manifest": str(args.manifest.resolve()),
        "selection": str(args.selection.resolve()) if args.selection is not None else None,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned_samples": len(rows),
        "completed_samples": len(rows),
        "failed_samples": 0,
        "checkpoint_optimizer_steps": model_spec["optimizer_steps"],
        "checkpoint_epoch": model_spec["checkpoint_epoch"],
        "csv": str(csv_path),
    }
    (shard_dir / f"metadata_shard_{args.shard_index:03d}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        help="Optional frozen subset; omit to evaluate the complete test split.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="checkpoint_latest.pt")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--expected-test-samples", type=int, default=1000)
    parser.add_argument("--view-chunk-size", type=int, default=4)
    parser.add_argument("--surface-samples", type=int, default=3000)
    parser.add_argument("--metric-seed", type=int, default=7)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
