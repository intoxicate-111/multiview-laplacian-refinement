#!/usr/bin/env python3
from __future__ import annotations

"""Contract-gated frozen Arm-B/Arm-E evaluation on compatible OOD domains."""

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from evaluate_sofa50_recovery_aware_ablation import _load_spec
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.differentiable_sparse_recovery import recovery_forward_audit
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


ARM_B = "B_lap_plus_refine"
ARM_E = "E_direct_vertex_residual"
ARM_H = "Hybrid_B_laplacian_E_anchor"
FORBIDDEN_MODEL_INPUT_FIELDS = {
    "clean_reference_vertices",
    "clean_reference_faces",
    "gt_vertices",
    "gt_faces",
    "target_positions",
    "raw_laplacian_target",
    "normalized_laplacian_target",
    "laplacian_target",
    "training_target",
    "clean_vertices",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _inference_loader_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Disable only the training-loss-side clean tensor; model/recovery stay frozen."""
    result = copy.deepcopy(dict(config))
    training = result.setdefault("training", {})
    recovery_loss = training.get("recovery_aware_geometry_loss")
    if isinstance(recovery_loss, Mapping):
        training["recovery_aware_geometry_loss"] = dict(recovery_loss)
        training["recovery_aware_geometry_loss"]["enabled"] = False
    return result


def _set_execution_view_chunk_size(
    specs: Sequence[Mapping[str, Any]], chunk_size: int | None
) -> None:
    """Apply the model's tested execution-only view chunking at inference.

    Chunking changes neither the checkpoint nor the 28-view aggregation.  The
    model encodes/projects each view chunk independently, concatenates the
    per-view samples in their original order, and then performs the unchanged
    all-view aggregation.
    """

    if chunk_size is None:
        return
    if chunk_size < 1:
        raise ValueError("view_chunk_size must be positive")
    for spec in specs:
        spec["model"].image_view_chunk_size = int(chunk_size)


def _cuda_timed_forward(model, sample: Mapping[str, Any], spec: Mapping[str, Any], device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=spec["amp_dtype"],
        enabled=bool(spec["amp_enabled"]),
    ):
        output = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if output.confidence_prediction is not None:
        raise RuntimeError("Frozen B/E representation arms must not instantiate confidence")
    return output.predicted_laplacian.float().detach().cpu().numpy().astype(np.float64), elapsed


def _input_contract(static: Mapping[str, Any], prepared_sample: Mapping[str, Any]) -> dict[str, Any]:
    intrinsics = np.asarray(static["intrinsics"])
    extrinsics = np.asarray(static["extrinsics"])
    image_paths = static.get("image_paths", [])
    forbidden_present = sorted(FORBIDDEN_MODEL_INPUT_FIELDS.intersection(prepared_sample))
    return {
        "view_count": len(image_paths) if image_paths else int(intrinsics.shape[0]),
        "prepared_image_size": int(static.get("prepared_image_size", 0)),
        "intrinsics_shape": list(intrinsics.shape),
        "extrinsics_shape": list(extrinsics.shape),
        "model_input_fields": sorted(prepared_sample),
        "forbidden_gt_or_target_model_input_fields": forbidden_present,
        "passed": bool(
            (len(image_paths) if image_paths else int(intrinsics.shape[0])) == 28
            and int(static.get("prepared_image_size", 0)) == 960
            and intrinsics.shape == (28, 3, 3)
            and extrinsics.shape == (28, 4, 4)
            and not forbidden_present
        ),
    }


def _hybrid_pcg(
    b_prediction: np.ndarray,
    direct_vertices: np.ndarray,
    static: Mapping[str, Any],
    regularization: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        result, audit = recovery_forward_audit(
            torch.as_tensor(b_prediction, dtype=torch.float64, device=device),
            torch.as_tensor(direct_vertices, dtype=torch.float64, device=device),
            torch.as_tensor(static["edge_index"], dtype=torch.long, device=device),
            torch.as_tensor(static["vertex_degree"], dtype=torch.float64, device=device),
            regularization=float(regularization),
            maximum_iterations=2048,
            tolerance=1e-4,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    if not audit.converged:
        raise RuntimeError(
            f"Hybrid float64 PCG failed: iterations={audit.iterations}, "
            f"relative_residual={audit.relative_residual:.6g}"
        )
    return result.detach().cpu().numpy(), {
        "pcg_iterations": audit.iterations,
        "pcg_relative_residual": audit.relative_residual,
        "pcg_converged": audit.converged,
        "pcg_dtype": "float64",
        "pcg_tolerance": 1e-4,
        "pcg_maximum_iterations": 2048,
        "pcg_runtime_seconds": runtime,
    }


def evaluate_shard(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    shard_path = output / "shards" / f"{args.domain_name}_{args.shard_index:02d}.json"
    if shard_path.is_file():
        print(f"resume: {shard_path}")
        return
    device = torch.device(args.device)
    b_spec = _load_spec(args.arm_b_run.resolve(), device)
    e_spec = _load_spec(args.arm_e_run.resolve(), device)
    _set_execution_view_chunk_size((b_spec, e_spec), args.view_chunk_size)
    if b_spec["parameter_count"] != e_spec["parameter_count"]:
        raise RuntimeError("Arm B/E parameter counts differ")
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    if len(dataset) < 1:
        raise RuntimeError("OOD manifest contains no test samples")
    b_lambda = float(b_spec["config"]["recovery"]["lambda"])
    if b_lambda != 1e-2:
        raise RuntimeError(f"Frozen Arm B lambda changed: {b_lambda}")
    loader_config = _inference_loader_config(b_spec["config"])
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    limit = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    selected = [index for index in range(limit) if index % args.shard_count == args.shard_index]
    try:
        for progress, index in enumerate(selected, start=1):
            static = dataset.load_static(index)
            sample_id = str(static["sample_id"])
            prepared = _load_device_item(dataset, index, loader_config, device)
            conditioned = _exact_query_sample(prepared.sample, device)
            audit = _input_contract(static, conditioned)
            audit.update({"sample_id": sample_id, "domain": args.domain_name})
            if not audit["passed"]:
                raise RuntimeError(f"{sample_id}: input contract failed: {audit}")

            # Both frozen models receive this exact same mapping. The clean mesh
            # remains outside it and is accessed only after both forwards finish.
            b_prediction, b_inference = _cuda_timed_forward(
                b_spec["model"], conditioned, b_spec, device
            )
            e_displacement, e_inference = _cuda_timed_forward(
                e_spec["model"], conditioned, e_spec, device
            )
            initial = Mesh(
                np.asarray(static["vertices"], dtype=np.float64),
                np.asarray(static["faces"], dtype=np.int64),
            ).ensure_normals()
            clean = _clean_mesh(static)
            correspondence_valid = bool(
                initial.vertices.shape == clean.vertices.shape
                and np.array_equal(initial.faces, clean.faces)
            )
            if not correspondence_valid:
                raise RuntimeError(f"{sample_id}: same-index clean correspondence is invalid")
            laplacian, lap_data = uniform_sparse_laplacian(initial.faces, initial.num_vertices)
            component_count, labels = component_labels(lap_data)
            b_vertices, solver = regularized_sparse_solve(
                laplacian,
                b_prediction,
                initial.vertices,
                labels,
                component_count,
                b_lambda,
                atol=1e-12,
                btol=1e-12,
                maxiter=100000,
            )
            if not solver["all_converged"]:
                raise RuntimeError(f"{sample_id}: Arm B LSMR failed")
            e_vertices = initial.vertices + e_displacement
            meshes = {
                "initial": initial,
                ARM_B: Mesh(b_vertices, initial.faces.copy()).ensure_normals(),
                ARM_E: Mesh(e_vertices, initial.faces.copy()).ensure_normals(),
            }
            hybrid_solver = None
            if args.hybrid_lambda is not None:
                hybrid_vertices, hybrid_solver = _hybrid_pcg(
                    b_prediction, e_vertices, static, args.hybrid_lambda, device
                )
                meshes[ARM_H] = Mesh(hybrid_vertices, initial.faces.copy()).ensure_normals()
            geometry = {
                state: _geometry_row(args.domain_name, sample_id, state, mesh, clean, initial)
                for state, mesh in meshes.items()
            }
            initial_cd = float(geometry["initial"]["chamfer"])
            arm_outputs = [
                (ARM_B, b_vertices, b_inference, float(solver["runtime_seconds"]), None),
                (ARM_E, e_vertices, e_inference, 0.0, None),
            ]
            if args.hybrid_lambda is not None:
                arm_outputs.append(
                    (
                        ARM_H,
                        hybrid_vertices,
                        b_inference + e_inference,
                        float(hybrid_solver["pcg_runtime_seconds"]),
                        hybrid_solver,
                    )
                )
            for arm, vertices, inference_runtime, recovery_runtime, pcg_audit in arm_outputs:
                metric = geometry[arm]
                refined_cd = float(metric["chamfer"])
                item = {
                        "domain": args.domain_name,
                        "arm": arm,
                        "sample_id": sample_id,
                        "test_index": index,
                        "initial_chamfer": initial_cd,
                        "refined_chamfer": refined_cd,
                        "relative_chamfer_gain": (initial_cd - refined_cd) / initial_cd,
                        "p2s": float(metric["p2s"]),
                        "p2s_p95": float(metric["p2s_p95"]),
                        "fscore": float(metric["fscore"]),
                        "normal_consistency": float(metric["normal_consistency"]),
                        "introduced_flipped_faces": int(metric["introduced_flipped_faces"]),
                        "normalized_flip_rate": float(metric["introduced_flipped_faces"] / initial.num_faces),
                        "new_degenerate_faces": int(metric["new_degenerate_faces"]),
                        "same_index_recovered_vertex_rms": float(
                            np.sqrt(np.mean(np.sum((vertices - clean.vertices) ** 2, axis=1)))
                        ),
                        "improved": refined_cd < initial_cd,
                        "worsened": refined_cd > initial_cd,
                        "vertices": initial.num_vertices,
                        "faces": initial.num_faces,
                        "model_inference_seconds": inference_runtime,
                        "recovery_seconds": recovery_runtime,
                        "total_inference_recovery_seconds": inference_runtime + recovery_runtime,
                        "hybrid_lambda": args.hybrid_lambda if arm == ARM_H else None,
                    }
                if pcg_audit:
                    item.update(pcg_audit)
                rows.append(item)
            audits.append(audit)
            print(f"{args.domain_name} {progress}/{len(selected)} {sample_id}", flush=True)
            del prepared, conditioned
            torch.cuda.empty_cache()
    except Exception as exc:
        if args.preflight:
            _write_json(
                shard_path,
                {
                    "domain": args.domain_name,
                    "status": "unavailable",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "read_only": True,
                    "execution_view_chunk_size": args.view_chunk_size,
                    "rows": [],
                    "audits": audits,
                },
            )
            print(f"UNAVAILABLE {args.domain_name}: {type(exc).__name__}: {exc}")
            return
        raise
    _write_json(
        shard_path,
        {
            "domain": args.domain_name,
            "status": "available",
            "manifest": str(args.manifest.resolve()),
            "dataset_test_samples": len(dataset),
            "evaluated_samples": len(selected),
            "read_only": True,
            "gt_used_for_inference": False,
            "same_model_input_mapping_for_b_e": True,
            "arm_b_checkpoint": b_spec["checkpoint"],
            "arm_b_checkpoint_sha256": b_spec["checkpoint_sha256"],
            "arm_e_checkpoint": e_spec["checkpoint"],
            "arm_e_checkpoint_sha256": e_spec["checkpoint_sha256"],
            "parameter_count": b_spec["parameter_count"],
            "execution_view_chunk_size": args.view_chunk_size,
            "all_28_views_aggregated_once": True,
            "hybrid_lambda": args.hybrid_lambda,
            "metric_protocol": METRIC_PROTOCOL,
            "rows": rows,
            "audits": audits,
        },
    )


def _parse_domain(value: str) -> tuple[str, Path, int]:
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise ValueError("Domain must be NAME|MANIFEST|SHARDS")
    return parts[0], Path(parts[1]), int(parts[2])


def _aggregate(rows: Sequence[Mapping[str, Any]], arm: str, domain: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    faces = sum(int(row["faces"]) for row in selected)
    return {
        "domain": domain,
        "arm": arm,
        "valid_samples": len(selected),
        "initial_chamfer": float(np.mean([float(row["initial_chamfer"]) for row in selected])),
        "refined_chamfer": float(np.mean([float(row["refined_chamfer"]) for row in selected])),
        "relative_chamfer_gain": float(np.mean([float(row["relative_chamfer_gain"]) for row in selected])),
        "p2s": float(np.mean([float(row["p2s"]) for row in selected])),
        "p2s_p95": float(np.mean([float(row["p2s_p95"]) for row in selected])),
        "fscore": float(np.mean([float(row["fscore"]) for row in selected])),
        "normal_consistency": float(np.mean([float(row["normal_consistency"]) for row in selected])),
        "normalized_flip_rate": sum(int(row["introduced_flipped_faces"]) for row in selected) / faces,
        "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in selected)),
        "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in selected)),
        "same_index_recovered_vertex_rms": float(np.mean([float(row["same_index_recovered_vertex_rms"]) for row in selected])),
        "improved": int(sum(bool(row["improved"]) for row in selected)),
        "worsened": int(sum(bool(row["worsened"]) for row in selected)),
        "model_inference_seconds": float(np.mean([float(row["model_inference_seconds"]) for row in selected])),
        "total_inference_recovery_seconds": float(np.mean([float(row["total_inference_recovery_seconds"]) for row in selected])),
    }


def _paired(rows: Sequence[Mapping[str, Any]], domain: str, left_arm: str = ARM_B, right_arm: str = ARM_E) -> dict[str, Any]:
    left = {row["sample_id"]: row for row in rows if row["arm"] == left_arm}
    right = {row["sample_id"]: row for row in rows if row["arm"] == right_arm}
    if left.keys() != right.keys():
        raise RuntimeError(f"{domain}: {left_arm}/{right_arm} paired IDs differ")
    pairs = [(left[key], right[key]) for key in sorted(left)]
    return {
        "domain": domain,
        "comparison": f"{right_arm}_vs_{left_arm}",
        "samples": len(pairs),
        "right_lower_chamfer": int(sum(float(right["refined_chamfer"]) < float(left["refined_chamfer"]) for left, right in pairs)),
        "right_lower_vertex_rms": int(sum(float(right["same_index_recovered_vertex_rms"]) < float(left["same_index_recovered_vertex_rms"]) for left, right in pairs)),
        "right_lower_p2s_p95": int(sum(float(right["p2s_p95"]) < float(left["p2s_p95"]) for left, right in pairs)),
        "right_higher_fscore": int(sum(float(right["fscore"]) > float(left["fscore"]) for left, right in pairs)),
        "right_higher_normal": int(sum(float(right["normal_consistency"]) > float(left["normal_consistency"]) for left, right in pairs)),
        "right_lower_flip_rate": int(sum(float(right["normalized_flip_rate"]) < float(left["normalized_flip_rate"]) for left, right in pairs)),
    }


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    domains = [_parse_domain(value) for value in args.domain]
    aggregate: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    domain_audits: list[dict[str, Any]] = []
    for name, manifest, shard_count in domains:
        payloads = [
            _read(output / "shards" / f"{name}_{index:02d}.json")
            for index in range(shard_count)
        ]
        unavailable = [payload for payload in payloads if payload["status"] != "available"]
        if unavailable:
            domain_audits.append(
                {
                    "domain": name,
                    "manifest": str(manifest),
                    "available": False,
                    "reasons": [payload.get("reason") for payload in unavailable],
                }
            )
            continue
        rows = [row for payload in payloads for row in payload["rows"]]
        expected = len(PreparedMeshDataset.from_manifest(manifest.resolve(), "test"))
        sample_ids = {row["sample_id"] for row in rows}
        arms = sorted({row["arm"] for row in rows})
        expected_arms = [ARM_B, ARM_E, ARM_H] if ARM_H in arms else [ARM_B, ARM_E]
        if set(arms) != set(expected_arms) or len(rows) != len(expected_arms) * expected or len(sample_ids) != expected:
            raise RuntimeError(f"{name}: expected {len(expected_arms) * expected} rows, found {len(rows)}")
        audit_pass = all(
            payload["read_only"]
            and not payload["gt_used_for_inference"]
            and payload["same_model_input_mapping_for_b_e"]
            and all(audit["passed"] for audit in payload["audits"])
            for payload in payloads
        )
        domain_audits.append(
            {
                "domain": name,
                "manifest": str(manifest),
                "available": True,
                "valid_samples": expected,
                "input_contract_audit": audit_pass,
            }
        )
        aggregate.extend(_aggregate(rows, arm, name) for arm in expected_arms)
        paired.append(_paired(rows, name, ARM_B, ARM_E))
        if ARM_H in expected_arms:
            paired.append(_paired(rows, name, ARM_B, ARM_H))
            paired.append(_paired(rows, name, ARM_E, ARM_H))
        all_rows.extend(rows)
    implementation_audit = bool(
        all(item.get("input_contract_audit", True) for item in domain_audits)
        and any(item["available"] for item in domain_audits)
    )
    summary = {
        "implementation_audit": implementation_audit,
        "read_only": True,
        "models_retrained": False,
        "ood_tuning": False,
        "metric_protocol": METRIC_PROTOCOL,
        "domains": domain_audits,
        "aggregate": aggregate,
        "paired": paired,
    }
    _write_json(output / "ood_summary.json", summary)
    _write_csv(output / "ood_aggregate.csv", aggregate)
    _write_csv(output / "ood_paired.csv", paired)
    _write_csv(output / "ood_per_sample.csv", all_rows)
    lines = [
        "# Frozen Sofa50 Arm B/E/Hybrid OOD evaluation",
        "",
        f"Implementation/read-only audit: **{str(implementation_audit).lower()}**. No fine-tuning or OOD selection was performed.",
        "",
        "| Domain | Arm | Valid | Initial CD | Refined CD | Mean gain | Improved/worsened | P2S p95 | F-score | Normal | Flip rate | New deg. | Vertex RMS | Runtime |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['domain']} | {row['arm']} | {row['valid_samples']} | {row['initial_chamfer']:.8g} | {row['refined_chamfer']:.8g} | {row['relative_chamfer_gain']:.2%} | {row['improved']}/{row['worsened']} | {row['p2s_p95']:.8g} | {row['fscore']:.8g} | {row['normal_consistency']:.8g} | {row['normalized_flip_rate']:.3%} | {row['new_degenerate_faces']} | {row['same_index_recovered_vertex_rms']:.8g} | {row['total_inference_recovery_seconds']:.3f}s |"
        )
    lines.extend(("", "Unavailable domains are recorded explicitly in `ood_summary.json`.", ""))
    (output / "OOD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"implementation_audit": implementation_audit, "domains": domain_audits}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-name")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--arm-b-run", type=Path)
    parser.add_argument("--arm-e-run", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--view-chunk-size",
        type=int,
        help=(
            "Execution-only image-view encoder chunk size. All 28 sampled-view "
            "features are concatenated in order before the unchanged aggregation."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--hybrid-lambda",
        type=float,
        help="Validation-selected fixed direct-anchor hybrid regularization.",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--domain", action="append", default=[])
    args = parser.parse_args()
    if args.merge_only:
        if not args.domain:
            parser.error("--domain is required for merge")
        merge(args)
    else:
        if not args.domain_name or args.manifest is None or args.arm_b_run is None or args.arm_e_run is None:
            parser.error("evaluation requires domain name, manifest, and both frozen runs")
        if not 0 <= args.shard_index < args.shard_count:
            raise ValueError("Invalid shard index")
        evaluate_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
