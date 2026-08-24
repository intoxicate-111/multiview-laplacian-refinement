#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate direct-vertex Arm E and merge it with the frozen Sofa50 A-D study."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from evaluate_sofa50_recovery_aware_ablation import EXTENSION_ARMS
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.controlled_displacement import (
    DIRECT_VERTEX_DISPLACEMENT,
    displacement_target,
    recover_direct_displacement,
)
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint


ARM_E = "E_direct_vertex_residual"
GEOMETRY_FIELDS = (
    "initial_chamfer",
    "refined_chamfer",
    "relative_chamfer_gain",
    "eta",
    "p2s",
    "p2s_p95",
    "fscore",
    "normal_consistency",
    "same_index_recovered_vertex_rms",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _checkpoint(run: Path) -> Path:
    for name in ("checkpoint_best.pt", "best.pt"):
        path = run / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No validation-selected checkpoint in {run}")


def _training_stability(run: Path) -> dict[str, Any]:
    history = _read_list(run / "training_step_history.json")
    rows = [row for row in history if row.get("prediction_displacement_rms") is not None]
    if not rows:
        raise RuntimeError("Arm E did not record direct-vertex runtime diagnostics")
    return {
        "logged_intervals": len(rows),
        "latest_optimizer_step": int(rows[-1]["optimizer_steps"]),
        "latest_vertex_loss": float(rows[-1]["train_loss"]),
        "prediction_displacement_rms_mean": float(np.mean([float(row["prediction_displacement_rms"]) for row in rows])),
        "prediction_displacement_rms_max": float(max(float(row["prediction_displacement_rms"]) for row in rows)),
        "prediction_displacement_mean": float(np.mean([float(row["prediction_displacement_mean"]) for row in rows])),
        "delta_v_gradient_norm": float(np.mean([float(row["delta_v_gradient_norm"]) for row in rows])),
        "image_encoder_gradient_norm": float(np.mean([float(row["image_encoder_gradient_norm"]) for row in rows])),
        "graph_block_gradient_norm": float(np.mean([float(row["graph_block_gradient_norm"]) for row in rows])),
        "prediction_head_gradient_norm": float(np.mean([float(row["prediction_head_gradient_norm"]) for row in rows])),
        "nan_inf_count": int(sum(int(row["nan_inf_count"]) for row in rows)),
        "seconds_per_step": float(np.mean([float(row["interval_seconds"]) / int(row["optimizer_steps_in_interval"]) for row in rows])),
        "peak_gpu_memory_mb": float(max(float(row["peak_gpu_memory_mb"]) for row in rows)),
    }


def _read_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(path)
    return value


def evaluate(args: argparse.Namespace) -> None:
    run = args.run.resolve()
    output = args.output_dir.resolve()
    config_payload = _read(run / "run_config.json")
    config = config_payload.get("experiment_config", config_payload)
    metrics = _read(run / "metrics.json")
    checkpoint = _checkpoint(run)
    device = torch.device(args.device)
    model = _build_model(config, None, False).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)

    if config.get("prediction_semantics") != DIRECT_VERTEX_DISPLACEMENT:
        raise RuntimeError("Arm E checkpoint is not direct_vertex_displacement")
    if config["training"]["loss"] != "mse":
        raise RuntimeError("Arm E does not use direct vertex MSE")
    recovery = config["recovery"]
    forbidden_active = any(
        bool(recovery.get(key))
        for key in ("laplacian_operator_used", "sparse_integration", "pcg_or_lsmr", "visibility_gate", "confidence_weighting", "optimizer", "postprocessing")
    )
    if forbidden_active or recovery.get("lambda") is not None:
        raise RuntimeError("Arm E recovery path contains a forbidden operation")

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    split_ids: dict[str, list[str]] = {}
    for split in ("validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), split)
        split_ids[split] = list(dataset.sample_ids)
        predicted_arrays: list[np.ndarray] = []
        target_arrays: list[np.ndarray] = []
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            prepared = _load_device_item(dataset, index, config, device)
            conditioned = _exact_query_sample(prepared.sample, device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                output_value = model(conditioned)
            if output_value.confidence_prediction is not None:
                raise RuntimeError("Arm E unexpectedly instantiated confidence")
            prediction = output_value.predicted_laplacian.float().detach().cpu().numpy().astype(np.float64)
            target = displacement_target(static).cpu().numpy().astype(np.float64)
            initial = Mesh(
                torch.as_tensor(static["vertices"]).cpu().numpy().astype(np.float64),
                torch.as_tensor(static["faces"]).cpu().numpy().astype(np.int64),
            ).ensure_normals()
            clean = _clean_mesh(static)
            recovered = np.asarray(
                recover_direct_displacement(initial.vertices, prediction), dtype=np.float64
            )
            refined = Mesh(recovered, initial.faces.copy()).ensure_normals()
            sample_id = str(static["sample_id"])
            initial_geometry = _geometry_row("v2_strong_smoothing", sample_id, "initial", initial, clean, initial)
            clean_geometry = _geometry_row("v2_strong_smoothing", sample_id, "clean", clean, clean, initial)
            refined_geometry = _geometry_row("v2_strong_smoothing", sample_id, ARM_E, refined, clean, initial)
            initial_cd = float(initial_geometry["chamfer"])
            clean_cd = float(clean_geometry["chamfer"])
            refined_cd = float(refined_geometry["chamfer"])
            displacement_error = prediction - target
            rows.append(
                {
                    "arm": ARM_E,
                    "split": split,
                    "sample_id": sample_id,
                    "initial_chamfer": initial_cd,
                    "refined_chamfer": refined_cd,
                    "relative_chamfer_gain": (initial_cd - refined_cd) / initial_cd,
                    "eta": (initial_cd - refined_cd) / (initial_cd - clean_cd),
                    "p2s": float(refined_geometry["p2s"]),
                    "p2s_p95": float(refined_geometry["p2s_p95"]),
                    "fscore": float(refined_geometry["fscore"]),
                    "normal_consistency": float(refined_geometry["normal_consistency"]),
                    "introduced_flipped_faces": int(refined_geometry["introduced_flipped_faces"]),
                    "new_degenerate_faces": int(refined_geometry["new_degenerate_faces"]),
                    "same_index_recovered_vertex_rms": float(np.sqrt(np.mean(np.sum((recovered - clean.vertices) ** 2, axis=1)))),
                    "displacement_error_rms": float(np.sqrt(np.mean(np.sum(displacement_error ** 2, axis=1)))),
                    "mean_predicted_displacement_magnitude": float(np.mean(np.linalg.norm(prediction, axis=1))),
                    "rms_predicted_displacement_magnitude": float(np.sqrt(np.mean(np.sum(prediction ** 2, axis=1)))),
                    "improved": refined_cd < initial_cd,
                    "worsened": refined_cd > initial_cd,
                    "vertices": initial.num_vertices,
                    "faces": initial.num_faces,
                }
            )
            predicted_arrays.append(prediction)
            target_arrays.append(target)
            print(f"{ARM_E} {split} {index + 1}/{len(dataset)} {sample_id}", flush=True)
            torch.cuda.empty_cache()
        arrays[f"{split}_prediction"] = np.concatenate(predicted_arrays, axis=0)
        arrays[f"{split}_target"] = np.concatenate(target_arrays, axis=0)

    shard = output / "shards"
    _write_json(
        shard / f"{ARM_E}.json",
        {
            "arm": ARM_E,
            "run": str(run),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "config": config,
            "training_metrics": metrics,
            "training_stability": _training_stability(run),
            "split_ids": split_ids,
            "rows": rows,
        },
    )
    np.savez_compressed(shard / f"{ARM_E}_prediction_arrays.npz", **arrays)


def _aggregate(rows: Sequence[Mapping[str, Any]], arm: str, split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm and row["split"] == split]
    result = {"arm": arm, "split": split, "samples": len(selected)}
    for field in GEOMETRY_FIELDS:
        result[field] = _mean(selected, field)
    result.update(
        {
            "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in selected)),
            "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in selected)),
            "improved": int(sum(bool(row["improved"]) for row in selected)),
            "worsened": int(sum(bool(row["worsened"]) for row in selected)),
        }
    )
    if arm == ARM_E:
        result["displacement_error_rms"] = _mean(selected, "displacement_error_rms")
        result["mean_predicted_displacement_magnitude"] = _mean(selected, "mean_predicted_displacement_magnitude")
    return result


def _paired(left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    left = {str(row["sample_id"]): row for row in left_rows}
    right = {str(row["sample_id"]): row for row in right_rows}
    if left.keys() != right.keys() or len(left) != 50:
        raise RuntimeError("paired test sample IDs are not the exact same 50 meshes")
    pairs = [(left[key], right[key]) for key in sorted(left)]
    return {
        "samples": len(pairs),
        "right_lower_chamfer": sum(r["refined_chamfer"] < l["refined_chamfer"] for l, r in pairs),
        "right_lower_vertex_rms": sum(r["same_index_recovered_vertex_rms"] < l["same_index_recovered_vertex_rms"] for l, r in pairs),
        "right_lower_p2s_p95": sum(r["p2s_p95"] < l["p2s_p95"] for l, r in pairs),
        "right_higher_fscore": sum(r["fscore"] > l["fscore"] for l, r in pairs),
        "right_higher_normal": sum(r["normal_consistency"] > l["normal_consistency"] for l, r in pairs),
        "right_fewer_flips": sum(r["introduced_flipped_faces"] < l["introduced_flipped_faces"] for l, r in pairs),
    }


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    ad_dir = args.ad_report_dir.resolve()
    payloads = [_read(ad_dir / "shards" / f"{arm}.json") for arm in EXTENSION_ARMS]
    e_payload = _read(output / "shards" / f"{ARM_E}.json")
    all_payloads = payloads + [e_payload]
    all_rows = [row for payload in all_payloads for row in payload["rows"]]
    arms = EXTENSION_ARMS + (ARM_E,)
    aggregate = [_aggregate(all_rows, arm, split) for arm in arms for split in ("validation", "test")]
    validation = {row["arm"]: row for row in aggregate if row["split"] == "validation"}
    test = {row["arm"]: row for row in aggregate if row["split"] == "test"}
    best_laplacian = min(EXTENSION_ARMS[1:], key=lambda arm: validation[arm]["refined_chamfer"])
    test_rows = {arm: [row for row in all_rows if row["arm"] == arm and row["split"] == "test"] for arm in arms}
    paired = {
        "B_vs_E": _paired(test_rows[EXTENSION_ARMS[1]], test_rows[ARM_E]),
        "best_laplacian_vs_E": _paired(test_rows[best_laplacian], test_rows[ARM_E]),
    }
    configs = [payload["config"] for payload in all_payloads]
    parameter_counts = [int(payload["parameter_count"]) for payload in all_payloads]
    e_config = e_payload["config"]
    e_metrics = e_payload["training_metrics"]
    same_ids = all(
        [row["sample_id"] for row in payload["rows"] if row["split"] == split]
        == [row["sample_id"] for row in e_payload["rows"] if row["split"] == split]
        for payload in payloads
        for split in ("validation", "test")
    )
    implementation_audit = bool(
        same_ids
        and len(set(parameter_counts)) == 1
        and e_config["prediction_semantics"] == DIRECT_VERTEX_DISPLACEMENT
        and e_config["training"]["loss"] == "mse"
        and not e_config["confidence"]["enabled"]
        and int(e_metrics["optimizer_steps"]) == 20000
        and int(e_metrics["global_batch_meshes"]) == 8
        and int(e_metrics["distributed_world_size"]) == 8
        and e_payload["training_stability"]["nan_inf_count"] == 0
    )
    best_pair = paired["best_laplacian_vs_E"]
    relative_gap = (test[ARM_E]["refined_chamfer"] - test[best_laplacian]["refined_chamfer"]) / test[best_laplacian]["refined_chamfer"]
    if abs(relative_gap) <= 0.02 and 20 <= best_pair["right_lower_chamfer"] <= 30:
        conclusion = "H1_supported: direct vertex supervision performs approximately as well as the validation-selected Laplacian arm."
    elif relative_gap > 0 and best_pair["right_lower_chamfer"] < 20:
        conclusion = "H2_supported: the validation-selected recovery-aware Laplacian arm materially outperforms direct vertex regression."
    elif relative_gap < 0 and best_pair["right_lower_chamfer"] > 30:
        conclusion = "H1_supported_or_stronger: direct vertex regression outperforms the validation-selected Laplacian arm."
    else:
        conclusion = "inconclusive_between_H1_and_H2: aggregate and paired evidence are mixed."
    summary = {
        "implementation_audit": implementation_audit,
        "metric_protocol": METRIC_PROTOCOL,
        "arm_e_forward": "delta_v_pred=f_theta(I,C,V_input,F); V_refined=V_input+delta_v_pred",
        "arm_e_loss": "mean_i ||delta_v_pred_i-(V_clean_i-V_input_i)||_2^2",
        "parameter_counts": dict(zip(arms, parameter_counts, strict=True)),
        "best_validation_selected_laplacian_arm": best_laplacian,
        "aggregate": aggregate,
        "paired_test": paired,
        "training_stability": e_payload["training_stability"],
        "conclusion": conclusion,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "implementation_audit.json", {"passed": implementation_audit, "same_sample_ids": same_ids, "parameter_counts": parameter_counts})
    _write_csv(output / "per_sample.csv", all_rows)
    _write_csv(output / "aggregate.csv", aggregate)
    _write_json(output / "paired_test.json", paired)

    lines = [
        "# Sofa50 v2 A/B/C/D/E representation ablation",
        "",
        f"Implementation audit: **{str(implementation_audit).lower()}**.",
        "",
        "Arm E forward: `delta_v_pred=f_theta(I,C,V_input,F); V_refined=V_input+delta_v_pred`.",
        "",
        "Arm E loss: `mean_i ||delta_v_pred_i-(V_clean_i-V_input_i)||_2^2`.",
        "",
        f"All model parameter counts: `{parameter_counts}`.",
        "",
        "## Validation/test geometry",
        "",
        "| Split | Arm | Initial CD | Refined CD | Gain | Eta | P2S | P2S p95 | F-score | Normal | Flips | New deg. | Improved/worsened | Vertex RMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['split']} | {row['arm']} | {row['initial_chamfer']:.9g} | {row['refined_chamfer']:.9g} | {row['relative_chamfer_gain']:.2%} | {row['eta']:.9g} | {row['p2s']:.9g} | {row['p2s_p95']:.9g} | {row['fscore']:.9g} | {row['normal_consistency']:.9g} | {row['introduced_flipped_faces']} | {row['new_degenerate_faces']} | {row['improved']}/{row['worsened']} | {row['same_index_recovered_vertex_rms']:.9g} |"
        )
    lines.extend((
        "",
        "## Arm E displacement metrics",
        "",
        "| Split | RMS(delta_v_pred-delta_v_gt) | Mean predicted displacement |",
        "|---|---:|---:|",
    ))
    for split in ("validation", "test"):
        row = next(item for item in aggregate if item["arm"] == ARM_E and item["split"] == split)
        lines.append(f"| {split} | {row['displacement_error_rms']:.9g} | {row['mean_predicted_displacement_magnitude']:.9g} |")
    lines.extend((
        "",
        "## Paired test comparisons (right side is Arm E)",
        "",
        "| Comparison | E lower CD | E lower vertex RMS | E lower P2S p95 | E higher F | E higher normal | E fewer flips |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ))
    for name, values in paired.items():
        lines.append(f"| {name} | {values['right_lower_chamfer']}/50 | {values['right_lower_vertex_rms']}/50 | {values['right_lower_p2s_p95']}/50 | {values['right_higher_fscore']}/50 | {values['right_higher_normal']}/50 | {values['right_fewer_flips']}/50 |")
    stability = e_payload["training_stability"]
    lines.extend((
        "",
        "## Arm E training stability",
        "",
        f"Logged `{stability['logged_intervals']}` intervals through step `{stability['latest_optimizer_step']}`; NaN/Inf count `{stability['nan_inf_count']}`; mean/max predicted displacement RMS `{stability['prediction_displacement_rms_mean']:.9g}` / `{stability['prediction_displacement_rms_max']:.9g}`; mean step time `{stability['seconds_per_step']:.4f}s`; peak GPU memory `{stability['peak_gpu_memory_mb']:.1f} MiB`.",
        "",
        "## H1 vs H2 conclusion",
        "",
        f"Validation-selected Laplacian arm: **{best_laplacian}**.",
        "",
        conclusion,
        "",
        "Matched visualizations are generated by the dependent visualization job and indexed in `visuals/comparison_manifest.json`.",
        "",
    ))
    (output / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--ad-report-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        if args.ad_report_dir is None:
            parser.error("--ad-report-dir is required with --merge-only")
        merge(args)
    else:
        if args.manifest is None or args.run is None:
            parser.error("--manifest and --run are required for evaluation")
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
