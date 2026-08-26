#!/usr/bin/env python3
from __future__ import annotations

"""Create the validation-only old-domain specialist/frozen fusion report."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: list[dict[str, str]], name: str) -> dict[str, Any]:
    faces = sum(int(row["faces"]) for row in rows)
    initial = float(np.mean([float(row["initial_chamfer"]) for row in rows]))
    refined = float(np.mean([float(row["refined_chamfer"]) for row in rows]))
    return {
        "method": name,
        "samples": len(rows),
        "initial_chamfer": initial,
        "refined_chamfer": refined,
        "aggregate_relative_gain": (initial - refined) / initial,
        "same_index_recovered_vertex_rms": float(
            np.mean([float(row["same_index_recovered_vertex_rms"]) for row in rows])
        ),
        "p2s": float(np.mean([float(row["p2s"]) for row in rows])),
        "p2s_p95": float(np.mean([float(row["p2s_p95"]) for row in rows])),
        "fscore": float(np.mean([float(row["fscore"]) for row in rows])),
        "normal_consistency": float(
            np.mean([float(row["normal_consistency"]) for row in rows])
        ),
        "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in rows)),
        "normalized_flip_rate": sum(int(row["introduced_flipped_faces"]) for row in rows) / faces,
        "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in rows)),
        "improved": int(sum(row["improved"].lower() == "true" for row in rows)),
        "worsened": int(sum(row["worsened"].lower() == "true" for row in rows)),
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specialist-summary", required=True, type=Path)
    parser.add_argument("--specialist-per-sample", required=True, type=Path)
    parser.add_argument("--lambda-selection", required=True, type=Path)
    parser.add_argument("--lambda-per-sample", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    specialist_summary = read_json(args.specialist_summary)
    selection = read_json(args.lambda_selection)
    if specialist_summary.get("split") != "validation" or specialist_summary.get("test_opened") is not False:
        raise RuntimeError("Specialist report is not validation-only")
    if selection.get("selection_split") != "validation" or selection.get("test_accessed") is not False:
        raise RuntimeError("Lambda selection is not validation-only")
    selected_lambda = float(selection["selected_lambda"])
    specialist_rows = read_csv(args.specialist_per_sample)
    b_rows = [row for row in specialist_rows if row["arm"] == "B_recovery_aware"]
    e_rows = [row for row in specialist_rows if row["arm"] == "E_direct_vertex"]
    frozen_rows = [
        row for row in read_csv(args.lambda_per_sample) if float(row["lambda"]) == selected_lambda
    ]
    for name, rows in (("B", b_rows), ("E", e_rows), ("Frozen", frozen_rows)):
        if len(rows) != 25:
            raise RuntimeError(f"Expected 25 validation {name} rows, found {len(rows)}")
    ids = [row["sample_id"] for row in frozen_rows]
    b_by_id = {row["sample_id"]: row for row in b_rows}
    e_by_id = {row["sample_id"]: row for row in e_rows}
    if set(ids) != set(b_by_id) or set(ids) != set(e_by_id):
        raise RuntimeError("Validation sample identities differ across B/E/Frozen")
    frozen_aggregate = aggregate(frozen_rows, "Frozen B+E")
    aggregates = [
        aggregate(b_rows, "Arm B"),
        aggregate(e_rows, "Arm E"),
        frozen_aggregate,
    ]
    pairwise = {}
    for name, comparison in (("B", b_by_id), ("E", e_by_id)):
        differences = np.asarray(
            [
                float(next(row for row in frozen_rows if row["sample_id"] == sample_id)["refined_chamfer"])
                - float(comparison[sample_id]["refined_chamfer"])
                for sample_id in ids
            ],
            dtype=np.float64,
        )
        pairwise[f"Frozen_vs_{name}"] = {
            "frozen_wins": int(np.sum(differences < 0)),
            "frozen_losses": int(np.sum(differences > 0)),
            "ties": int(np.sum(differences == 0)),
            "mean_cd_difference_frozen_minus_specialist": float(np.mean(differences)),
            "median_cd_difference_frozen_minus_specialist": float(np.median(differences)),
        }
    contract = bool(
        specialist_summary.get("contract_audit")
        and selection.get("contract_audit")
        and specialist_summary["arm_b_checkpoint_sha256"] == selection["arm_b_checkpoint_sha256"]
        and specialist_summary["arm_e_checkpoint_sha256"] == selection["arm_e_checkpoint_sha256"]
        and all(row["pcg_converged"].lower() == "true" for row in frozen_rows)
        and max(float(row["pcg_relative_residual"]) for row in frozen_rows) <= 1e-8
    )
    payload = {
        "contract_audit": contract,
        "split": "validation",
        "test_accessed": False,
        "selected_lambda": selected_lambda,
        "selected_at_grid_boundary": bool(selection["selected_at_grid_boundary"]),
        "arm_b_checkpoint_sha256": specialist_summary["arm_b_checkpoint_sha256"],
        "arm_e_checkpoint_sha256": specialist_summary["arm_e_checkpoint_sha256"],
        "aggregate": aggregates,
        "paired": pairwise,
        "solver": selection["solver"],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_validation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Old-domain native-1920 frozen B+E validation",
        "",
        f"Contract audit: **{str(contract).lower()}**. Test accessed: **false**.",
        "",
        f"Validation-selected `lambda_old`: **{fmt(selected_lambda)}**"
        + (" (predeclared-grid boundary)." if selection["selected_at_grid_boundary"] else "."),
        "",
        "| Method | CD | Gain | VRMS | P2S mean | P2S p95 | F-score | Normal | Flips / rate | New deg. | Improved/worsened |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['method']} | {fmt(row['refined_chamfer'])} | "
            f"{100 * row['aggregate_relative_gain']:+.2f}% | "
            f"{fmt(row['same_index_recovered_vertex_rms'])} | {fmt(row['p2s'])} | "
            f"{fmt(row['p2s_p95'])} | {fmt(row['fscore'])} | "
            f"{fmt(row['normal_consistency'])} | {row['introduced_flipped_faces']} / "
            f"{100 * row['normalized_flip_rate']:.3f}% | {row['new_degenerate_faces']} | "
            f"{row['improved']}/{row['worsened']} |"
        )
    lines.extend(
        [
            "",
            "| Comparison | Frozen wins/losses/ties | Mean CD difference | Median CD difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in pairwise.items():
        lines.append(
            f"| {name.replace('_', ' ')} | {row['frozen_wins']}/{row['frozen_losses']}/{row['ties']} | "
            f"{fmt(row['mean_cd_difference_frozen_minus_specialist'])} | "
            f"{fmt(row['median_cd_difference_frozen_minus_specialist'])} |"
        )
    lines.extend(
        [
            "",
            "Both specialist checkpoints and `lambda_old` were selected without test access. "
            "All frozen validation solves used float64 PCG at tolerance `1e-8` and maximum 2048 iterations.",
            "",
        ]
    )
    (output / "FROZEN_VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    if not contract:
        raise RuntimeError("Frozen validation report contract failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
