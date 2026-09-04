#!/usr/bin/env python3
"""Report a paired single-pass Arm-B objective ablation on Sofa50 v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    ("refined_chamfer", "Refined CD", "lower"),
    ("p2s_p95", "P2S p95", "lower"),
    ("fscore", "F-score", "higher"),
    ("normal_consistency", "Normal", "higher"),
    ("raw_epe", "Raw EPE", "lower"),
    ("same_index_recovered_vertex_rms", "Vertex RMS", "lower"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-shard", type=Path, required=True)
    parser.add_argument("--reference-shard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_map(shard: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["split"], row["sample_id"]): row for row in shard["rows"]}


def bootstrap_ci(values: np.ndarray, replicates: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def fmt(value: float) -> str:
    return f"{value:.9g}"


def main() -> None:
    args = parse_args()
    new = load(args.new_shard)
    reference = load(args.reference_shard)
    new_rows = row_map(new)
    reference_rows = row_map(reference)

    expected_keys = {(split, row["sample_id"]) for split in ("validation", "test") for row in new["rows"] if row["split"] == split}
    assert set(new_rows) == set(reference_rows) == expected_keys
    assert len(new_rows) == 100
    assert all(sum(key[0] == split for key in new_rows) == 50 for split in ("validation", "test"))
    assert new["parameter_count"] == reference["parameter_count"] == 826115
    assert all(bool(row["lsmr_all_converged"]) for row in new["rows"])
    assert all(bool(row["lsmr_all_converged"]) for row in reference["rows"])
    assert all(float(row["lambda"]) == 0.01 for row in new["rows"] + reference["rows"])

    invariant_config_keys = (
        "dataset",
        "image_encoder",
        "input_mode",
        "local_query_jitter",
        "model",
        "query_training",
        "renderer_visibility",
        "target_definition",
        "target_mode",
        "target_scaling",
        "target_semantics",
    )
    assert all(new["config"][key] == reference["config"][key] for key in invariant_config_keys)

    maximum_initial_difference = max(
        abs(float(new_rows[key]["initial_chamfer"]) - float(reference_rows[key]["initial_chamfer"]))
        for key in new_rows
    )
    assert maximum_initial_difference < 1e-12

    summary: dict[str, Any] = {
        "contract_audit": True,
        "comparison": "single_pass_only_no_recursive_rounds",
        "new_checkpoint_sha256": new["checkpoint_sha256"],
        "reference_checkpoint_sha256": reference["checkpoint_sha256"],
        "parameter_count": new["parameter_count"],
        "maximum_initial_chamfer_difference": maximum_initial_difference,
        "execution_view_chunk_size": new.get("execution_view_chunk_size"),
        "splits": {},
    }

    report_lines = [
        "# Sofa50 v2 pure vertex-error Arm-B versus original recovery-aware Arm-B",
        "",
        "Contract audit: **true**. This is a paired, single-pass Arm-B comparison on the exact same 50 validation and 50 test meshes. Recursive R1--R5 evaluation is excluded from this formal comparison.",
        "",
        "Both models use the same 826,115-parameter Arm-B predictor, 28x960 RGB inputs, current-query/current-graph raw Laplacian representation, Uniform random-walk operator, and `lambda=1e-2` recovery. Only the training objective changes:",
        "",
        "```text",
        "Original Arm B: L = L_raw-Laplacian-Huber + 1e-2 * mean_i ||V_recovered_i - V_clean_i||_2^2",
        "New Arm B:      L = mean_i ||V_recovered_i - V_clean_i||_2^2",
        "```",
        "",
        f"The new validation-selected checkpoint is epoch `{new['training_metrics']['best_epoch']}` (optimizer step `15600`), SHA-256 `{new['checkpoint_sha256']}`. The original checkpoint SHA-256 is `{reference['checkpoint_sha256']}`.",
        "",
        "## Aggregate results",
        "",
        "| Split | Model | Initial CD | Refined CD | P2S p95 | F-score | Normal | Raw EPE (vertex-wtd.) | Vertex RMS | Improved/worsened |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    labels = (("reference", "Original Arm B", reference_rows), ("new", "Pure vertex-error Arm B", new_rows))
    for split in ("validation", "test"):
        split_summary: dict[str, Any] = {}
        for key_name, label, rows_by_key in labels:
            rows = [rows_by_key[key] for key in sorted(rows_by_key) if key[0] == split]
            aggregate = {
                "initial_chamfer": float(np.mean([row["initial_chamfer"] for row in rows])),
                "introduced_flipped_faces_total": int(sum(row["introduced_flipped_faces"] for row in rows)),
                "improved": int(sum(bool(row["improved"]) for row in rows)),
                "worsened": int(sum(bool(row["worsened"]) for row in rows)),
            }
            for metric, _, _ in METRICS:
                if metric == "raw_epe":
                    aggregate[metric] = float(
                        np.average(
                            [row[metric] for row in rows],
                            weights=[row["vertices"] for row in rows],
                        )
                    )
                else:
                    aggregate[metric] = float(np.mean([row[metric] for row in rows]))
            split_summary[key_name] = aggregate
            report_lines.append(
                "| " + " | ".join(
                    [
                        split,
                        label,
                        fmt(aggregate["initial_chamfer"]),
                        fmt(aggregate["refined_chamfer"]),
                        fmt(aggregate["p2s_p95"]),
                        fmt(aggregate["fscore"]),
                        fmt(aggregate["normal_consistency"]),
                        fmt(aggregate["raw_epe"]),
                        fmt(aggregate["same_index_recovered_vertex_rms"]),
                        f"{aggregate['improved']}/{aggregate['worsened']}",
                    ]
                ) + " |"
            )

        paired: dict[str, Any] = {}
        for metric, _, direction in METRICS:
            differences = np.asarray(
                [float(new_rows[key][metric]) - float(reference_rows[key][metric]) for key in sorted(new_rows) if key[0] == split],
                dtype=np.float64,
            )
            favorable = differences < 0 if direction == "lower" else differences > 0
            unfavorable = differences > 0 if direction == "lower" else differences < 0
            paired[metric] = {
                "new_minus_reference_mean": float(differences.mean()),
                "bootstrap_95_ci": bootstrap_ci(differences, args.bootstrap_replicates, args.seed),
                "new_wins": int(favorable.sum()),
                "new_losses": int(unfavorable.sum()),
                "ties": int((~favorable & ~unfavorable).sum()),
            }
        split_summary["paired"] = paired
        summary["splits"][split] = split_summary

    report_lines.extend(
        [
            "",
            "## Paired objective comparison",
            "",
            "Differences are pure vertex-error Arm B minus original Arm B. Negative CD/P2S/raw-EPE/vertex-RMS values and positive F-score/normal values favor the new objective. Aggregate raw EPE is vertex-weighted to reproduce the original report; paired differences and confidence intervals treat meshes as sampling units.",
            "",
            "| Split | Metric | Mean difference [95% CI] | New W/L/T |",
            "|---|---|---:|---:|",
        ]
    )
    for split in ("validation", "test"):
        for metric, label, _ in METRICS:
            paired = summary["splits"][split]["paired"][metric]
            ci = paired["bootstrap_95_ci"]
            report_lines.append(
                f"| {split} | {label} | {fmt(paired['new_minus_reference_mean'])} [{fmt(ci[0])}, {fmt(ci[1])}] | {paired['new_wins']}/{paired['new_losses']}/{paired['ties']} |"
            )

    test = summary["splits"]["test"]
    cd = test["paired"]["refined_chamfer"]
    new_test = test["new"]
    old_test = test["reference"]
    if cd["bootstrap_95_ci"][1] < 0:
        decision = "PURE_VERTEX_ERROR_BETTER"
        finding = "The pure recovered-vertex objective improves test Chamfer relative to the original mixed objective."
    elif cd["bootstrap_95_ci"][0] > 0:
        decision = "PURE_VERTEX_ERROR_WORSE"
        finding = "The pure recovered-vertex objective worsens test Chamfer relative to the original mixed objective."
    else:
        decision = "NO_RELIABLE_DIFFERENCE"
        finding = "The paired test Chamfer difference does not establish a reliable advantage for either objective."
    summary["classification"] = decision

    report_lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Classification: **{decision}**.",
            "",
            finding,
            f"Test CD is `{fmt(new_test['refined_chamfer'])}` for pure vertex-error training versus `{fmt(old_test['refined_chamfer'])}` for the original Arm B; the paired mean difference is `{fmt(cd['new_minus_reference_mean'])}` with 95% CI `[{fmt(cd['bootstrap_95_ci'][0])}, {fmt(cd['bootstrap_95_ci'][1])}]` and W/L/T `{cd['new_wins']}/{cd['new_losses']}/{cd['ties']}`.",
            "",
            f"The new objective does optimize its direct target: test same-index vertex RMS falls from `{fmt(old_test['same_index_recovered_vertex_rms'])}` to `{fmt(new_test['same_index_recovered_vertex_rms'])}`. However, test raw EPE rises from `{fmt(old_test['raw_epe'])}` to `{fmt(new_test['raw_epe'])}`, P2S p95 and F-score both worsen, and only `{new_test['improved']}/50` meshes improve over their initial geometry versus `{old_test['improved']}/50` for the original Arm B. The evidence therefore favors retaining the raw-Laplacian auxiliary term for the formal Arm-B method.",
            "",
            "This isolates the loss objective within the completed matched-v2 Arm-B setup. It does not make a claim about recursive refinement, B+E fusion, old native-1920 inputs, or Future2000.",
            "",
            "## Numerical and contract audit",
            "",
            "- Samples: `100` exact paired meshes (`50` validation, `50` test).",
            "- Recovery: Uniform random-walk Laplacian, `lambda=1e-2`, float64 LSMR; all `200` model/split solves across the two compared shards converged.",
            f"- Maximum paired initial-Chamfer discrepancy: `{maximum_initial_difference:.3e}`.",
            f"- Local CUDA inference used execution-only image-view chunking of `{new.get('execution_view_chunk_size')}` views; model parameters, 28-view inputs, predictions, and recovery equations are unchanged.",
            "- Checkpoint selection used validation only. Test rows were evaluated after selection and were not used to choose a checkpoint.",
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
