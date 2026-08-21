#!/usr/bin/env python3
"""Build a portable, auditable comparison index from an ExMesh result tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


METHOD_ORDER = (
    "exmesh_initial",
    "ours",
    "exmesh_official",
    "neural_deferred_shading",
    "nvdiffrec",
    "neuralangelo",
    "matcha",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene-id", default="24")
    return parser.parse_args()


def optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def percent_change(value: float | None, reference: float | None) -> float | None:
    if value is None or reference in (None, 0.0):
        return None
    return 100.0 * (value - reference) / reference


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_json_if_present(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    summary_path = root / "summary.csv"
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    selected = [row for row in rows if row["scene_id"] == str(args.scene_id)]
    selected.sort(key=lambda row: METHOD_ORDER.index(row["method"]))

    by_method = {row["method"]: row for row in selected}
    ours_status = load_json_if_present(root / "ours" / f"scan{args.scene_id}" / "status.json")
    initial_cd = optional_float(by_method.get("exmesh_initial", {}).get("chamfer"))
    ours_cd = optional_float(by_method.get("ours", {}).get("chamfer"))
    records: list[dict[str, object]] = []
    for row in selected:
        cd = optional_float(row.get("chamfer"))
        records.append(
            {
                "scene_id": int(row["scene_id"]),
                "method": row["method"],
                "success": row["success"].lower() == "true",
                "initialization": row["initialization"] or None,
                "num_views": optional_int(row.get("num_views")),
                "official_accuracy_d2s_mm": optional_float(row.get("official_accuracy_d2s")),
                "official_completeness_s2d_mm": optional_float(row.get("official_completeness_s2d")),
                "official_chamfer_overall_mm": cd,
                "delta_cd_vs_initial_mm": None if cd is None or initial_cd is None else cd - initial_cd,
                "relative_cd_vs_initial_percent": percent_change(cd, initial_cd),
                "delta_cd_vs_ours_mm": None if cd is None or ours_cd is None else cd - ours_cd,
                "vertices": optional_int(row.get("vertices")),
                "faces": optional_int(row.get("faces")),
                "runtime_sec": optional_float(row.get("runtime_sec")),
                "peak_gpu_memory_mib": optional_float(row.get("peak_gpu_memory")),
                "notes": row["notes"],
            }
        )

    ours = next(record for record in records if record["method"] == "ours")
    ours_eligible = bool(ours_status.get("primary_benchmark_eligible", ours["success"]))
    ours_audit = ours_status.get("contract_audit", bool(ours["success"]))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "official ExMesh DTU protocol",
        "scene_id": int(args.scene_id),
        "source_summary": str(summary_path),
        "ours_comparison_available": bool(ours["success"]),
        "ours_primary_benchmark_eligible": ours_eligible,
        "contract_audit": ours_audit,
        "warning": (
            "The frozen HF result is a Sofa50-to-ExMesh zero-shot domain-transfer diagnostic, "
            "not an ExMesh-only trained primary benchmark result."
            if ours["success"] and not ours_eligible
            else None if ours["success"] else
            "No same-protocol learned-Laplacian output exists for this ExMesh scene."
        ),
        "records": records,
    }
    (root.parent / "comparison_scan24.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    csv_path = root.parent / "comparison_scan24.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# ExMesh protocol comparison export",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "This export uses the official ExMesh DTU observations, cameras, normalization, "
        "initialization, coordinate frame, and evaluator. Standalone Sofa50/OpenMVS metric "
        "rows are excluded; `ours` is an ExMesh inference result from a frozen Sofa50-trained checkpoint.",
        "",
        "## Scan24 result table",
        "",
        "| Method | State | CD / overall (mm) | D2S (mm) | S2D (mm) | vs initial | Vertices | Faces | Runtime (s) | Peak GPU (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        relative = record["relative_cd_vs_initial_percent"]
        relative_text = "—" if relative is None else f"{relative:+.2f}%"
        lines.append(
            "| {method} | {state} | {cd} | {d2s} | {s2d} | {relative} | {vertices} | {faces} | {runtime} | {memory} |".format(
                method=record["method"],
                state="success" if record["success"] else "not run",
                cd=fmt(record["official_chamfer_overall_mm"]),
                d2s=fmt(record["official_accuracy_d2s_mm"]),
                s2d=fmt(record["official_completeness_s2d_mm"]),
                relative=relative_text,
                vertices=fmt(record["vertices"], 0),
                faces=fmt(record["faces"], 0),
                runtime=fmt(record["runtime_sec"], 1),
                memory=fmt(record["peak_gpu_memory_mib"], 0),
            )
        )
    interpretation = [
            "",
            "## Interpretation",
            "",
            "- Official ExMesh improves scan24 CD from 0.528230 mm to 0.443233 mm "
            "(16.09% lower than its initial mesh).",
            "- The official NDS sanity run reaches 6.973501 mm and is worse than the shared "
            "ExMesh initial mesh on this scene; it uses NDS's official visual-hull initialization, "
            "so this is an end-to-end reconstruction comparison, not a same-initialization ablation.",
    ]
    if ours["success"]:
        interpretation.append(
            "- `ours` is the frozen latest HF model applied to the official ExMesh input, "
            "initial mesh, cameras, coordinate frame, and evaluator without retraining or tuning."
        )
        if not ours_eligible:
            interpretation.append(
                "- This `ours` row is explicitly exploratory: the checkpoint was trained on "
                "Sofa50 synthetic-current data, so it is a zero-shot domain-transfer diagnostic "
                "and is not eligible for the strict ExMesh-only primary benchmark claim."
            )
    else:
        interpretation.extend(
            [
                "- A numerical comparison against `ours` is not currently available because no "
                "successful `ours/scan24/status.json` or same-protocol final mesh exists.",
                "- The old Sofa50 learned outputs and `runs/refined/refined.obj` are not imported: "
                "their data/camera/normalization contracts differ.",
            ]
        )
    if ours["success"]:
        primary_variant = load_json_if_present(
            root / "ours" / f"scan{args.scene_id}" / "variant_status.json"
        )
        sensitivity = ours_status.get("sensitivity_views28", {})
        runtime_environment = ours_status.get("runtime_environment", {})
        interpretation.extend(
            [
                "",
                "## HF prediction-to-optimization diagnostic",
                "",
                "| HF inference arm | Views | CD / overall (mm) | D2S (mm) | S2D (mm) | Flipped faces | Mean confidence | Runtime (s) | Peak GPU (MiB) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                "| all49 (primary) | {views} | {cd} | {d2s} | {s2d} | {flips} | {confidence} | {runtime} | {memory} |".format(
                    views=primary_variant.get("num_views", ours_status.get("num_views")),
                    cd=fmt(ours_status.get("metrics", {}).get("overall")),
                    d2s=fmt(ours_status.get("metrics", {}).get("mean_d2s")),
                    s2d=fmt(ours_status.get("metrics", {}).get("mean_s2d")),
                    flips=fmt(primary_variant.get("introduced_flipped_faces"), 0),
                    confidence=fmt(primary_variant.get("mean_confidence")),
                    runtime=fmt(primary_variant.get("runtime_sec"), 1),
                    memory=fmt(primary_variant.get("peak_gpu_memory_mib"), 0),
                ),
                "| uniform28 (sensitivity) | {views} | {cd} | {d2s} | {s2d} | {flips} | {confidence} | {runtime} | {memory} |".format(
                    views=sensitivity.get("num_views"),
                    cd=fmt(sensitivity.get("metrics", {}).get("overall")),
                    d2s=fmt(sensitivity.get("metrics", {}).get("mean_d2s")),
                    s2d=fmt(sensitivity.get("metrics", {}).get("mean_s2d")),
                    flips=fmt(sensitivity.get("introduced_flipped_faces"), 0),
                    confidence=fmt(sensitivity.get("mean_confidence")),
                    runtime=fmt(sensitivity.get("runtime_sec"), 1),
                    memory=fmt(sensitivity.get("peak_gpu_memory_mib"), 0),
                ),
                "",
                "- The 49-view and 28-view outcomes are nearly identical, so the degradation is "
                "not explained by using more views at inference than during training.",
                "- ExMesh provides surface-evaluation GT rather than a vertex-correspondent raw-"
                "Laplacian target for the shared initial graph. Raw prediction arrays are exported, "
                "but Raw EPE is deliberately not fabricated; D2S/S2D/CD measure the recovered geometry.",
                "- Runtime environment: {gpu}; PyTorch {torch}; CUDA {cuda}.".format(
                    gpu=runtime_environment.get("gpu_name", "unknown"),
                    torch=runtime_environment.get("torch_version", "unknown"),
                    cuda=runtime_environment.get("cuda_runtime", "unknown"),
                ),
                "",
                "![Fixed-camera mesh comparison](exmesh_baselines/ours/scan24/visualizations/scan24_fixed_camera_panel.png)",
                "",
                "Detailed audit: `exmesh_baselines/ours/scan24/ZERO_SHOT_REPORT.md`.",
            ]
        )
    lines.extend(
        interpretation
        + [
            "",
            "## Export contents",
            "",
            "- `exmesh_baselines/`: complete current HPC result tree, including all released "
            "intermediate/final meshes, metrics, configs, logs, and failure state.",
            "- `comparison_scan24.csv` and `comparison_scan24.json`: normalized scan24 table.",
            "- `FILE_MANIFEST.csv`: portable file inventory with SHA-256 checksums.",
            "",
            "This is a frozen snapshot. Newly completed HPC jobs must be synchronized into a "
            "new snapshot or explicitly merged; they are not represented here automatically.",
        ]
    )
    (root.parent / "COMPARISON_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_path = root.parent / "FILE_MANIFEST.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "size_bytes", "sha256"))
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            writer.writerow((path.relative_to(root.parent), path.stat().st_size, digest.hexdigest()))


if __name__ == "__main__":
    main()
