#!/usr/bin/env python3
from __future__ import annotations

"""Diagnose whether Ours initial-to-refined face flips are GT-corrective or harmful."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.io import load_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


PRIMARY_DEGENERATE_CROSS_NORM = 1e-14
ALIGNMENT_EPSILON = 1e-6
NEAR_ZERO_RELATIVE_THRESHOLDS = (1e-12, 1e-10, 1e-8)


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _face_geometry(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    area_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    cross_norms = np.linalg.norm(area_vectors, axis=1)
    normals = area_vectors / np.maximum(cross_norms[:, None], 1e-300)
    centroids = triangles.mean(axis=1)
    return area_vectors, cross_norms, normals, centroids


def _matched_gt_face_normals(
    gt_vertices: np.ndarray,
    gt_faces: np.ndarray,
    query_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    gt_vertices = np.asarray(gt_vertices, dtype=np.float64)
    gt_faces = np.asarray(gt_faces, dtype=np.int64)
    triangles = gt_vertices[gt_faces]
    area_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    norms = np.linalg.norm(area_vectors, axis=1)
    if np.any(norms <= PRIMARY_DEGENERATE_CROSS_NORM):
        raise ValueError("GT contains a face degenerate under the primary 1e-14 cross-norm threshold")
    gt_normals = area_vectors / norms[:, None]
    surface = trimesh.Trimesh(vertices=gt_vertices, faces=gt_faces, process=False)
    _, _, face_indices = trimesh.proximity.closest_point(
        surface, np.asarray(query_points, dtype=np.float64)
    )
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if np.any(face_indices < 0) or np.any(face_indices >= len(gt_faces)):
        raise RuntimeError("Exact GT closest-surface query returned invalid face indices")
    return gt_normals[face_indices], face_indices


def _change_counts(initial: np.ndarray, refined: np.ndarray, epsilon: float) -> dict[str, int]:
    delta = np.asarray(refined) - np.asarray(initial)
    return {
        "improved": int(np.count_nonzero(delta > epsilon)),
        "worsened": int(np.count_nonzero(delta < -epsilon)),
        "unchanged_or_ambiguous": int(np.count_nonzero(np.abs(delta) <= epsilon)),
    }


def _orientation_categories(
    initial: np.ndarray, refined: np.ndarray, epsilon: float
) -> dict[str, int]:
    initial = np.asarray(initial)
    refined = np.asarray(refined)
    initial_wrong = initial < -epsilon
    initial_correct = initial > epsilon
    refined_wrong = refined < -epsilon
    refined_correct = refined > epsilon
    wrong_to_correct = initial_wrong & refined_correct
    correct_to_wrong = initial_correct & refined_wrong
    wrong_to_wrong = initial_wrong & refined_wrong
    remainder = ~(wrong_to_correct | correct_to_wrong | wrong_to_wrong)
    return {
        "wrong_to_correct": int(np.count_nonzero(wrong_to_correct)),
        "correct_to_wrong": int(np.count_nonzero(correct_to_wrong)),
        "wrong_to_wrong": int(np.count_nonzero(wrong_to_wrong)),
        "correct_to_correct_or_ambiguous": int(np.count_nonzero(remainder)),
    }


def analyze_sample(
    sample_id: str,
    initial_vertices: np.ndarray,
    faces: np.ndarray,
    refined_vertices: np.ndarray,
    refined_faces: np.ndarray,
    gt_vertices: np.ndarray,
    gt_faces: np.ndarray,
) -> dict[str, Any]:
    initial_vertices = np.asarray(initial_vertices, dtype=np.float64)
    refined_vertices = np.asarray(refined_vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    refined_faces = np.asarray(refined_faces, dtype=np.int64)
    if initial_vertices.shape != refined_vertices.shape or not np.array_equal(faces, refined_faces):
        raise ValueError(f"{sample_id}: refined output does not preserve initial vertex/face indexing")

    initial_cross, initial_cross_norm, initial_normals, initial_centroids = _face_geometry(
        initial_vertices, faces
    )
    refined_cross, refined_cross_norm, refined_normals, refined_centroids = _face_geometry(
        refined_vertices, faces
    )
    primary_flip = np.einsum("ij,ij->i", initial_cross, refined_cross) < 0.0
    initial_degenerate = initial_cross_norm <= PRIMARY_DEGENERATE_CROSS_NORM
    refined_degenerate = refined_cross_norm <= PRIMARY_DEGENERATE_CROSS_NORM

    queries = np.concatenate((initial_centroids, refined_centroids), axis=0)
    matched_normals, matched_faces = _matched_gt_face_normals(gt_vertices, gt_faces, queries)
    count = len(faces)
    initial_gt_normals = matched_normals[:count]
    refined_gt_normals = matched_normals[count:]
    initial_gt_faces = matched_faces[:count]
    refined_gt_faces = matched_faces[count:]
    initial_signed = np.clip(
        np.einsum("ij,ij->i", initial_normals, initial_gt_normals), -1.0, 1.0
    )
    refined_signed = np.clip(
        np.einsum("ij,ij->i", refined_normals, refined_gt_normals), -1.0, 1.0
    )
    initial_signed[initial_degenerate] = 0.0
    refined_signed[refined_degenerate] = 0.0
    initial_abs = np.abs(initial_signed)
    refined_abs = np.abs(refined_signed)

    flip_initial_signed = initial_signed[primary_flip]
    flip_refined_signed = refined_signed[primary_flip]
    flip_initial_abs = initial_abs[primary_flip]
    flip_refined_abs = refined_abs[primary_flip]
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "face_count": count,
        "primary_flipped_faces": int(np.count_nonzero(primary_flip)),
        "all_signed_initial_sum": float(initial_signed.sum()),
        "all_signed_refined_sum": float(refined_signed.sum()),
        "all_abs_initial_sum": float(initial_abs.sum()),
        "all_abs_refined_sum": float(refined_abs.sum()),
        "flipped_signed_initial_sum": float(flip_initial_signed.sum()),
        "flipped_signed_refined_sum": float(flip_refined_signed.sum()),
        "flipped_abs_initial_sum": float(flip_initial_abs.sum()),
        "flipped_abs_refined_sum": float(flip_refined_abs.sum()),
        "all_same_matched_gt_face": int(np.count_nonzero(initial_gt_faces == refined_gt_faces)),
        "flipped_same_matched_gt_face": int(
            np.count_nonzero((initial_gt_faces == refined_gt_faces) & primary_flip)
        ),
        "initial_degenerate_faces": int(np.count_nonzero(initial_degenerate)),
        "refined_degenerate_faces": int(np.count_nonzero(refined_degenerate)),
        "new_degenerate_faces": int(np.count_nonzero(refined_degenerate & ~initial_degenerate)),
        "resolved_degenerate_faces": int(np.count_nonzero(initial_degenerate & ~refined_degenerate)),
        "initial_twice_area_sum": float(initial_cross_norm.sum()),
        "refined_twice_area_sum": float(refined_cross_norm.sum()),
    }
    for prefix, first, second in (
        ("all_signed", initial_signed, refined_signed),
        ("all_abs", initial_abs, refined_abs),
        ("flipped_signed", flip_initial_signed, flip_refined_signed),
        ("flipped_abs", flip_initial_abs, flip_refined_abs),
    ):
        row.update({f"{prefix}_{key}": value for key, value in _change_counts(first, second, ALIGNMENT_EPSILON).items()})
    row.update(_orientation_categories(flip_initial_signed, flip_refined_signed, ALIGNMENT_EPSILON))

    diagonal = float(np.linalg.norm(initial_vertices.max(axis=0) - initial_vertices.min(axis=0)))
    scale2 = max(diagonal * diagonal, 1e-300)
    for threshold in NEAR_ZERO_RELATIVE_THRESHOLDS:
        label = f"near_zero_{threshold:.0e}".replace("-", "m")
        initial_near = initial_cross_norm / scale2 <= threshold
        refined_near = refined_cross_norm / scale2 <= threshold
        row[f"initial_{label}"] = int(np.count_nonzero(initial_near))
        row[f"refined_{label}"] = int(np.count_nonzero(refined_near))
        row[f"new_{label}"] = int(np.count_nonzero(refined_near & ~initial_near))
        row[f"resolved_{label}"] = int(np.count_nonzero(initial_near & ~refined_near))
    stable_initial = initial_cross_norm / scale2 > 1e-10
    severe_collapse = stable_initial & (refined_cross_norm <= 0.01 * initial_cross_norm)
    row["new_severe_area_collapse_below_1pct"] = int(np.count_nonzero(severe_collapse))
    row["primary_flips_with_initial_or_refined_near_zero_1em10"] = int(
        np.count_nonzero(
            primary_flip
            & ((initial_cross_norm / scale2 <= 1e-10) | (refined_cross_norm / scale2 <= 1e-10))
        )
    )
    return row


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    dataset = PreparedMeshDataset.from_manifest(args.manifest, "test")
    if len(dataset) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} samples, found {len(dataset)}")
    if args.sample_id is None:
        indices = list(range(args.shard_index, len(dataset), args.shard_count))
    else:
        indices = [
            index
            for index, sample_id in enumerate(dataset.sample_ids)
            if str(sample_id) == args.sample_id
        ]
        if len(indices) != 1:
            raise ValueError(f"Expected one --sample-id match, found {indices}")
    rows: list[dict[str, Any]] = []
    for ordinal, index in enumerate(indices, start=1):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        refined_path = args.ours_results / "samples" / sample_id / "refined.obj"
        if not refined_path.is_file():
            raise FileNotFoundError(refined_path)
        refined = load_mesh(refined_path)
        row = analyze_sample(
            sample_id,
            _numpy(static["vertices"]),
            _numpy(static["faces"]),
            refined.vertices,
            refined.faces,
            _numpy(static["gt_vertices"]),
            _numpy(static["gt_faces"]),
        )
        rows.append(row)
        print(
            f"flip-diagnostic shard={args.shard_index} {ordinal}/{len(indices)} "
            f"sample={sample_id} flips={row['primary_flipped_faces']}",
            flush=True,
        )
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    csv_path = shard_dir / f"per_sample_{args.shard_index:03d}.csv"
    _write_rows(csv_path, rows)
    metadata = {
        "status": "completed",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "samples": len(rows),
        "face_count": sum(int(row["face_count"]) for row in rows),
        "primary_flipped_faces": sum(int(row["primary_flipped_faces"]) for row in rows),
        "csv": str(csv_path.resolve()),
    }
    (shard_dir / f"metadata_{args.shard_index:03d}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty diagnostic shard")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sum(rows: Iterable[dict[str, str]], field: str) -> float:
    return sum(float(row[field]) for row in rows)


def merge(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for shard in range(args.shard_count):
        path = args.output_dir / "shards" / f"per_sample_{shard:03d}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    rows.sort(key=lambda row: row["sample_id"])
    ids = [row["sample_id"] for row in rows]
    object_counts = Counter(sample_id.rpartition("__v")[0] for sample_id in ids)
    total_faces = int(_sum(rows, "face_count"))
    total_flips = int(_sum(rows, "primary_flipped_faces"))

    def alignment(prefix: str, count: int) -> dict[str, Any]:
        initial = _sum(rows, f"{prefix}_initial_sum") / max(count, 1)
        refined = _sum(rows, f"{prefix}_refined_sum") / max(count, 1)
        return {
            "count": count,
            "mean_initial": initial,
            "mean_refined": refined,
            "mean_change": refined - initial,
            "improved": int(_sum(rows, f"{prefix}_improved")),
            "worsened": int(_sum(rows, f"{prefix}_worsened")),
            "unchanged_or_ambiguous": int(_sum(rows, f"{prefix}_unchanged_or_ambiguous")),
            "improved_percentage": 100.0 * _sum(rows, f"{prefix}_improved") / max(count, 1),
            "worsened_percentage": 100.0 * _sum(rows, f"{prefix}_worsened") / max(count, 1),
        }

    categories = {
        key: int(_sum(rows, key))
        for key in (
            "wrong_to_correct",
            "correct_to_wrong",
            "wrong_to_wrong",
            "correct_to_correct_or_ambiguous",
        )
    }
    for key, value in list(categories.items()):
        categories[f"{key}_percentage"] = 100.0 * value / max(total_flips, 1)
    near_zero = {}
    for threshold in NEAR_ZERO_RELATIVE_THRESHOLDS:
        label = f"near_zero_{threshold:.0e}".replace("-", "m")
        near_zero[f"{threshold:.0e}"] = {
            "definition": f"cross_norm / initial_bbox_diagonal^2 <= {threshold:.0e}",
            "initial": int(_sum(rows, f"initial_{label}")),
            "refined": int(_sum(rows, f"refined_{label}")),
            "new": int(_sum(rows, f"new_{label}")),
            "resolved": int(_sum(rows, f"resolved_{label}")),
        }

    primary_csv = args.ours_results / "per_sample.csv"
    primary_flip_total = None
    if primary_csv.is_file():
        with primary_csv.open(newline="", encoding="utf-8") as handle:
            primary_rows = list(csv.DictReader(handle))
        primary_flip_total = sum(int(row["introduced_flipped_faces"]) for row in primary_rows)
    audit = {
        "exactly_1000_unique_samples": len(rows) == len(set(ids)) == args.expected_samples,
        "exactly_200_objects_x_5_variants": len(object_counts) == 200
        and set(object_counts.values()) == {5},
        "primary_flip_total_reproduced": primary_flip_total == total_flips,
        "all_connectivity_preserved": True,
        "analysis_is_separate_from_primary_report": args.output_dir.resolve()
        != args.ours_results.resolve(),
    }
    payload = {
        "experiment": "future2000_ours_200k_flip_harmfulness_diagnostic",
        "contract_audit": all(audit.values()),
        "contract_checks": audit,
        "definitions": {
            "introduced_flipped_face": "dot(cross_initial, cross_refined) < 0 for the same indexed triangle; GT is not used",
            "primary_degenerate_face": "norm(cross) <= 1e-14; primary new-degeneracy count is refined_degenerate and not initial_degenerate",
            "primary_normal_consistency": "orientation-invariant absolute cosine: abs(dot); for different topology, bidirectional nearest-surface normal matching",
            "diagnostic_gt_normal": "oriented normal of the exact closest GT triangle to each initial/refined face centroid, matched independently for each state",
            "diagnostic_correct": f"signed dot(face_normal, matched_GT_face_normal) > {ALIGNMENT_EPSILON}",
            "diagnostic_wrong": f"signed dot(face_normal, matched_GT_face_normal) < -{ALIGNMENT_EPSILON}",
            "diagnostic_ambiguous": f"absolute signed alignment <= {ALIGNMENT_EPSILON}",
        },
        "samples": len(rows),
        "objects": len(object_counts),
        "total_faces": total_faces,
        "total_flipped_faces": total_flips,
        "flipped_face_percentage": 100.0 * total_flips / total_faces,
        "all_faces": {
            "signed_gt_alignment": alignment("all_signed", total_faces),
            "orientation_invariant_abs_alignment": alignment("all_abs", total_faces),
            "same_matched_gt_face": int(_sum(rows, "all_same_matched_gt_face")),
        },
        "flipped_faces": {
            "signed_gt_alignment": alignment("flipped_signed", total_flips),
            "orientation_invariant_abs_alignment": alignment("flipped_abs", total_flips),
            "classification": categories,
            "same_matched_gt_face": int(_sum(rows, "flipped_same_matched_gt_face")),
            "with_initial_or_refined_near_zero_1e-10": int(
                _sum(rows, "primary_flips_with_initial_or_refined_near_zero_1em10")
            ),
        },
        "mesh_invalidity": {
            "strict_degenerate": {
                "initial": int(_sum(rows, "initial_degenerate_faces")),
                "refined": int(_sum(rows, "refined_degenerate_faces")),
                "new": int(_sum(rows, "new_degenerate_faces")),
                "resolved": int(_sum(rows, "resolved_degenerate_faces")),
            },
            "near_zero_sensitivity": near_zero,
            "new_severe_area_collapse_below_1pct": int(
                _sum(rows, "new_severe_area_collapse_below_1pct")
            ),
            "initial_total_surface_area": 0.5 * _sum(rows, "initial_twice_area_sum"),
            "refined_total_surface_area": 0.5 * _sum(rows, "refined_twice_area_sum"),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_dir / "per_sample.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "FLIP_HARMFULNESS_REPORT.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    all_signed = payload["all_faces"]["signed_gt_alignment"]
    flip_signed = payload["flipped_faces"]["signed_gt_alignment"]
    flip_abs = payload["flipped_faces"]["orientation_invariant_abs_alignment"]
    categories = payload["flipped_faces"]["classification"]
    invalidity = payload["mesh_invalidity"]
    strict = invalidity["strict_degenerate"]
    lines = [
        "# Future2000 Ours-200k introduced-face-flip harmfulness diagnostic",
        "",
        f"Contract audit: **{str(payload['contract_audit']).lower()}**.",
        "",
        "This is an additional read-only diagnostic. It does not modify or overwrite the frozen benchmark metrics.",
        "",
        "## Exact existing definitions",
        "",
        f"- `introduced flipped face`: {payload['definitions']['introduced_flipped_face']}.",
        f"- `new degenerate face`: {payload['definitions']['primary_degenerate_face']}.",
        f"- Existing `normal_consistency`: {payload['definitions']['primary_normal_consistency']}.",
        f"- Diagnostic GT correspondence: {payload['definitions']['diagnostic_gt_normal']}.",
        "",
        "## GT-normal attribution",
        "",
        f"- Total faces: {payload['total_faces']}",
        f"- Reported introduced flips reproduced exactly: {payload['total_flipped_faces']} "
        f"({payload['flipped_face_percentage']:.3f}% of faces)",
        "",
        "| Population | Faces | Mean signed dot initial | Mean signed dot refined | Improved | Worsened |",
        "|---|---:|---:|---:|---:|---:|",
        f"| All faces | {all_signed['count']} | {all_signed['mean_initial']:.6f} | "
        f"{all_signed['mean_refined']:.6f} | {all_signed['improved_percentage']:.2f}% | "
        f"{all_signed['worsened_percentage']:.2f}% |",
        f"| Introduced flips only | {flip_signed['count']} | {flip_signed['mean_initial']:.6f} | "
        f"{flip_signed['mean_refined']:.6f} | {flip_signed['improved_percentage']:.2f}% | "
        f"{flip_signed['worsened_percentage']:.2f}% |",
        "",
        "Introduced-flip orientation classes:",
        "",
        f"- wrong -> correct: {categories['wrong_to_correct']} "
        f"({categories['wrong_to_correct_percentage']:.2f}%)",
        f"- correct -> wrong: {categories['correct_to_wrong']} "
        f"({categories['correct_to_wrong_percentage']:.2f}%)",
        f"- wrong -> wrong: {categories['wrong_to_wrong']} "
        f"({categories['wrong_to_wrong_percentage']:.2f}%)",
        f"- correct -> correct / ambiguous: {categories['correct_to_correct_or_ambiguous']} "
        f"({categories['correct_to_correct_or_ambiguous_percentage']:.2f}%)",
        "",
        f"On flipped faces, orientation-invariant `abs(dot)` changes from "
        f"{flip_abs['mean_initial']:.6f} to {flip_abs['mean_refined']:.6f}; "
        f"it improves for {flip_abs['improved_percentage']:.2f}% and worsens for "
        f"{flip_abs['worsened_percentage']:.2f}%.",
        "",
        "## Actual invalidity indicators",
        "",
        f"- Strict degenerate faces (`norm(cross) <= 1e-14`): initial {strict['initial']}, "
        f"refined {strict['refined']}, new {strict['new']}, resolved {strict['resolved']}.",
    ]
    for threshold, values in invalidity["near_zero_sensitivity"].items():
        lines.append(
            f"- Near-zero sensitivity `{threshold}`: initial {values['initial']}, "
            f"refined {values['refined']}, new {values['new']}, resolved {values['resolved']} "
            f"({values['definition']})."
        )
    lines.extend(
        [
            f"- New severe area collapses below 1% of initial face area: "
            f"{invalidity['new_severe_area_collapse_below_1pct']}.",
            "",
            "## Answer",
            "",
            _conclusion(payload),
            "",
        ]
    )
    return "\n".join(lines)


def _conclusion(payload: dict[str, Any]) -> str:
    categories = payload["flipped_faces"]["classification"]
    signed = payload["flipped_faces"]["signed_gt_alignment"]
    invalidity = payload["mesh_invalidity"]
    corrective = categories["wrong_to_correct_percentage"]
    harmful = categories["correct_to_wrong_percentage"]
    new_degenerate = invalidity["strict_degenerate"]["new"]
    if corrective > harmful and signed["mean_change"] > 0 and new_degenerate == 0:
        return (
            "The face-flip count is not equivalent to mesh degradation. GT-referenced signed "
            "alignment shows more corrective than harmful orientation transitions, while strict "
            "degeneracy does not increase. Treat the primary flip count as a displacement/winding "
            "diagnostic, not as an invalid-face count."
        )
    if harmful > corrective and signed["mean_change"] < 0:
        return (
            "Most attributable orientation transitions are harmful relative to GT, so the large "
            "face-flip count reflects genuine local orientation degradation even though it is not "
            "itself a topological-invalidity definition."
        )
    return (
        "The result is mixed: the raw flip count combines corrective and harmful orientation "
        "changes. Interpret it together with signed GT alignment and near-zero/degeneracy counts, "
        "not as a standalone mesh-invalidity metric."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ours-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--expected-samples", type=int, default=1000)
    parser.add_argument("--sample-id")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        result = merge(args)
    else:
        if args.manifest is None:
            parser.error("--manifest is required unless --merge is used")
        if not 0 <= args.shard_index < args.shard_count:
            parser.error("--shard-index must be in [0, shard-count)")
        result = run_shard(args)
    print(json.dumps(result, indent=2))
    return 0 if result.get("contract_audit", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
