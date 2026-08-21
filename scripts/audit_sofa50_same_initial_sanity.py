#!/usr/bin/env python3
from __future__ import annotations

"""Gate the full Sofa50 same-initial benchmark on representative-run evidence."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = ("ours", "exmesh", "nds", "nvdiffrec")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_status(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload.get("row", payload)
    if not isinstance(row, dict):
        raise ValueError(f"Invalid status row: {path}")
    return payload, row


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_id = str(manifest["representative_sample_id"])
    source = next(row for row in manifest["samples"] if row["sample_id"] == sample_id)
    expected_sha = str(source["common_initial_mesh_sha256"])
    expected_images = [str(Path(value).resolve()) for value in source["image_paths"]]
    coordinate_path = args.coordinate_audit.resolve()
    coordinate = json.loads(coordinate_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    check("coordinate_projection_audit", coordinate.get("contract_audit") is True, str(coordinate_path))
    check("representative_sample_matches_coordinate_audit", coordinate.get("sample_id") == sample_id, coordinate.get("sample_id"))
    initial_path = Path(str(source["common_initial_mesh"]))
    observed_sha = _sha256(initial_path) if initial_path.is_file() else None
    check("canonical_initial_sha_unchanged", observed_sha == expected_sha, {"expected": expected_sha, "observed": observed_sha})

    method_rows: dict[str, Any] = {}
    for method in METHODS:
        status_path = args.sanity_root / method / "samples" / sample_id / "status.json"
        check(f"{method}_status_exists", status_path.is_file(), str(status_path))
        if not status_path.is_file():
            continue
        payload, row = _load_status(status_path)
        method_rows[method] = row
        check(f"{method}_completed", row.get("status") == "completed", row.get("status"))
        check(f"{method}_sample_id", row.get("sample_id") == sample_id, row.get("sample_id"))
        check(f"{method}_common_initial_sha", row.get("common_initial_mesh_sha256") == expected_sha, row.get("common_initial_mesh_sha256"))
        check(f"{method}_view_count", int(row.get("view_count", -1)) == 28, row.get("view_count"))
        check(f"{method}_final_mesh_exists", Path(str(row.get("final_mesh", ""))).is_file(), row.get("final_mesh"))
        if method == "ours":
            check("ours_source_identity", row.get("common_initial_identity_audit") is True, row.get("common_initial_identity_audit"))
            sample_root = status_path.parent
            for name in (
                "predicted_raw_laplacian.npy",
                "predicted_confidence.npy",
                "recovery_weight.npy",
                "visibility_used.npz",
                "recovery_config.json",
            ):
                check(f"ours_artifact_{name}", (sample_root / name).is_file(), str(sample_root / name))
            continue
        check(f"{method}_source_identity", row.get("common_initial_source_identity_audit") is True, row.get("common_initial_source_identity_audit"))
        check(f"{method}_adapter_identity", row.get("common_initial_identity_audit") is True, row.get("common_initial_identity_audit"))
        contract_path = status_path.parent / "input_contract.json"
        check(f"{method}_input_contract_exists", contract_path.is_file(), str(contract_path))
        if contract_path.is_file():
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            observed_images = [str(Path(value).resolve()) for value in contract.get("source_images", [])]
            check(f"{method}_same_rgb_paths", observed_images == expected_images, {"count": len(observed_images)})
            check(f"{method}_forbidden_gt_fields", contract.get("forbidden_fields_consumed") == [], contract.get("forbidden_fields_consumed"))
        commands = payload.get("commands", [])
        joined = "\n".join(str(value) for value in commands)
        if method == "nds":
            check("nds_custom_initial_argument", "--initial_mesh" in joined, joined)
            check("nds_visual_hull_not_requested", "visual_hull" not in joined.lower(), joined)
        elif method == "exmesh":
            check("exmesh_pgsr_not_requested", "pgsr" not in joined.lower() and "tsdf" not in joined.lower(), joined)
        elif method == "nvdiffrec":
            check("nvdiffrec_fixed_topology_output", row.get("output_connectivity_preserved") is True, row.get("output_connectivity_preserved"))

    passed = bool(checks) and all(item["passed"] for item in checks)
    result = {
        "contract_audit": passed,
        "full_benchmark_submission_allowed": passed,
        "sample_id": sample_id,
        "common_initial_mesh": str(initial_path),
        "common_initial_mesh_sha256": expected_sha,
        "methods": list(METHODS),
        "checks": checks,
        "failed_checks": [item for item in checks if not item["passed"]],
        "method_rows": method_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--coordinate-audit", required=True, type=Path)
    parser.add_argument("--sanity-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    result = audit(parser.parse_args())
    print(json.dumps({"contract_audit": result["contract_audit"], "failed_checks": len(result["failed_checks"]), "output": str(Path(result["common_initial_mesh"]).parent)}, indent=2))
    return 0 if result["contract_audit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
