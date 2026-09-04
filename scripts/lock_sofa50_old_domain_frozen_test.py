#!/usr/bin/env python3
from __future__ import annotations

"""Lock frozen old-domain B/E validation selections for one E/Hybrid test opening."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-audit", required=True, type=Path)
    parser.add_argument("--specialist-summary", required=True, type=Path)
    parser.add_argument("--lambda-selection", required=True, type=Path)
    parser.add_argument("--frozen-validation", required=True, type=Path)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--final-test-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    final_test = args.final_test_dir.resolve()
    if output.exists():
        raise RuntimeError("Frozen test authorization already exists; refusing to overwrite it")
    if final_test.exists():
        raise RuntimeError("Frozen final-test output already exists; refusing another test opening")

    audit = read_json(args.contract_audit.resolve())
    specialists = read_json(args.specialist_summary.resolve())
    selection = read_json(args.lambda_selection.resolve())
    frozen = read_json(args.frozen_validation.resolve())
    benchmark = read_json(args.benchmark_manifest.resolve())
    b_checkpoint = Path(specialists["arm_b_checkpoint"]).resolve()
    e_checkpoint = Path(specialists["arm_e_checkpoint"]).resolve()
    b_sha = sha256_file(b_checkpoint)
    e_sha = sha256_file(e_checkpoint)
    sample_ids = benchmark.get("sample_ids", [])

    checks = {
        "historical_contract_passed": audit.get("contract_audit") is True,
        "specialists_validation_only": specialists.get("contract_audit") is True
        and specialists.get("split") == "validation"
        and specialists.get("test_opened") is False,
        "lambda_validation_only": selection.get("contract_audit") is True
        and selection.get("selection_split") == "validation"
        and selection.get("test_accessed") is False,
        "frozen_validation_only": frozen.get("contract_audit") is True
        and frozen.get("split") == "validation"
        and frozen.get("test_accessed") is False,
        "selected_lambda_unchanged": float(selection["selected_lambda"])
        == float(frozen["selected_lambda"]),
        "checkpoint_shas_unchanged": b_sha
        == specialists["arm_b_checkpoint_sha256"]
        == selection["arm_b_checkpoint_sha256"]
        == frozen["arm_b_checkpoint_sha256"]
        and e_sha
        == specialists["arm_e_checkpoint_sha256"]
        == selection["arm_e_checkpoint_sha256"]
        == frozen["arm_e_checkpoint_sha256"],
        "exact_test_count": len(sample_ids) == 25 and len(set(sample_ids)) == 25,
        "test_output_absent_before_lock": not final_test.exists(),
    }
    contract = all(checks.values())
    authorization = {
        "contract_audit": contract,
        "contract_checks": checks,
        "scope": "old_domain_frozen_b_e_vs_nds_nvdiffrec_exmesh",
        "final_selection_locked": True,
        "validation_only_selection": True,
        "authorize_single_test_open": contract,
        "test_output_directory": str(final_test),
        "arm_b_checkpoint": str(b_checkpoint),
        "arm_b_checkpoint_sha256": b_sha,
        "arm_e_checkpoint": str(e_checkpoint),
        "arm_e_checkpoint_sha256": e_sha,
        "lambda_old": float(selection["selected_lambda"]),
        "lambda_grid": selection["lambda_grid"],
        "selected_at_grid_boundary": bool(selection["selected_at_grid_boundary"]),
        "specialist_summary_sha256": sha256_file(args.specialist_summary.resolve()),
        "lambda_selection_sha256": sha256_file(args.lambda_selection.resolve()),
        "frozen_validation_sha256": sha256_file(args.frozen_validation.resolve()),
        "benchmark_manifest": str(args.benchmark_manifest.resolve()),
        "benchmark_manifest_sha256": sha256_file(args.benchmark_manifest.resolve()),
        "test_sample_ids_sha256": hashlib.sha256(
            ("\n".join(sample_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "arm_b_test_previously_opened": True,
        "arm_e_or_frozen_test_open_count_before_authorization": 0,
        "test_metric_used_to_select_e_or_lambda": False,
        "authorization_source": (
            "User explicitly requested on 2026-08-28: run the test set and add the "
            "previous three external comparators."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(authorization, indent=2, sort_keys=True))
    if not contract:
        raise RuntimeError(f"Frozen test selection lock failed: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
