#!/usr/bin/env python3
from __future__ import annotations

"""Lock every validation-selected artifact before authorizing one sealed test run."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


REQUIRED_STEPS = (0, 100, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000)


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
    parser.add_argument("--step0", required=True, type=Path)
    parser.add_argument("--continuous-run", required=True, type=Path)
    parser.add_argument("--continuous-validation", required=True, type=Path)
    parser.add_argument("--final-test-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    audit = read_json(args.contract_audit.resolve())
    specialists = read_json(args.specialist_summary.resolve())
    selection = read_json(args.lambda_selection.resolve())
    step0 = read_json(args.step0.resolve())
    validation = read_json(args.continuous_validation.resolve())
    run = args.continuous_run.resolve()
    checkpoint = run / "checkpoint_best.pt"
    latest_checkpoint = run / "checkpoint_latest.pt"
    if args.final_test_dir.resolve().exists():
        raise RuntimeError("Final test output already exists; refusing a second test authorization")
    required_checkpoints = {
        step: run / "checkpoints" / f"checkpoint_step_{step:06d}.pt" for step in REQUIRED_STEPS
    }
    missing = [str(path) for path in required_checkpoints.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required continuous checkpoints are missing: {missing}")
    if not latest_checkpoint.is_file():
        raise RuntimeError("Continuous checkpoint_latest.pt is missing")
    latest_payload = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
    optimizer_steps = int(latest_payload.get("optimizer_steps", -1))
    checkpoint_sha = sha256_file(checkpoint)
    validation_checkpoint = Path(validation["checkpoint"]).resolve()
    checks = {
        "historical_contract_passed": audit.get("contract_audit") is True,
        "specialists_validation_only": specialists.get("contract_audit") is True
        and specialists.get("split") == "validation"
        and specialists.get("test_opened") is False,
        "lambda_validation_only": selection.get("contract_audit") is True
        and selection.get("selection_split") == "validation"
        and selection.get("test_accessed") is False,
        "step0_validation_only": step0.get("contract_audit") is True
        and step0.get("split") == "validation"
        and step0.get("test_accessed") is False,
        "continuous_selected_on_validation": validation.get("selection_eligible") is True
        and validation.get("split") == "validation",
        "continuous_checkpoint_identity": validation_checkpoint == checkpoint.resolve()
        and validation.get("checkpoint_sha256") == checkpoint_sha,
        "training_completed_20000_steps": optimizer_steps == 20000,
        "all_required_checkpoints_present": not missing,
        "selected_lambda_unchanged": float(selection["selected_lambda"])
        == float(step0["solver"]["lambda"])
        == float(validation["solver"]["lambda"]),
        "checkpoint_shas_unchanged": specialists["arm_b_checkpoint_sha256"]
        == step0["checkpoint_identity"]["B"]["sha256"]
        and specialists["arm_e_checkpoint_sha256"]
        == step0["checkpoint_identity"]["E"]["sha256"],
        "continuous_solver_passed": validation["solver"]["failed"] == 0
        and float(validation["solver"]["relative_residual_max"]) <= 1e-8,
        "test_output_absent_before_lock": not args.final_test_dir.resolve().exists(),
    }
    authorization = {
        "contract_audit": all(checks.values()),
        "contract_checks": checks,
        "final_selection_locked": True,
        "validation_only_selection": True,
        "authorize_single_test_open": all(checks.values()),
        "test_open_count_before_authorization": 0,
        "test_output_directory": str(args.final_test_dir.resolve()),
        "arm_b_checkpoint": specialists["arm_b_checkpoint"],
        "arm_b_checkpoint_sha256": specialists["arm_b_checkpoint_sha256"],
        "arm_e_checkpoint": specialists["arm_e_checkpoint"],
        "arm_e_checkpoint_sha256": specialists["arm_e_checkpoint_sha256"],
        "lambda_old": float(selection["selected_lambda"]),
        "continuous_checkpoint": str(checkpoint),
        "continuous_checkpoint_sha256": checkpoint_sha,
        "continuous_optimizer_steps": optimizer_steps,
        "required_checkpoint_shas": {
            str(step): sha256_file(path) for step, path in required_checkpoints.items()
        },
        "test_metric_used_before_lock": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(authorization, indent=2, sort_keys=True))
    if not authorization["contract_audit"]:
        raise RuntimeError(f"Final selection lock failed: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
