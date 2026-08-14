from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_local_shell_scripts_never_submit_scheduler_jobs() -> None:
    for path in sorted((ROOT / "scripts/local").glob("*.sh")):
        text = path.read_text(encoding="utf-8")
        assert "#SBATCH" not in text, path
        assert "sbatch" not in text, path
        assert "srun" not in text, path


def test_local_comparison_runner_has_no_scheduler_commands() -> None:
    path = ROOT / "scripts/local/run_future2000_comparisons.sh"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH" not in text
    assert "sbatch" not in text
    assert "srun" not in text
    assert "/networkhome/" not in text


def test_local_runner_lists_complete_strict_task_set() -> None:
    path = ROOT / "scripts/local/run_future2000_comparisons.sh"
    result = subprocess.run(
        ["bash", str(path), "list"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for task in (
        "learned",
        "openmvs",
        "nds",
        "nerf2mesh",
        "da3",
        "exmesh",
        "qualitative",
        "report",
        "all",
    ):
        assert task in result.stdout


def test_local_setup_contains_fixes_for_observed_environment_failures() -> None:
    text = (
        ROOT / "scripts/local/setup_future2000_comparison_envs.sh"
    ).read_text(encoding="utf-8")
    assert "setuptools<81" in text
    assert "cuda-cudart-dev" in text
    assert "env -u VCPKG_ROOT" in text
    assert "#SBATCH" not in text
    assert "sbatch" not in text
