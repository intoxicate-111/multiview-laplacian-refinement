#!/usr/bin/env python3
from __future__ import annotations

"""Plot continuous B+E test checkpoint geometry and quality trajectories."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps = np.asarray([int(row["step"]) for row in rows])

    def values(field: str) -> np.ndarray:
        return np.asarray([float(row[field]) for row in rows])

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(steps, values("test_chamfer"), "o-", label="Chamfer")
    best_index = int(np.argmin(values("test_chamfer")))
    ax.scatter(
        [steps[best_index]],
        [values("test_chamfer")[best_index]],
        color="crimson",
        zorder=5,
        label=f"lowest: step {steps[best_index]}",
    )
    ax.set_title("Matched-v2 test Chamfer")
    ax.set_ylabel("distance")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, values("test_vertex_rms"), "o-", label="vertex RMS")
    ax.plot(steps, values("test_p2s_p95"), "s-", label="P2S p95")
    ax.set_title("Same-index and tail geometry")
    ax.set_ylabel("distance")
    ax.legend()

    ax = axes[1, 0]
    curvature_fields = {
        "2H magnitude p95": "twice_mean_curvature_magnitude_error_p95",
        "dihedral mean": "dihedral_angle_error_degrees_mean",
        "face-normal mean": "face_normal_angle_error_degrees_mean",
    }
    for label, field in curvature_fields.items():
        data = values(field)
        ax.plot(steps, 100.0 * (data / data[0] - 1.0), "o-", label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Curvature/angle error relative to step 0")
    ax.set_ylabel("change (%) — lower is better")
    ax.legend()

    ax = axes[1, 1]
    distortion_fields = {
        "edge log-ratio mean": "absolute_log_edge_length_ratio_mean",
        "area log-ratio mean": "absolute_log_face_area_ratio_mean",
        "introduced flips": "introduced_flips",
    }
    for label, field in distortion_fields.items():
        data = values(field)
        ax.plot(steps, 100.0 * (data / data[0] - 1.0), "o-", label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Mesh distortion relative to step 0")
    ax.set_ylabel("change (%) — lower is better")
    ax.legend()

    for ax in axes.ravel():
        ax.set_xlabel("optimizer step")
        ax.grid(True, alpha=0.25)
        ax.ticklabel_format(style="plain", axis="x")
    figure.suptitle("Continuous pretrained Arm-B + Arm-E test trajectory", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
