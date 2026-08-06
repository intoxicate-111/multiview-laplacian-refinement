#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a saved multi-mesh training history.")
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    history = json.loads(args.history.read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in history]
    validation_rows = [row for row in history if row.get("validation_loss") is not None]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0, 0].set_title("Weighted training objective")
    axes[0, 1].plot(
        [row["epoch"] for row in validation_rows],
        [row["validation_loss"] for row in validation_rows],
        marker="o",
        markersize=3,
        label="validation",
    )
    axes[0, 1].set_title("Weighted validation objective")
    axes[1, 0].plot(
        epochs,
        [row["train_exact_query_loss"] for row in history],
        label="exact",
        alpha=0.8,
    )
    axes[1, 0].plot(
        epochs,
        [row["train_perturbed_query_loss"] for row in history],
        label="perturbed",
        alpha=0.8,
    )
    axes[1, 0].set_title("Training query subsets")
    axes[1, 0].legend()
    axes[1, 1].plot(
        epochs, [row["data_loading_seconds"] for row in history], label="data"
    )
    axes[1, 1].plot(
        epochs,
        [row["gpu_transfer_seconds"] for row in history],
        label="transfer",
    )
    axes[1, 1].plot(
        epochs,
        [row["forward_backward_seconds"] for row in history],
        label="forward/backward",
    )
    axes[1, 1].set_title("Epoch timing")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
