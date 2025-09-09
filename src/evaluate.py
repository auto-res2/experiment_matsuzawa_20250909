# src/evaluate.py
"""Evaluation, statistics and visualisation helpers.

All numerical results (JSON) are written to
    .research/iteration11/
All plots are written to
    .research/iteration11/images/

Both folders are created on demand.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────────
# Paths – must follow the mandatory specification
# ────────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = ROOT / ".research" / "iteration11"
IMAGE_DIR = RESEARCH_DIR / "images"

# create folders if they do not exist -------------------------------------------
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# Save results to disk -----------------------------------------------------------


def save_results(exp_key: str, results: Dict) -> Path:
    """Save *results* as JSON under the mandatory .research/iteration11/ path."""
    json_path = RESEARCH_DIR / f"{exp_key}_results.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    return json_path

# ────────────────────────────────────────────────────────────────────────────────
# Plot validation accuracy curves ------------------------------------------------


def plot_results(exp_key: str, results: Dict, title: str) -> Path:
    """Plot the per-epoch validation accuracies for every seed of the experiment."""
    fig = plt.figure(figsize=(6, 4))
    for seed, res in results.items():
        epochs = res["history"]["epoch"]
        vals = res["history"]["val_acc"]
        plt.plot(epochs, vals, label=f"seed{seed}")

        # annotate every point with its accuracy value for better readability
        for x, y in zip(epochs, vals):
            plt.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=6,
            )

    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title(title)
    plt.legend(loc="best")

    fig_path = IMAGE_DIR / f"accuracy_{exp_key}.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    return fig_path
