# src/evaluate.py
"""Evaluation, statistics and visualisation helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .preprocess import FIG_DIR, RESULTS_DIR

# ────────────────────────────────────────────────────────────────────────────────
# Save results to disk -----------------------------------------------------------

def save_results(exp_key: str, results: Dict) -> Path:
    json_path = RESULTS_DIR / f"{exp_key}_results.json"
    with open(json_path, "w") as fp:
        json.dump(results, fp, indent=2)
    return json_path

# ────────────────────────────────────────────────────────────────────────────────
# Plot validation accuracy curves ------------------------------------------------

def plot_results(exp_key: str, results: Dict, title: str) -> Path:
    fig = plt.figure(figsize=(6, 4))
    for seed, res in results.items():
        epochs = res["history"]["epoch"]
        vals = res["history"]["val_acc"]
        plt.plot(epochs, vals, label=f"seed{seed}")
        for x, y in zip(epochs, vals):
            plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6)

    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title(title)
    plt.legend()
    fig_path = FIG_DIR / f"accuracy_{exp_key}.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    return fig_path
