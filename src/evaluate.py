"""
evaluate.py – metrics & plotting utilities extracted from original utils/metrics.py
and utils/plots.py.
"""
from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt  # noqa: E402
import torch


# -----------------------------------------------------------------------------
# Accuracy helper identical to original implementation
# -----------------------------------------------------------------------------

def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy – identical to the experiment code."""
    return (pred.argmax(1) == target).float().mean().item()


# -----------------------------------------------------------------------------
# Plot helper identical to utils/plots.line_plot
# -----------------------------------------------------------------------------

def line_plot(
    xs: Sequence[float] | Sequence[int],
    ys: Sequence[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    pdf_path: str | pathlib.Path,
) -> None:
    pdf_path = pathlib.Path(pdf_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
