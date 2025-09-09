# src/evaluate.py
"""Evaluation utilities: metrics, validation loop, simple plotting."""
from __future__ import annotations

import pathlib
from typing import List

import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import Accuracy as _Accuracy

plt.rcParams["pdf.fonttype"] = 42  # arXiv-friendly fonts

__all__ = ["accuracy", "evaluate", "line_plot"]


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def accuracy(task: str = "multiclass", num_classes: int = 2) -> _Accuracy:  # noqa: D401
    """Return a TorchMetrics Accuracy object with sane defaults."""
    return _Accuracy(task=task, num_classes=num_classes)


# -----------------------------------------------------------------------------
# Validation loop
# -----------------------------------------------------------------------------


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """Run the *model* over *loader* without gradient tracking and return accuracy."""
    model.eval()
    acc_metric = accuracy().to(device)
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch["image"].to(device), batch["label"].to(device)
            preds = model(imgs)
            acc_metric.update(preds, labels)
    return acc_metric.compute().item()


# -----------------------------------------------------------------------------
# Quick-and-dirty plotting helpers
# -----------------------------------------------------------------------------


def _annotate(ax, ys: List[float]) -> None:
    for i, v in enumerate(ys, 1):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6)


def line_plot(values: List[float], title: str, ylabel: str, outfile: pathlib.Path) -> None:
    """Simple line plot of *values* → *outfile* (PDF)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(values) + 1), values, marker="o", label=title)
    _annotate(ax, values)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
