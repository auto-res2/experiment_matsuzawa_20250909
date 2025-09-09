# evaluate.py
"""Evaluation utilities: accuracy computation, metrics aggregation and plotting."""

from __future__ import annotations

from typing import Literal, Sequence
import torch
import matplotlib

matplotlib.use("Agg")  # headless back-end (servers / HPC)
import matplotlib.pyplot as plt  # noqa: E402 – after backend selection

# ---------------------------------------------------------------------------

def evaluate_model(model, data, split: Literal["train", "val", "test", "all"] = "val") -> float:
    """Return classification accuracy for the requested split."""

    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=-1)

    if split == "all":
        mask = torch.ones_like(data.y, dtype=torch.bool)
    else:
        mask = getattr(data, f"{split}_mask")
    correct = (preds[mask] == data.y[mask]).sum().item()
    return correct / int(mask.sum())

# ---------------------------------------------------------------------------


def line_plot(
    xs: Sequence[float],
    ys: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    pdf_file: str,
) -> None:
    """Utility for 2-D line plots – used for learning curves."""

    plt.figure()
    plt.plot(xs, ys, label=ylabel)
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.3f}", (x, y))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_file, bbox_inches="tight")
    plt.close()
