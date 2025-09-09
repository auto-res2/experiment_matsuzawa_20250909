"""src/evaluate.py – evaluation utilities and plotting"""
from __future__ import annotations

from typing import List

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocess import IMAGE_DIR

# ---------------------------------------------------------------------------
#                               METRICS
# ---------------------------------------------------------------------------

def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return (pred.argmax(dim=-1) == y).float().mean().item() * 100.0


def effective_rank(X: torch.Tensor) -> float:
    """Effective rank = exp(H), H – entropy of singular values."""
    U, S, _ = torch.linalg.svd(X.float(), full_matrices=False)
    p = S / S.sum()
    H = -(p * torch.log(p + 1e-10)).sum()
    return torch.exp(H).item() / X.size(1)

# ---------------------------------------------------------------------------
#                               PLOTTING
# ---------------------------------------------------------------------------

def plot_metric(xs: List, ys: List, *, xlabel: str, ylabel: str, title: str, filename: str):
    plt.figure(figsize=(5, 3))
    sns.lineplot(x=xs, y=ys, marker="o")
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(IMAGE_DIR / filename, bbox_inches="tight")
    plt.close()
