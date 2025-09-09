"""
evaluate.py – metrics, statistics, and plotting helpers
"""
from __future__ import annotations

import json, pathlib
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns
import torch

# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / ".research" / "iteration7" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  Basic classification evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, loader: Iterable):
        correct = 0
        total = 0
        for img, label in loader:
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            pred = model(img).argmax(1)
            correct += (pred == label).sum().item()
            total += label.size(0)
        return {"accuracy": correct / total}

# ---------------------------------------------------------------------------
#  Very small plotting helper (used by main.py)
# ---------------------------------------------------------------------------

def line_plot(x, y, title: str, xlabel: str, ylabel: str, filename: str):
    plt.figure()
    sns.lineplot(x=x, y=y, marker="o")
    for xi, yi in zip(x, y):
        plt.text(xi, yi, f"{yi:.2f}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([ylabel])
    plt.savefig(IMG_DIR / filename, bbox_inches="tight", format="pdf")
    plt.close()
