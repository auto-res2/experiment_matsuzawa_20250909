"""src/evaluate.py
Plotting & evaluation helpers.
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless back-end
import matplotlib.pyplot as plt  # type: ignore

# -----------------------------------------------------------------------------
# Configuration (optional – currently unused but loaded for completeness)
# -----------------------------------------------------------------------------
_cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as _f:
    CONF = yaml.safe_load(_f)

# -----------------------------------------------------------------------------
# Public functions
# -----------------------------------------------------------------------------

def plot_line(xs: List[int], ys: List[float], title: str, ylabel: str, fname: str) -> None:
    """Simple line plot that annotates values and stores a PDF under `fname`."""
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.2f}")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname, bbox_inches="tight", format="pdf")
    plt.close()
