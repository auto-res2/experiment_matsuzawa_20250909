import os
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # head-less back-end
import matplotlib.pyplot as plt
import seaborn as sns


def plot_line(x: List[int], ys: List[List[float]], labels: List[str],
              xlabel: str, ylabel: str, title: str, outfile: os.PathLike) -> str:
    """Utility helper to draw and save publication-quality line plots."""
    plt.figure(figsize=(6, 4))
    for y, lbl in zip(ys, labels):
        plt.plot(x, y, marker="o", label=lbl)
        for xi, yi in zip(x, y):
            plt.annotate(f"{yi:.2f}", (xi, yi))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.close()
    return str(outfile)
