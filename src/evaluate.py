"""evaluate.py
Evaluation utilities independent from the training loop: statistics,
plots, confusion matrices …  Currently only plotting of training curves
is required for the refactor.
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Directory where all images must be stored (mandatory iteration3 path)
# -----------------------------------------------------------------------------
IMG_DIR = Path('.research/iteration3/images')
IMG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------

def _save(fig, name: str) -> str:
    path = IMG_DIR / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)

# -----------------------------------------------------------------------------

def generate_figures(results: Dict[str, Any], cfg: Dict[str, Any]):
    """Create and save the training-loss & validation-accuracy curves."""
    figs: list[str] = []

    # -- training loss -----------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(results["train_loss"], label="train_loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss"); ax.legend()
    figs.append(_save(fig, f"train_loss_{cfg['experiment_name']}"))

    # -- validation accuracy ----------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(results["val_acc_curve"], label="val_acc")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend()
    figs.append(_save(fig, f"val_acc_{cfg['experiment_name']}"))

    results["figures"] = figs
    return figs
