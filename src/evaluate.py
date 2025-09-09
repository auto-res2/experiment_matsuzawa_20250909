"""src/evaluate.py
Plotting utilities and evaluation helpers.
"""
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # head-less back-end (safe for clusters)


def plot_training_curves(
    history_train_loss: List[float],
    history_val_acc: List[float],
    cfg,
    fig_dir: Path,
):
    """Generate and save two pdf figures – loss & accuracy curves."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    epochs_range = list(range(1, len(history_train_loss) + 1))
    # ---------- Loss ----------
    plt.figure()
    plt.plot(epochs_range, history_train_loss, label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("NLL Loss")
    plt.legend()
    loss_name = f"training_loss_{cfg.name}.pdf"
    plt.savefig(fig_dir / loss_name, bbox_inches="tight")
    plt.close()

    # ---------- Accuracy ----------
    plt.figure()
    plt.plot(epochs_range, history_val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    acc_name = f"accuracy_{cfg.name}.pdf"
    plt.savefig(fig_dir / acc_name, bbox_inches="tight")
    plt.close()

    return [loss_name, acc_name]
