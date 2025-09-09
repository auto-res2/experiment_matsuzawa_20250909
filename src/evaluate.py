import json
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .preprocess import FIG_DIR

# ---------------------------------------------------------------------------
#  Evaluation utilities ------------------------------------------------------
# ---------------------------------------------------------------------------

def test(model, data, split_idx: Dict[str, torch.Tensor]):
    """Return dict of accuracies and the raw logits."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1)
    accs = {}
    for k, idx in split_idx.items():
        accs[k] = (preds[idx] == data.y[idx]).float().mean().item()
    return accs, logits


# ---------------------------------------------------------------------------
#  Plot helpers --------------------------------------------------------------
# ---------------------------------------------------------------------------

def plot_training_loss(loss_history, tag: str):
    xs = list(range(1, len(loss_history) + 1))
    plt.figure()
    plt.plot(xs, loss_history, label="train_loss")
    for x, y in zip(xs, loss_history):
        if x % (max(len(xs) // 10, 1)) == 0:
            plt.annotate(f"{y:.2f}", (x, y))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = FIG_DIR / f"training_loss_{tag}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return fname