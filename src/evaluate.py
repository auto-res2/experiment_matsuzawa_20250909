"""src/evaluate.py
Evaluation, metrics and plotting utilities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Union

import torch
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    import torch_geometric
except ImportError:
    print("[FATAL] PyTorch-Geometric not available – aborting (STRICT NO-FALLBACK)")
    import sys
    sys.exit(1)

from torch_geometric.data import Data

__all__ = ["evaluate", "save_curve_pdf", "dump_json"]


def _unpack_model_output(model_out: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
    if isinstance(model_out, tuple):
        out, _ = model_out
    else:
        out = model_out
    return out


def evaluate(model: torch.nn.Module, data: Data, split: str, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        out_raw = model(data.x, data.edge_index)
        out = _unpack_model_output(out_raw)
        pred = out.argmax(dim=-1)  # shape (N,)
        if split == "val":
            mask = data.val_mask
        elif split == "test":
            mask = data.test_mask
        else:
            mask = data.train_mask
        correct = (pred[mask] == data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        return acc


# ------------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------------

def save_curve_pdf(xs: List[int], ys: List[float], title: str, ylabel: str, fname: str | Path):
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.3f}")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------
# JSON result helper
# ------------------------------------------------------------------

def dump_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
