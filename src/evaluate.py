"""src/evaluate.py
===================
Collection of utility functions for evaluation / statistics.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import List

# --------------------------------------------------------------------------------------
@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Micro-accuracy for classification."""
    return (logits.argmax(dim=-1) == y).float().mean().item()

# --------------------------------------------------------------------------------------
#  Effective rank (Shannon effective rank of singular values)
# --------------------------------------------------------------------------------------
def effective_rank(X: torch.Tensor, thresh: float = 0.9) -> float:
    u, s, _ = torch.linalg.svd(X, full_matrices=False)
    s = s.cpu().numpy()
    cumsum = np.cumsum(s)
    return float(np.searchsorted(cumsum, thresh * cumsum[-1]) + 1)

# --------------------------------------------------------------------------------------
#  Gradient slope (log-scale decay across depth)
# --------------------------------------------------------------------------------------

def grad_slope(grad_norms: List[float]) -> float:
    y = np.log10(np.asarray(grad_norms) + 1e-12)
    x = np.arange(len(y))
    slope, *_ = np.polyfit(x, y, 1)
    return float(slope)
