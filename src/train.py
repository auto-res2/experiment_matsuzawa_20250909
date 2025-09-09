"""src/train.py
Model architectures and training utilities.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Tuple, Union

try:
    import torch_geometric
    from torch_geometric.utils import softmax
except Exception as e:  # pragma: no cover – hard-fail if PyG missing
    print("[FATAL] PyTorch-Geometric not available – aborting (STRICT NO-FALLBACK)")
    import sys
    sys.exit(1)

__all__ = [
    "GCNBackbone",
    "EdgeGate",
    "NodeDepthController",
    "GradeGCN",
    "train_one_epoch",
]


# ============================================================
# Back-bone GCN
# ============================================================

class GCNBackbone(nn.Module):
    """Vanilla GCN with residual every two layers (identical to experiment script)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int):
        super().__init__()
        from torch_geometric.nn import GCNConv, LayerNorm

        if layers < 2:
            raise ValueError("layers must be >=2")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hidden))
        self.norms.append(LayerNorm(hidden))
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
            self.norms.append(LayerNorm(hidden))
        self.convs.append(GCNConv(hidden, out_dim))

    # --------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward with residual connection every two layers.
        The residual connects the input of the *current* two-layer block to its
        output, ensuring identical dimensionality (hidden → hidden).
        """
        for l, conv in enumerate(self.convs[:-1]):
            # start a new residual block on even layers
            if l % 2 == 0:
                x_res = x  # save for residual (same hidden dim)
            x = conv(x, edge_index)
            x = F.relu(x)
            # apply residual on odd layers where dimensions match
            if l % 2 == 1:
                x = x + x_res
            x = self.norms[l](x)
        x = self.convs[-1](x, edge_index)
        return x


# ============================================================
# GRADE-GNN components
# ============================================================

class EdgeGate(nn.Module):
    """Learnable sigmoid gate per edge (simplified – curvature regulariser only)."""

    def __init__(self, edge_index: torch.Tensor, kappa: torch.Tensor):
        super().__init__()
        self.edge_index = edge_index  # [2, E]
        self.theta = nn.Parameter(torch.zeros(edge_index.size(1)))
        self.register_buffer("kappa", kappa)

    # --------------------------------------------------------
    def forward(self):
        g = torch.sigmoid(self.theta)  # (0,1)
        return self.edge_index, g

    # --------------------------------------------------------
    def geo_loss(self, lambda_geo: float) -> torch.Tensor:
        return lambda_geo * (self.kappa * self.theta).mean()


class NodeDepthController(nn.Module):
    """Node-wise halting probability controller."""

    def __init__(self, feat_dim: int, tau_init: float = 0.5):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1)
        self.tau = nn.Parameter(torch.tensor(tau_init))

    # --------------------------------------------------------
    def forward(self, h: torch.Tensor, grad_norm: torch.Tensor):  # grad_norm placeholder
        prob = torch.sigmoid(self.fc(h).squeeze())  # (N,)
        mask = (prob >= self.tau).float().unsqueeze(1)
        # Return float scalar to avoid device casting issues downstream
        return h * mask, float(mask.mean().item())


class GradeGCN(nn.Module):
    """Full GRADE-GNN wrapper around GCN back-bone."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int,
        layers: int,
        edge_index: torch.Tensor,
        kappa: torch.Tensor,
        lambda_geo: float,
        lambda_grad: float,
        tau_init: float,
    ):
        super().__init__()
        self.edge_gate = EdgeGate(edge_index, kappa)
        self.backbone = GCNBackbone(in_dim, out_dim, hidden, layers)
        # Depth controller must match the feature dimension it receives (out_dim)
        self.depth_ctl = NodeDepthController(out_dim, tau_init)
        self.lambda_geo = lambda_geo
        self.lambda_grad = lambda_grad  # retained for completeness

    # --------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, float]:
        edge_index, g = self.edge_gate()
        _ = softmax(g, edge_index[0])  # placeholder – not used in GCN op here
        x = self.backbone(x, edge_index)
        x, depth_ratio = self.depth_ctl(x, torch.tensor(0.0, device=x.device))
        return x, depth_ratio

    # --------------------------------------------------------
    def extra_loss(self) -> torch.Tensor:
        return self.edge_gate.geo_loss(self.lambda_geo)


# ============================================================
# Training loop
# ============================================================

from torch_geometric.data import Data  # after PyG availability check

def _unpack_model_output(model_out: Union[torch.Tensor, Tuple[torch.Tensor, Union[torch.Tensor, float]]]):
    """Utility that makes the training/eval code agnostic to whether the model
    returns a single tensor (logits) or a tuple (logits, aux_stat).
    The auxiliary statistic is converted to a Python float when it is a scalar
    tensor to ensure subsequent `float(aux)` calls are always safe regardless of
    device placement (CPU/GPU).
    """
    if isinstance(model_out, tuple):
        out, aux = model_out
    else:
        out = model_out
        aux = torch.tensor(0.0, device=out.device)

    # Convert 0-dim tensor -> float for safe downstream casting
    if torch.is_tensor(aux) and aux.ndim == 0:
        aux = aux.item()
    return out, aux


def train_one_epoch(
    model: nn.Module,
    data: Data,  # full-batch only in this compact demo
    optimizer: Optimizer,
    device: torch.device | str,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    data = data.to(device)
    out_raw = model(data.x, data.edge_index)
    out, depth_ratio = _unpack_model_output(out_raw)

    loss = F.cross_entropy(out[data.train_mask], data.y.squeeze()[data.train_mask])

    loss_total = loss
    # models that implement extra_loss provide attribute; fallback 0
    if hasattr(model, "extra_loss"):
        loss_total = loss_total + model.extra_loss()

    loss_total.backward()
    optimizer.step()
    return loss.item(), float(depth_ratio)
