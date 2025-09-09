"""src/train.py
Model architectures and training utilities for the GRADE-GNN study.
The file only collects functionality that is strictly related to the
forward / training logic so that the remaining pipeline pieces
(pre-processing, evaluation, orchestration) can live in their dedicated
modules.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv

# ---------------------------------------------------------------------------
#  Back-bones
# ---------------------------------------------------------------------------


class GCNBackbone(nn.Module):
    """Vanilla GCN with an arbitrary number of hidden layers."""

    def __init__(self, in_channels: int, hidden: int, out_channels: int, layers: int):
        super().__init__()
        self.layers = layers
        self.convs = nn.ModuleList()
        # first
        self.convs.append(GCNConv(in_channels, hidden))
        # hidden
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        # last
        self.convs.append(GCNConv(hidden, out_channels))
        self.dropout = 0.5

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return x


class GradeGNN(nn.Module):
    """GCN backbone augmented with the GRADE-GNN modules (§ ‘New Method’)."""

    def __init__(
        self,
        data,
        kappa: Tensor,
        hidden: int = 256,
        layers: int = 64,
        lambda_geo: float = 1e-3,
        lambda_grad: float = 1e-3,
        tau_init: float = 0.5,
    ):
        super().__init__()
        self.lambda_geo = lambda_geo
        self.lambda_grad = lambda_grad
        self.tau = nn.Parameter(torch.tensor(tau_init))

        self.register_buffer("kappa", kappa)
        self.register_buffer("edge_mask", torch.ones_like(kappa))

        out_channels = int(data.y.max()) + 1 if data.y.dim() == 1 else data.y.size(-1)
        self.backbone = GCNBackbone(
            in_channels=data.num_features,
            hidden=hidden,
            out_channels=out_channels,
            layers=layers,
        )
        # Learnable gates θₑ  – initialised to zero (sigmoid → 0.5)
        self.theta_e = nn.Parameter(torch.zeros_like(kappa))

    # ---------------------------------------------------------------------
    #  forward + auxiliary losses
    # ---------------------------------------------------------------------

    def forward(self, x: Tensor, edge_index: Tensor):
        g_e = torch.sigmoid(self.theta_e) * self.edge_mask  # (E,)
        out = self.backbone(x, edge_index, edge_weight=g_e)
        return out

    def extra_losses(self, grads_on_edges: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        """Return (L_geo, L_grad) as described in the paper."""
        l_geo = self.lambda_geo * torch.mean(self.kappa * torch.sigmoid(self.theta_e))
        if grads_on_edges is not None:
            l_grad = self.lambda_grad * torch.mean((1.0 - grads_on_edges) ** 2)
        else:
            l_grad = torch.tensor(0.0, device=self.theta_e.device)
        return l_geo, l_grad


# ---------------------------------------------------------------------------
#  Train one split  (used by src.main)
# ---------------------------------------------------------------------------

def run_one_split(
    *,
    data,
    model_type: str,
    hyper: Dict,
    dataset_name: str,
    method_name: str,
    compute_or_load_curvature,
    split_id: int = 0,
    seed: int = 0,
):
    """Complete routine: model creation → training → early-stop evaluation."""

    # local import to avoid circular dependency
    from .preprocess import set_deterministic  # type: ignore

    set_deterministic(seed)
    kappa = compute_or_load_curvature(data, dataset_name)

    # ------------- create model ------------------------------------------------
    if model_type == "gcn2":
        layers = 2
    elif model_type == "gcn32":
        layers = 32
    else:
        layers = 64

    if method_name.startswith("grade"):
        model = GradeGNN(
            data,
            kappa=kappa,
            hidden=hyper.get("hidden", 256),
            layers=layers,
            lambda_geo=0.0 if "grad_only" in method_name else hyper["lambda_geo"],
            lambda_grad=hyper["lambda_grad"],
            tau_init=hyper["tau_init"],
        )
    else:
        model = GCNBackbone(
            in_channels=data.num_features,
            hidden=hyper.get("hidden", 256),
            out_channels=int(data.y.max()) + 1 if data.y.dim() == 1 else data.y.size(-1),
            layers=layers,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    # fallback split masks (datasets like Cora already include masks)
    if not hasattr(data, "train_mask"):
        n = data.num_nodes
        perm = torch.randperm(n, device=device)
        data.train_mask = torch.zeros(n, dtype=torch.bool, device=device)
        data.val_mask = torch.zeros(n, dtype=torch.bool, device=device)
        data.test_mask = torch.zeros(n, dtype=torch.bool, device=device)
        data.train_mask[perm[: int(0.6 * n)]] = True
        data.val_mask[perm[int(0.6 * n) : int(0.8 * n)]] = True
        data.test_mask[perm[int(0.8 * n) :]] = True

    optimiser = torch.optim.Adam(
        model.parameters(), lr=hyper["lr"], weight_decay=hyper["weight_decay"]
    )

    best_val = -math.inf
    best_test = -math.inf
    patience, waited = 200, 0
    t0 = time.time()
    epoch_times = []

    for epoch in range(1, 1001):
        t_epoch = time.time()
        model.train()
        optimiser.zero_grad()
        out = model(data.x, data.edge_index)
        loss_main = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        grads_on_edges = None
        if isinstance(model, GradeGNN):
            if epoch % 10 == 0:
                loss_main.backward(retain_graph=True)
                with torch.no_grad():
                    grads_on_edges = model.theta_e.grad.detach().abs()
                optimiser.zero_grad()
            l_geo, l_grad = model.extra_losses(grads_on_edges)
            loss = loss_main + l_geo + l_grad
        else:
            loss = loss_main

        loss.backward()
        optimiser.step()
        epoch_times.append(time.time() - t_epoch)

        # ---------------- evaluation / early stop ----------------------
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                acc_val = accuracy(logits[data.val_mask], data.y[data.val_mask])
                acc_test = accuracy(logits[data.test_mask], data.y[data.test_mask])
            if acc_val > best_val:
                best_val, best_test = acc_val, acc_test
                waited = 0
            else:
                waited += 1
            if waited > patience:
                break

    wall = time.time() - t0
    output = {
        "dataset": dataset_name,
        "method": method_name,
        "backbone": model_type,
        "seed": seed,
        "split": split_id,
        "val_acc": best_val,
        "test_acc": best_test,
        "wall_clock": wall,
        "secs_per_epoch": float(np.mean(epoch_times) if epoch_times else 0.0),
    }
    return output


# ---------------------------------------------------------------------------
#  Lightweight utility (kept here to avoid another cross-import)
# ---------------------------------------------------------------------------

def accuracy(logits: Tensor, y: Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == y).sum().item() / y.size(0))
