"""src/train.py
All model architectures and the training routine live here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# -----------------------------------------------------------------------------
# Dataclass based configuration ------------------------------------------------
# -----------------------------------------------------------------------------

@dataclass
class ControllerConfig:
    hidden: int = 32
    tau: float = 0.5
    kl_alpha: float = 0.1
    kl_edge: float = 0.1


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 2000
    patience: int = 100


@dataclass
class ExperimentConfig:
    name: str
    dataset_name: str
    backbone: str  # "gcn" | "gat" | "sage"
    variant: str   # "vanilla" | "baseline" | "meta"
    depth: int
    num_classes: int
    hidden_dim: int = 128
    batch_size: int = 0  # 0 => full-batch
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

# -----------------------------------------------------------------------------
#  Meta-MPNN -------------------------------------------------------------------
# -----------------------------------------------------------------------------
class MetaLayer(nn.Module):
    """One message-passing layer with node- and edge-wise gates."""

    def __init__(self, in_dim: int, out_dim: int, ctrl_cfg: ControllerConfig):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ctrl_cfg = ctrl_cfg

        # Linear maps for self / neighbour messages
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_nei = nn.Linear(in_dim, out_dim, bias=False)

        # Controller (two outputs: alpha & p)
        self.ctrl = nn.Sequential(
            nn.Linear(2 * in_dim + 2, ctrl_cfg.hidden),
            nn.ReLU(),
            nn.Linear(ctrl_cfg.hidden, 2),
        )

        self.register_forward_hook(self._shape_guard)

    @torch.no_grad()
    def _shape_guard(self, module, _inputs, _outputs):  # pylint: disable=unused-argument
        """Fail fast if the concatenated embedding has a wrong size."""
        x, *_ = _inputs  # type: ignore
        expected = 2 * self.in_dim + 2
        got = 2 * x.size(1) + 2
        assert got == expected, (
            f"Controller input wrong size: got {got}, expected {expected}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        grad_sig: torch.Tensor,
        struct_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            var_local = torch.var(x, dim=1, keepdim=True)
        z = torch.cat([x, var_local, grad_sig, struct_feat], dim=1)
        alpha_logit, p_logit = self.ctrl(z).split(1, dim=1)

        # Straight-through Gumbel-sigmoid for edge gate
        gumbel_noise = (-torch.empty_like(p_logit).exponential_()).log()
        p = torch.sigmoid((p_logit + gumbel_noise) / self.ctrl_cfg.tau)
        alpha = torch.sigmoid(alpha_logit)

        # Message passing (mean aggregate)
        msg = self.lin_nei(x)[edge_index[0]] * p
        agg = torch.zeros_like(self.lin_self(x))
        agg = agg.index_add(0, edge_index[1], msg)
        out = alpha * self.lin_self(x) + (1 - alpha) * agg
        return out, alpha, p


class MetaMPNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        depth: int,
        num_classes: int,
        ctrl_cfg: ControllerConfig,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(MetaLayer(in_dim, hidden, ctrl_cfg))
        for _ in range(depth - 2):
            self.layers.append(MetaLayer(hidden, hidden, ctrl_cfg))
        self.layers.append(MetaLayer(hidden, num_classes, ctrl_cfg))

    def forward(self, data: Data) -> torch.Tensor:  # type: ignore
        x, edge_index = data.x, data.edge_index
        grad_sig = torch.zeros((x.size(0), 1), device=x.device)
        struct_feat = torch.zeros((x.size(0), 1), device=x.device)
        for layer in self.layers:
            x, _, _ = layer(x, edge_index, grad_sig, struct_feat)
            x = F.relu(x)
        return F.log_softmax(x, dim=1)

# -----------------------------------------------------------------------------
#  Vanilla baselines -----------------------------------------------------------
# -----------------------------------------------------------------------------
class VanillaGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int, num_classes: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(depth - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, num_classes))

    def forward(self, data: Data):  # type: ignore
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

# -----------------------------------------------------------------------------
#  Model factory ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_model(cfg: ExperimentConfig, in_dim: int) -> nn.Module:
    if cfg.variant == "meta":
        return MetaMPNN(in_dim, cfg.hidden_dim, cfg.depth, cfg.num_classes, cfg.controller)
    if cfg.backbone == "gcn":
        return VanillaGCN(in_dim, cfg.hidden_dim, cfg.depth, cfg.num_classes)
    raise ValueError(
        f"Unsupported combination – backbone={cfg.backbone}, variant={cfg.variant}")

# -----------------------------------------------------------------------------
#  Training loop ----------------------------------------------------------------
# -----------------------------------------------------------------------------
from .evaluate import plot_training_curves

def train(
    model: nn.Module,
    data: Data,
    cfg: ExperimentConfig,
    device: torch.device,
    fig_dir: Path,
) -> Dict[str, float]:
    """Full training loop + evaluation. Returns metrics as a dict."""
    data = data.to(device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    criterion = nn.NLLLoss()

    if not (
        hasattr(data, "train_mask")
        and hasattr(data, "val_mask")
        and hasattr(data, "test_mask")
    ):
        raise RuntimeError(
            "Dataset must contain train/val/test masks – no fallback allowed!"
        )

    best_val_acc: float = 0.0
    best_epoch: int = -1
    patience_counter: int = 0
    history_train_loss: List[float] = []
    history_val_acc: List[float] = []

    for epoch in range(1, cfg.optim.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            val_acc = (
                (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            )
        history_train_loss.append(loss.item())
        history_val_acc.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.optim.patience:
            break

    # -------------------- final evaluation --------------------
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        test_acc = (
            (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        )

    # -------------------- visualisations ----------------------
    fig_files = plot_training_curves(
        history_train_loss, history_val_acc, cfg, fig_dir
    )

    return {
        "best_val_acc": best_val_acc,
        "test_acc_last": test_acc,
        "best_epoch": best_epoch,
        "epochs_ran": len(history_train_loss),
        "figures": fig_files,
    }
