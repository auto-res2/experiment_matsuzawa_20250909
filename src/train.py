"""src/train.py – model definitions and training loop for AdaSmooth-ODE experiments"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GCN2Conv  # base layers

# ---------------------------------------------------------------------------
# APPNP import – the exact location changed across PyG versions --------------
# ---------------------------------------------------------------------------
# In some releases (e.g. 2.4.0) the ``APPNP`` class resides inside
# ``torch_geometric.nn``, while in older/newer releases it is re-exported from
# ``torch_geometric.nn.models``.  To remain compatible with both we try the
# new path first and fall back to the old one.  If neither works we raise a
# *hard* error in accordance with the NO-Fallback policy.
# ---------------------------------------------------------------------------

try:
    from torch_geometric.nn import APPNP as _APPNP
except ImportError:  # pragma: no cover – triggered on very old wheels
    try:
        from torch_geometric.nn.models import APPNP as _APPNP
    except ImportError as e:  # noqa: F841
        _APPNP = None  # handled later

from torch_geometric.data import Data

import torch_sparse  # required for Laplacian matmul

try:
    from torch_geometric.nn import DGNConv  # optional, newer PyG installs
except ImportError:  # pragma: no cover – old wheels
    DGNConv = None  # handled later

from .preprocess import DEVICE, get_normalised_laplacian
from .evaluate import accuracy, effective_rank

# ---------------------------------------------------------------------------
#                           AdaSmooth-ODE LAYER
# ---------------------------------------------------------------------------

class AdaSmoothODELayer(nn.Module):
    """Polynomial approximation of exp(−τ_i L) with node–adaptive time τ_i."""

    def __init__(self, in_dim: int, out_dim: int, poly_order: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.gate = nn.Linear(in_dim, 1, bias=True)  # predicts positive τ_i
        self.poly_order = poly_order

    @torch.cuda.amp.autocast(enabled=True)
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ):
        tau = F.softplus(self.gate(x)).clamp_max(5.0)  # regularise very large values early-on
        L = get_normalised_laplacian(edge_index, edge_weight, x.size(0))

        out = x
        x_l = x  # cache for power iterations
        for k in range(1, self.poly_order + 1):
            x_l = torch_sparse.matmul(L, x_l)
            coeff = ((-tau) ** k) / math.factorial(k)
            out = out + coeff * x_l
        return self.linear(out)


# ---------------------------------------------------------------------------
#                           AdaSmooth-ODE MODEL
# ---------------------------------------------------------------------------

class AdaSmoothODE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        poly_order: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.layer1 = AdaSmoothODELayer(in_dim, hidden_dim, poly_order)
        self.layer2 = AdaSmoothODELayer(hidden_dim, out_dim, poly_order)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x, edge_index)
        return x


# ---------------------------------------------------------------------------
#                               BASELINES
# ---------------------------------------------------------------------------

def make_baseline(model_name: str, in_dim: int, hidden: int, out_dim: int, num_layers: int):
    model_name = model_name.lower()

    if model_name == "gcn":
        layers = nn.ModuleList([])
        layers.append(GCNConv(in_dim, hidden, cached=True, normalize=True))
        for _ in range(num_layers - 2):
            layers.append(GCNConv(hidden, hidden, cached=True, normalize=True))
        layers.append(GCNConv(hidden, out_dim, cached=True, normalize=True))

        class _GCN(nn.Module):
            def __init__(self, convs):
                super().__init__()
                self.convs = convs

            def forward(self, data: Data):
                x, edge_index = data.x, data.edge_index
                for conv in self.convs[:-1]:
                    x = conv(x, edge_index)
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
                return self.convs[-1](x, edge_index)

        return _GCN(layers)

    if model_name == "appnp":
        if _APPNP is None:
            raise RuntimeError("APPNP class not found in current PyG install – aborting as per NO-Fallback policy.")
        # Signature: (in_channels, hidden_channels, out_channels, K=10, alpha=0.1, dropout=0.5)
        return _APPNP(in_dim, hidden, out_dim, K=10, alpha=0.1, dropout=0.5)

    if model_name == "gcnii":
        class _GCNII(nn.Module):
            def __init__(self, in_dim_: int, hidden_dim_: int, out_dim_: int, num_layers_: int):
                super().__init__()
                self.convs = nn.ModuleList(
                    [GCN2Conv(hidden_dim_, alpha=0.5, theta=1.0, layer=i + 1) for i in range(num_layers_)]
                )
                self.lin1 = nn.Linear(in_dim_, hidden_dim_)
                self.lin2 = nn.Linear(hidden_dim_, out_dim_)

            def forward(self, data: Data):
                x, edge_index = data.x, data.edge_index
                x = F.dropout(x, p=0.5, training=self.training)
                h0 = F.relu(self.lin1(x))
                x = h0
                for conv in self.convs:
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = F.relu(conv(x, h0, edge_index))
                return self.lin2(x)

        return _GCNII(in_dim, hidden, out_dim, num_layers)

    if model_name == "dgn":
        if DGNConv is None:
            raise RuntimeError("DGNConv not available – aborting as per NO-Fallback policy.")

        class _DGN(nn.Module):
            def __init__(self, in_dim_: int, hidden_dim_: int, out_dim_: int, num_layers_: int):
                super().__init__()
                self.convs = nn.ModuleList()
                self.convs.append(DGNConv(in_dim_, hidden_dim_, num_groups=4))
                for _ in range(num_layers_ - 2):
                    self.convs.append(DGNConv(hidden_dim_, hidden_dim_, num_groups=4))
                self.convs.append(DGNConv(hidden_dim_, out_dim_, num_groups=4))

            def forward(self, data: Data):
                x, edge_index = data.x, data.edge_index
                for conv in self.convs[:-1]:
                    x = conv(x, edge_index)
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
                return self.convs[-1](x, edge_index)

        return _DGN(in_dim, hidden, out_dim, num_layers)

    raise ValueError(f"Unknown baseline '{model_name}'.")


# ---------------------------------------------------------------------------
#                           TRAINING WRAPPER
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model_name: str  # "adasmooth" or baseline name
    depth: int  # used only for baselines; −1 for AdaSmooth
    hidden_dim: int = 64
    poly_order: int = 10
    dropout: float = 0.5
    lr: float | str = 0.01  # allow scientific notation strings from YAML
    weight_decay: float | str = 5e-4
    epochs: int = 2000
    patience: int = 100
    seed: int = 42

    # YAML parses numbers like "5e-4" as strings.  Cast them proactively so the
    # optimiser receives proper floats and we avoid "<' not supported" errors.
    def __post_init__(self):
        object.__setattr__(self, "lr", float(self.lr))
        object.__setattr__(self, "weight_decay", float(self.weight_decay))


class Trainer:
    """Lightweight wrapper that encapsulates optimisation logic and early-stopping."""

    def __init__(self, data: Data, config: TrainConfig):
        torch.backends.cudnn.benchmark = True
        self.data = data.to(DEVICE)
        self.config = config

        in_dim = data.num_node_features
        out_dim = int(data.y.max().item() + 1)

        if config.model_name == "adasmooth":
            self.model = AdaSmoothODE(
                in_dim,
                config.hidden_dim,
                out_dim,
                poly_order=config.poly_order,
                dropout=config.dropout,
            )
        else:
            self.model = make_baseline(config.model_name, in_dim, config.hidden_dim, out_dim, config.depth)

        self.model = self.model.to(DEVICE)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ------------------------- optimisation ------------------------------ #

    def train(self) -> None:
        data = self.data
        best_val = 1e9
        best_state = None
        no_improve = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = self.model(data)
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                self.model.eval()
                val_logits = self.model(data)
                val_loss = F.cross_entropy(val_logits[data.val_mask], data.y[data.val_mask]).item()

            if val_loss < best_val:
                best_val, no_improve = val_loss, 0
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1

            if no_improve >= self.config.patience:
                break

        # restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ------------------------ evaluation helpers ------------------------- #

    def test(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data)
            acc = accuracy(logits[self.data.test_mask], self.data.y[self.data.test_mask])
            erank = effective_rank(logits[self.data.test_mask])
        return acc, erank, logits.detach().cpu()
