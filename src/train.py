import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ---------------------------------------------------------------------------
#  Edge gating and node-wise halting modules (GRADE-GNN building blocks)
# ---------------------------------------------------------------------------
class EdgeGate(nn.Module):
    """Differentiable edge gates with curvature & sparsity losses."""

    def __init__(self, edge_index: torch.Tensor, kappa: torch.Tensor, num_nodes: int):
        super().__init__()
        self.register_buffer("edge_index", edge_index)  # [2, E]
        self.register_buffer("kappa", kappa)            # [E]
        self.theta = nn.Parameter(torch.zeros(edge_index.size(1)))  # learnable logits
        self.num_nodes = num_nodes

    # ------------------------------------------------------------------
    def forward(self):
        """Return (edge_index, edge_weights) for the current gates."""
        g = torch.sigmoid(self.theta)                   # (0,1)
        return self.edge_index, g

    # ------------------------------------------------------------------
    def l_geo(self, lambda_geo: float):
        g = torch.sigmoid(self.theta)
        return lambda_geo * (self.kappa * g).mean()

    # ------------------------------------------------------------------
    def sparsity(self, coef: float = 1e-4):
        return coef * torch.sigmoid(self.theta).mean()


class NodeDepthController(nn.Module):
    """Per-node adaptive halting (1-layer MLP mapping hᵢ→σ)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, h: torch.Tensor, tau: float):
        d = torch.sigmoid(self.fc(h)).squeeze(-1)       # [N]
        mask = (d >= tau).float().view(-1, 1)           # 1 keep, 0 stop
        return h * mask, mask.mean()


# ---------------------------------------------------------------------------
#  Baseline & GRADE-GNN models
# ---------------------------------------------------------------------------
class VanillaGCN(nn.Module):
    """Standard GCN with variable depth."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, out_dim))
        self.dropout = dropout

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GradeGCN(nn.Module):
    """GCN + GRADE-GNN modules (edge-gates & node-depth controller)."""

    def __init__(
        self,
        data,
        hidden: int,
        layers: int,
        dropout: float,
        lambda_geo: float,
        tau: float,
        graph_curvature_fn,
    ):
        super().__init__()
        from .preprocess import graph_curvature  # late import to avoid circularity

        in_dim, out_dim = data.num_features, int(data.y.max().item()) + 1
        # --- curvature & gates ------------------------------------------------
        kappa = graph_curvature_fn(data.edge_index, data.num_nodes)
        self.edge_gate = EdgeGate(data.edge_index, kappa, data.num_nodes)
        self.depth_ctl = NodeDepthController(hidden)
        # --- GCN backbone -----------------------------------------------------
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden, cached=True, normalize=True))
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden, hidden, cached=True, normalize=True))
        self.convs.append(GCNConv(hidden, out_dim, cached=True, normalize=True))

        self.dropout = dropout
        self.lambda_geo = lambda_geo
        self.tau = tau

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        edge_index, g = self.edge_gate()
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, g)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, _ = self.depth_ctl(x, self.tau)
        return self.convs[-1](x, edge_index, g)

    # ------------------------------------------------------------------
    def aux_loss(self):
        return self.edge_gate.l_geo(self.lambda_geo) + self.edge_gate.sparsity()


# ---------------------------------------------------------------------------
#  One-epoch training helper -------------------------------------------------
# ---------------------------------------------------------------------------

def train(model: nn.Module, data, train_idx: torch.Tensor, optimiser):
    model.train()
    optimiser.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    if hasattr(model, "aux_loss"):
        loss = loss + model.aux_loss()
    loss.backward()
    optimiser.step()
    return loss.item()
