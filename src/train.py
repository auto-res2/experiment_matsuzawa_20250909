# train.py
"""Model architectures and training routines extracted from the original monolithic
script.  All heavy lifting (forward-prop, back-prop, checkpointing) happens here."""

from __future__ import annotations

import time
import pathlib
import json
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from torch_geometric.utils import to_dense_adj  # noqa: F401 – retained for future use
from torch_geometric.loader import NeighborLoader  # noqa: F401 – retained for future use

###############################################################################
#                                    MODEL                                   #
###############################################################################


class Controller(nn.Module):
    """Two-layer MLP that outputs logits for the self-gate α and edge gate p."""

    def __init__(self, in_dim: int, cfg_controller):
        super().__init__()
        self.tau = cfg_controller.tau
        self.lambda_kl = cfg_controller.lambda_kl
        self.h1 = nn.Linear(in_dim, cfg_controller.hidden)
        self.h2 = nn.Linear(cfg_controller.hidden, 2)  # α  and  p logits

    # ---------------------------------------------------------------------
    def forward(self, ctrl_in: torch.Tensor) -> torch.Tensor:  # (N, in_dim)
        h = F.relu(self.h1(ctrl_in))
        logits = self.h2(h)
        return logits  # [:,0] = α ;  [:,1] = p


class MetaLayer(nn.Module):
    """One message-passing layer with on-the-fly gating (Meta-MPNN)."""

    def __init__(self, in_dim: int, out_dim: int, cfg_controller):
        super().__init__()
        # ctrl input = [x , var(x) , deg , grad_norm]  =>  in_dim + 3
        self.ctrl_in_dim = in_dim + 3
        self.ctrl = Controller(self.ctrl_in_dim, cfg_controller)
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        deg: torch.Tensor,
        grad_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ctrl input = concat[x , var(x_N2) , deg , grad]
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        ctrl_in = torch.cat([x, var, deg, grad_norm], dim=1)
        logits = self.ctrl(ctrl_in)
        alpha = torch.sigmoid(logits[:, 0:1])
        # message passing – neighbourhood aggregation
        row, col = edge_index
        p_edge = torch.sigmoid(logits[row, 1:2])  # broadcast node→edge
        aggr = torch.zeros_like(x)
        aggr.index_add_(0, row, p_edge * x[col])
        h = F.relu(alpha * self.lin_self(x) + (1.0 - alpha) * self.lin_neigh(aggr))
        return h, alpha.detach(), p_edge.detach()


class MetaGCN(nn.Module):
    """GCN/PairNorm/Meta variants (depth configurable)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        cfg_exp,  # full experiment cfg so we can access controller sub-cfg
        variant: str = "vanilla",
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv, PairNorm  # local import to avoid circular-dep

        self.depth = depth
        layers = []
        if variant == "vanilla":
            for _ in range(depth):
                layers.append(GCNConv(in_dim if not layers else out_dim, out_dim, cached=True))
        elif variant == "pairnorm":
            for _ in range(depth):
                layers.append(
                    nn.Sequential(
                        GCNConv(in_dim if not layers else out_dim, out_dim, cached=True),
                        PairNorm("scale"),
                    )
                )
        elif variant == "meta":
            for _ in range(depth):
                layers.append(MetaLayer(in_dim if not layers else out_dim, out_dim, cfg_exp.controller))
        else:
            raise ValueError(f"Unknown variant '{variant}'")
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(out_dim, out_dim)

    # ------------------------------------------------------------------
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #   degree feature ----------------------------------------------------
        deg = torch.log1p(
            torch.bincount(edge_index[0], minlength=x.size(0)).float()
        ).unsqueeze(1).to(x.device)
        grad_stub = torch.zeros_like(deg)  # placeholder – will be updated via hooks

        for layer in self.layers:
            if isinstance(layer, MetaLayer):
                x, _, _ = layer(x, edge_index, deg, grad_stub)
            else:
                x = layer(x, edge_index)
        return self.classifier(x)

###############################################################################
#                               TRAINING LOOP                                #
###############################################################################

from evaluate import evaluate_model, line_plot  # noqa: E402 – after definition
from utils import dump_json  # noqa: E402 – local util


# ---------------------------- CONSTANT PATHS ------------------------------
_ITER_DIR = pathlib.Path(".research/iteration22")
_IMG_DIR = _ITER_DIR / "images"


def _select_device(pref: Optional[str] = None) -> str:
    """Return a valid device string.  Falls back to CPU if CUDA is unavailable."""

    if pref is not None and pref.lower().startswith("cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref or ("cuda" if torch.cuda.is_available() else "cpu")


def train_one(model: nn.Module, data, cfg_exp, device: Optional[str] = None) -> None:
    """Train a *single* (dataset, backbone, variant, seed) configuration."""

    device = _select_device(device)
    model = model.to(device)
    data = data.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg_exp.optim.lr, weight_decay=cfg_exp.optim.weight_decay
    )

    best_val: float = 0.0
    patience_ctr: int = cfg_exp.optim.patience

    history: Dict[str, list] = {"val_acc": [], "train_loss": []}
    start = time.perf_counter()

    # ------------------------------------------------------------------
    for epoch in range(cfg_exp.optim.epochs):
        model.train()
        opt.zero_grad()
        out = model(data)
        loss = cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        val_acc = evaluate_model(model, data, split="val")
        history["val_acc"].append(val_acc)
        history["train_loss"].append(loss.item())

        if val_acc > best_val:
            best_val = val_acc
            patience_ctr = cfg_exp.optim.patience
            torch.save(model.state_dict(), "/tmp/best_meta_mpn.pt")
        else:
            patience_ctr -= 1
            if patience_ctr == 0:
                break

    wall = time.perf_counter() - start

    # ------------------------------ testing ---------------------------------
    model.load_state_dict(torch.load("/tmp/best_meta_mpn.pt"))
    test_acc = evaluate_model(model, data, split="test")

    # -------------------------- visualisations ------------------------------
    _IMG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_file = _IMG_DIR / f"{cfg_exp.name}.pdf"
    line_plot(
        list(range(len(history["train_loss"]))),
        history["train_loss"],
        title=f"Train loss – {cfg_exp.name}",
        xlabel="epoch",
        ylabel="loss",
        pdf_file=str(pdf_file),
    )

    # ------------------------------ logging ---------------------------------
    result: Dict[str, Any] = {
        "test_acc": test_acc,
        "best_val": best_val,
        "wall_clock_s": wall,
        "epochs_run": len(history["train_loss"]),
        "figure": str(pdf_file),
    }

    # ----------- persist JSON & echo to stdout for verification -------------
    _ITER_DIR.mkdir(parents=True, exist_ok=True)
    out_json = _ITER_DIR / f"{cfg_exp.name}.json"
    dump_json(result, out_json)

    print("\n===== EXPERIMENT:", cfg_exp.name, "=====")
    print(json.dumps(result, indent=2))
    print("Figure saved:", pdf_file)
