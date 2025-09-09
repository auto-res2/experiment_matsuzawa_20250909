"""src/train.py
=================
Model and training utilities for the Meta-MPNN experiments.
All heavy lifting (models, optimisation, experiment loops) lives here.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.utils import scatter_sum, scatter_std
from torch_geometric.nn import GCNConv
import matplotlib
matplotlib.use("Agg")  # head-less image backend
import matplotlib.pyplot as plt

from .evaluate import accuracy, effective_rank, grad_slope
from .preprocess import load_dataset, generate_masks

# --------------------------------------------------------------------------------------
#  Reproducibility
# --------------------------------------------------------------------------------------
SEEDS = [11, 29, 97]

def set_seed(seed: int):
    """Force deterministic behaviour."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------------------
#  Meta-MPNN Layer
# --------------------------------------------------------------------------------------
class MetaLayer(nn.Module):
    """One message-passing layer equipped with learnable node / edge gates."""

    def __init__(self, in_dim: int, out_dim: int, ctrl_hidden: int, tau: float):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.nei_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.ctrl = nn.Sequential(
            nn.Linear(in_dim + 4, ctrl_hidden), nn.ReLU(), nn.Linear(ctrl_hidden, 2)
        )
        self.tau = tau

    # ------------------------------------------------------------------
    # straight-through Gumbel-sigmoid
    # ------------------------------------------------------------------
    def _gumbel_sigmoid(self, logits: torch.Tensor, hard: bool = True):
        eps1 = -torch.empty_like(logits).exponential_().log()
        eps2 = -torch.empty_like(logits).exponential_().log()
        y = torch.sigmoid((logits + eps1 - eps2) / self.tau)
        if hard:
            y_hard = (y > 0.5).float()
            y = (y_hard - y).detach() + y
        return y

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        grad_sig: torch.Tensor,
        struc_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            var2 = scatter_std(x[edge_index[0]], edge_index[0], dim=0)
        z = torch.cat([x, var2, grad_sig, struc_feat], dim=1)
        alpha_logits, p_logits = self.ctrl(z).split(1, dim=1)
        alpha = torch.sigmoid(alpha_logits)
        p = self._gumbel_sigmoid(p_logits)
        msg = self.nei_lin(x)[edge_index[0]] * p
        out = alpha * self.self_lin(x) + (1 - alpha) * scatter_sum(msg, edge_index[1], dim=0)
        return out, alpha, p

# --------------------------------------------------------------------------------------
#  GCN backbone (optionally equipped with Meta-layers)
# --------------------------------------------------------------------------------------
class GCNNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        depth: int,
        use_meta: bool,
        ctrl_hidden: int,
        tau: float,
    ):
        super().__init__()
        self.depth = depth
        self.use_meta = use_meta
        self.layers = nn.ModuleList()
        for l in range(depth):
            if use_meta:
                self.layers.append(
                    MetaLayer(in_dim if l == 0 else hidden, hidden, ctrl_hidden, tau)
                )
            else:
                self.layers.append(
                    GCNConv(in_dim if l == 0 else hidden, hidden, cached=False, normalize=True)
                )
        self.cls = nn.Linear(hidden, out_dim)

    # ------------------------------------------------------------------
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        grad_sig = getattr(data, "grad_sig", torch.zeros_like(x[:, :1]))
        struc_feat = getattr(data, "struc_feat", torch.zeros_like(x[:, :1]))
        alphas, ps = [], []
        for layer in self.layers:
            if self.use_meta:
                x, alpha, p = layer(x, edge_index, grad_sig, struc_feat)
                alphas.append(alpha)
                ps.append(p)
            else:
                x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        out = self.cls(x)
        return out, alphas, ps, x

# --------------------------------------------------------------------------------------
#  Trainer class (supports Experiment-1 depth scaling)
# --------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        requested = cfg["device"].lower()
        self.device = torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")

    # ------------------------------------------------------------------
    def _train_single_split(
        self,
        data,
        model: nn.Module,
        mask_dict: Dict[str, torch.Tensor],
        epochs: int,
        patience: int,
    ) -> Dict[str, Any]:
        optimiser = AdamW(model.parameters(), **self.cfg["optim"])
        loss_fn = nn.CrossEntropyLoss()
        best_val, best_state, bad_counter = 0.0, None, 0
        grad_history: List[float] = []
        for epoch in range(epochs):
            model.train()
            optimiser.zero_grad()
            logits, _, _, _ = model(data)
            loss = loss_fn(logits[mask_dict["train"], :], data.y[mask_dict["train"]])
            loss.backward()
            # per-layer grad norms
            grad_history.append(
                np.mean([p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None])
            )
            optimiser.step()
            # ---------------- eval ----------------
            model.eval()
            with torch.no_grad():
                logits, _, _, emb = model(data)
            val_acc = accuracy(logits[mask_dict["val"]], data.y[mask_dict["val"]])
            if val_acc > best_val:
                best_val, best_state = val_acc, {k: v.cpu() for k, v in model.state_dict().items()}
                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter >= patience:
                break
        # -------------------------------- best model --------------------------------
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits, _, _, emb = model(data)
        test_acc = accuracy(logits[mask_dict["test"]], data.y[mask_dict["test"]])
        return {
            "best_val": best_val,
            "test_acc": test_acc,
            "effective_rank": effective_rank(emb.cpu()),
            "grad_slope": grad_slope(grad_history),
            "epochs_ran": epoch + 1,
        }

    # ------------------------------------------------------------------
    #  Experiment-1 : depth scaling stress test
    # ------------------------------------------------------------------
    def run_exp1(self):
        exp_cfg = self.cfg["experiment1"]
        research_dir = Path(".research/iteration2")
        img_dir = research_dir / "images"
        research_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(exist_ok=True)
        all_results: Dict[str, Any] = {}

        for dname in exp_cfg["datasets"]:
            ds = load_dataset(dname)
            data = ds[0].to(self.device)
            if data.y.dim() > 1:
                data.y = data.y.squeeze()
            mask_dict = generate_masks(data.num_nodes, device=self.device)
            for depth in exp_cfg["depths"]:
                for variant in exp_cfg["variants"]:
                    key = f"{dname}_{variant}_{depth}"
                    seed_res = []
                    for sd in SEEDS:
                        set_seed(sd)
                        model = GCNNet(
                            ds.num_features,
                            exp_cfg["hidden"],
                            ds.num_classes,
                            depth,
                            use_meta=(variant == "meta"),
                            ctrl_hidden=self.cfg["controller"]["hidden"],
                            tau=self.cfg["controller"]["tau"],
                        ).to(self.device)
                        start = time.time()
                        metrics = self._train_single_split(
                            data,
                            model,
                            mask_dict,
                            epochs=self.cfg["epochs"],
                            patience=self.cfg["patience"],
                        )
                        metrics["train_time"] = time.time() - start
                        seed_res.append(metrics)
                    # aggregate mean/std over seeds
                    agg = {k: float(np.mean([m[k] for m in seed_res])) for k in seed_res[0].keys()}
                    for k in seed_res[0].keys():
                        agg[f"{k}_std"] = float(np.std([m[k] for m in seed_res]))
                    all_results[key] = agg
            # ------------- plot (meta vs vanilla) -------------
            depths = exp_cfg["depths"]
            for variant in exp_cfg["variants"]:
                ys = [all_results[f"{dname}_{variant}_{d}"]["test_acc"] for d in depths]
                plt.plot(depths, ys, marker="o", label=variant)
                for (x, y) in zip(depths, ys):
                    plt.annotate(f"{y:.2f}", (x, y))
            plt.xlabel("Depth")
            plt.ylabel("Accuracy")
            plt.title(f"{dname} depth scaling")
            plt.legend()
            plt.grid(True)
            plt.savefig(img_dir / f"accuracy_{dname.lower()}.pdf", bbox_inches="tight")
            plt.close()

        # ---------------- write json & stdout ----------------
        out_file = Path(".research/iteration2/exp1_depth_scaling.json")
        with open(out_file, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print("DEPTH-SCALING STRESS-TEST (Experiment 1)")
        print(json.dumps(all_results, indent=2))
        print("Generated figures (stored in .research/iteration2/images):")
        for dname in exp_cfg["datasets"]:
            print(f"accuracy_{dname.lower()}.pdf")
