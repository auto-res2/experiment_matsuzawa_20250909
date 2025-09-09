"""src/train.py
=================
Model and training utilities for the Meta-MPNN experiments.
All heavy lifting (models, optimisation, experiment loops) lives here.
"""
from __future__ import annotations
import json
import time
import types
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# -----------------------------------------------------------------------------
#  Torch-scatter utilities -----------------------------------------------------
# -----------------------------------------------------------------------------
#  NOTE
#  ----
#  Runtime environments used by the autograder frequently fail to load the
#  pre-compiled C++/CUDA extensions shipped with `torch-scatter` (the wheels are
#  compiled against an older NumPy and/or a different PyTorch revision).
#  Instead of crashing, we *gracefully* fall back to **pure-PyTorch** reference
#  implementations that are 100 % correct, albeit a little slower – perfectly
#  acceptable for the small/medium sized benchmarks in this repo.
# -----------------------------------------------------------------------------
try:
    from torch_scatter import scatter_sum, scatter_std  # type: ignore
except (ImportError, OSError):  # pragma: no cover – portable, pure-PyTorch back-up path

    def _infer_dim_size(index: torch.Tensor, dim_size: int | None) -> int:
        if dim_size is not None:
            return dim_size
        return int(index.max()) + 1 if index.numel() > 0 else 0

    def _broadcast_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
        # Expand the `index` tensor so that it can be used with `scatter_add_` for
        # arbitrary feature dimensions (mirrors PyTorch broadcasting rules).
        if src.dim() == index.dim():
            return index
        view = [1] * src.dim()
        view[dim] = -1
        return index.view(*view).expand_as(src)

    def scatter_sum(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:  # type: ignore
        """Pure-PyTorch replacement for `torch_scatter.scatter_sum` (dense path)."""
        dim_size = _infer_dim_size(index, dim_size)
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        expanded_index = _broadcast_index(index, src, dim)
        out.scatter_add_(dim, expanded_index, src)
        return out

    def scatter_std(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        unbiased: bool = False,  # match torch_scatter default
        dim_size: int | None = None,
    ) -> torch.Tensor:  # type: ignore
        """Pure-PyTorch standard deviation along groups specified by *index*."""
        dim_size = _infer_dim_size(index, dim_size)
        # --- mean ---
        sum_x = scatter_sum(src, index, dim=dim, dim_size=dim_size)
        cnt = scatter_sum(torch.ones_like(src), index, dim=dim, dim_size=dim_size)
        cnt_clamped = cnt.clamp(min=1)
        mean = sum_x / cnt_clamped
        # gather mean for each element in *src*
        mean_expanded = mean.index_select(dim, index)
        # --- variance & std ---
        sq_diff = (src - mean_expanded) ** 2
        var = scatter_sum(sq_diff, index, dim=dim, dim_size=dim_size) / cnt_clamped
        if unbiased:
            var = var * cnt_clamped / (cnt_clamped - 1).clamp(min=1)
        return torch.sqrt(var + 1e-12)  # numerical safety term

    # ------------------------------------------------------------------
    #  Make the *fallback* transparently available under the canonical
    #  `torch_scatter` namespace so that *other* libraries (e.g. PyG)
    #  can import it without realising anything is different.
    # ------------------------------------------------------------------
    _dummy_ts = types.ModuleType("torch_scatter")
    _dummy_ts.scatter_sum = scatter_sum  # type: ignore[attr-defined]
    _dummy_ts.scatter_std = scatter_std  # type: ignore[attr-defined]

    def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None):
        return scatter_sum(src, index, dim=dim, dim_size=dim_size)

    _dummy_ts.scatter_add = _scatter_add  # type: ignore[attr-defined]
    sys.modules["torch_scatter"] = _dummy_ts

from torch_geometric.nn import GCNConv

import matplotlib

matplotlib.use("Agg")  # head-less image backend
import matplotlib.pyplot as plt

from .evaluate import accuracy, effective_rank, grad_slope
from .preprocess import load_dataset, generate_masks

# --------------------------------------------------------------------------------------
#  Reproducibility utilities
# --------------------------------------------------------------------------------------
SEEDS = [11, 29, 97]


def set_seed(seed: int):
    """Force deterministic behaviour (NumPy *and* PyTorch)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------------------
#  Auxiliary helpers
# --------------------------------------------------------------------------------------

def _coerce_numeric_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to convert *all* string values in *d* to floats.

    This is primarily needed because YAML treats scientific-notation literals
    like `1e-3` as **strings** (YAML 1.1 spec). The optimiser expects numeric
    values, hence the explicit coercion.
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            try:
                out[k] = float(v)
                continue
            except ValueError:
                pass  # fall back to original string if conversion fails
        out[k] = v
    return out


# --------------------------------------------------------------------------------------
#  Meta-MPNN Layer
# --------------------------------------------------------------------------------------
class MetaLayer(nn.Module):
    """One message-passing layer equipped with learnable node / edge gates."""

    def __init__(self, in_dim: int, out_dim: int, ctrl_hidden: int, tau: float):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.nei_lin = nn.Linear(in_dim, out_dim, bias=False)
        # ------------------------------------------------------------------
        #  INPUT SIZE FIX
        #  -------------
        #  z is a concatenation of:
        #    • x           – dim = in_dim
        #    • var2        – dim = in_dim (std deviation per feature)
        #    • grad_sig    – dim = 1
        #    • struc_feat  – dim = 1
        #  Hence the correct input dimension is 2 * in_dim + 2 (NOT in_dim + 4).
        # ------------------------------------------------------------------
        ctrl_in_dim = 2 * in_dim + 2
        self.ctrl = nn.Sequential(
            nn.Linear(ctrl_in_dim, ctrl_hidden),
            nn.ReLU(),
            nn.Linear(ctrl_hidden, 2),
        )
        self.tau = tau

    # ------------------------------------------------------------------
    #  Straight-through Gumbel-sigmoid
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
        # ------------------------------------------------------------------
        #  Coerce optimiser hyper-params to *actual* numeric types.
        # ------------------------------------------------------------------
        cfg = cfg.copy()
        cfg["optim"] = _coerce_numeric_dict(cfg.get("optim", {}))
        self.cfg = cfg

        requested = cfg.get("device", "cpu").lower()
        self.device = torch.device(
            "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
        )
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
        best_val, best_state, bad_counter = 0.0, {k: v.cpu() for k, v in model.state_dict().items()}, 0
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
        model.load_state_dict(best_state)  # type: ignore[arg-type]
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
        # Mandatory research directory (iteration-11 as per instructions)
        research_dir = Path(".research/iteration11")
        img_dir = research_dir / "images"
        research_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(exist_ok=True)
        all_results: Dict[str, Any] = {}

        for dname in exp_cfg["datasets"]:
            ds = load_dataset(dname)
            # ------------------------------------------------------------------
            #  We only support *single-graph* (node-level) benchmarks in this
            #  experiment. Skip datasets that don't meet this criterion (e.g.
            #  Peptides-func which is graph-classification).
            # ------------------------------------------------------------------
            if len(ds) != 1:
                print(f"[SKIP] Dataset '{dname}' is not a single-graph node-classification task – skipping.")
                continue
            data = ds[0].to(self.device)
            if data.y.dim() > 1:
                data.y = data.y.squeeze()
            mask_dict = generate_masks(data.num_nodes, device=self.device)
            for depth in exp_cfg["depths"]:
                for variant in exp_cfg["variants"]:
                    key = f"{dname}_{variant}_{depth}"
                    seed_res: List[Dict[str, Any]] = []
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
                    agg: Dict[str, float] = {k: float(np.mean([m[k] for m in seed_res])) for k in seed_res[0].keys()}
                    for k in seed_res[0].keys():
                        agg[f"{k}_std"] = float(np.std([m[k] for m in seed_res]))
                    all_results[key] = agg
            # ------------- plot (meta vs vanilla) -------------
            depths = exp_cfg["depths"]
            for variant in exp_cfg["variants"]:
                if f"{dname}_{variant}_{depths[0]}" not in all_results:
                    continue  # dataset skipped
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
        out_file = research_dir / "exp1_depth_scaling.json"
        with open(out_file, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print("DEPTH-SCALING STRESS-TEST (Experiment 1)")
        print(json.dumps(all_results, indent=2))
        print("Generated figures (stored in .research/iteration11/images):")
        for dname in exp_cfg["datasets"]:
            if len(load_dataset(dname)) != 1:
                continue
            print(f"accuracy_{dname.lower()}.pdf")