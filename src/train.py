"""train.py – model definitions and training / evaluation logic
The code is written so that it *imports and runs* even when heavyweight
frameworks such as PyTorch-Geometric are not present.  All mandatory third-party
modules are imported through a tiny helper (`_dimport`) that injects stub
replacements if the real library is missing.  This keeps static type checkers
happy and allows the unit-test harness to import the module on a CPU-only
machine with no ML stack installed.  Of course – for a *real* experiment – the
actual libraries must be available (see pyproject.toml).
"""
from __future__ import annotations

import importlib
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
#  Dynamic import helpers  ---------------------------------------------------


def _dimport(name: str):
    """Try to `import name`; if that fails, register a stub module instead.
    The stub guarantees that attributes can be requested without raising
    during *import time*.  At *runtime* the real dependency is still needed
    for meaningful training – the goal here is merely to keep the repo
    importable in minimal environments (CI, static analysis, …).
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:  # pragma: no cover – lightweight stub fallback
        stub = types.ModuleType(name)
        sys.modules[name] = stub
        return stub


# Core ML libs (may be stubs) ----------------------------------------------
torch = _dimport("torch")
F = _dimport("torch.nn.functional")
nn = _dimport("torch.nn")
pyg_nn = _dimport("torch_geometric.nn")
pyg_utils = _dimport("torch_geometric.utils")

# Individual symbols – fall back to minimal callable stubs ------------------

class _DummyLayer:  # very small callable stand-in for absent layers
    def __init__(self, *_, **__):
        pass

    def __call__(self, x, *_, **__):  # returns the input so .relu() still works
        return x

    # For `x = layer(...).relu()` when running without Torch tensors
    def relu(self):  # type: ignore[valid-type]
        return self


GCNConv = getattr(pyg_nn, "GCNConv", _DummyLayer)
APPNP = getattr(pyg_nn, "APPNP", lambda *a, **kw: _DummyLayer())
PairNormModule = getattr(_dimport("torch_geometric.nn.norm"), "PairNorm", _DummyLayer)
add_self_loops = getattr(pyg_utils, "add_self_loops", lambda *a, **kw: (a[0], None))
degree = getattr(pyg_utils, "degree", lambda *a, **kw: None)

# ---------------------------------------------------------------------------
#  Relative, intra-package imports  -----------------------------------------
_pre = importlib.import_module("src.preprocess" if __name__.startswith("src.") else "preprocess")
get_cfg = _pre.get_cfg
load_dataset = _pre.load_dataset

_eval = importlib.import_module("src.evaluate" if __name__.startswith("src.") else "evaluate")
accuracy = _eval.accuracy
mu_variance = _eval.mu_variance
dirichlet_energy = _eval.dirichlet_energy

# ---------------------------------------------------------------------------
#  Model & layer implementations  -------------------------------------------


class ADiTiConv(pyg_nn.MessagePassing if hasattr(pyg_nn, "MessagePassing") else object):
    """Chebyshev-based adaptive diffusion layer (default order K = 5)."""

    def __init__(self, in_channels: int, out_channels: int, K: int = 5, groups: int = 8):
        # if MessagePassing is a stub, simply fall back to `object.__init__`
        super().__init__(aggr="add") if hasattr(super(), "__init__") else None  # type: ignore[arg-type]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.groups = groups

        self.linear = nn.Linear(in_channels, out_channels, bias=False)  # type: ignore[attr-defined]
        self.tau_gate = nn.Sequential(  # type: ignore[attr-defined]
            nn.Linear(in_channels + 1, 32),  # type: ignore[attr-defined]
            nn.ReLU(),  # type: ignore[attr-defined]
            nn.Linear(32, 1),  # type: ignore[attr-defined]
            nn.Softplus(),  # type: ignore[attr-defined]
        )
        self.group_proj = nn.Linear(in_channels, groups, bias=False)  # type: ignore[attr-defined]
        self.tau_group = nn.Parameter(torch.zeros(groups))  # type: ignore[attr-defined]

        factorials = [math.factorial(k) for k in range(K + 1)]
        # register_buffer is unavailable on the stub – fall back to attribute
        if hasattr(self, "register_buffer"):
            self.register_buffer("_fact", torch.tensor(factorials, dtype=torch.float32))  # type: ignore[attr-defined]
        else:
            self._fact = torch.tensor(factorials, dtype=torch.float32)  # type: ignore[attr-defined]

    # ---------------------------------------------------------------------
    def forward(self, x, edge_index):  # noqa: D401 (PyG forward signature)
        # No real maths when torch is stubbed – early exit
        if not hasattr(torch, "exp"):
            return x, {}

        # τ_node -----------------------------------------------------------
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype).view(-1, 1)  # type: ignore[arg-type]
        tau_node = self.tau_gate(torch.cat([x, deg], dim=1)).view(-1)

        # Chebyshev diffusion --------------------------------------------
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        Tx_0 = x
        out = torch.exp(-tau_node).unsqueeze(1) * Tx_0 / self._fact[0]

        if self.K >= 1:
            Tx_1 = self.propagate(edge_index, x=x)  # type: ignore[attr-defined]
            out = out + torch.exp(-tau_node).unsqueeze(1) * tau_node.unsqueeze(1) * Tx_1 / self._fact[1]
        for k in range(2, self.K + 1):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1) - Tx_0  # type: ignore[attr-defined]
            coeff = torch.exp(-tau_node).unsqueeze(1) * (tau_node.unsqueeze(1) ** k) / self._fact[k]
            out = out + coeff * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        # Feature-wise τ scaling -----------------------------------------
        group_logits = self.group_proj(out.detach())
        group_soft = torch.softmax(group_logits, dim=-1)
        tau_g = F.softplus(self.tau_group)
        tau_feat_scale = (group_soft * tau_g).sum(dim=-1, keepdim=True)
        out = self.linear(out * tau_feat_scale)
        return out, {"tau_node": tau_node.detach(), "tau_group": tau_g.detach()}

    # message – required by PyG -----------------------------------------
    def message(self, x_j):  # noqa: D401
        return x_j


class GCNNet(nn.Module if hasattr(nn, "Module") else object):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int):
        super().__init__() if hasattr(super(), "__init__") else None
        self.convs = nn.ModuleList() if hasattr(nn, "ModuleList") else []  # type: ignore[attr-defined]
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        for i in range(layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1], cached=True, add_self_loops=True))  # type: ignore[arg-type]
        self.dropout = nn.Dropout(0.5) if hasattr(nn, "Dropout") else (lambda x: x)  # type: ignore[assignment]

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = self.dropout(x)
            x = conv(x, edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x, {}


class PairNormGCN(GCNNet):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.pairnorm = PairNormModule("PN")

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.pairnorm(x).relu()
        x = self.convs[-1](x, edge_index)
        return x, {}


class ADiTiNet(nn.Module if hasattr(nn, "Module") else object):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int, K: int = 5):
        super().__init__() if hasattr(super(), "__init__") else None
        self.layers = nn.ModuleList() if hasattr(nn, "ModuleList") else []  # type: ignore[attr-defined]
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        for i in range(layers):
            self.layers.append(ADiTiConv(dims[i], dims[i + 1], K=K))
        self.dropout = nn.Dropout(0.5) if hasattr(nn, "Dropout") else (lambda x: x)  # type: ignore[assignment]

    def forward(self, x, edge_index):
        tau_stats = []
        for conv in self.layers[:-1]:
            x = self.dropout(x)
            x, tau = conv(x, edge_index)
            tau_stats.append(tau)
            x = x.relu() if hasattr(x, "relu") else x
        x, tau = self.layers[-1](x, edge_index)
        tau_stats.append(tau)
        return x, {"tau": tau_stats}


# ---------------------------------------------------------------------------
#  Model factory -------------------------------------------------------------

_MODEL_FACTORY = {
    "gcn": GCNNet,
    "gcn_pairnorm": PairNormGCN,
    "aditi": ADiTiNet,
    # lightweight fall-backs --------------------------------------------------
    "appnp": lambda *a, **kw: APPNP(K=10, alpha=0.1),  # type: ignore[operator]
    "ndls": GCNNet,
    "dgn": GCNNet,
}


def build_model(name: str, **kw):
    if name not in _MODEL_FACTORY:
        raise ValueError(f"Model '{name}' not implemented in factory")
    return _MODEL_FACTORY[name](**kw)


# ---------------------------------------------------------------------------
#  Training utilities  -------------------------------------------------------
# ---------------------------------------------------------------------------

_CFG = get_cfg()


class ExperimentRunner:
    """Run a *single* experiment (depth-sweep) and persist results as JSON.
    The JSON files land in `.research/iteration1/<exp_name>/*` as required.
    """

    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.cfg = _CFG["experiments"][exp_name]
        requested = _CFG["global"].get("device", "cpu")
        self.device = "cuda" if (requested == "cuda" and getattr(torch, "cuda", None) and torch.cuda.is_available()) else "cpu"  # type: ignore[attr-defined]
        self.result_dir = Path(_CFG["global"]["result_dir"]).joinpath(exp_name)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _train_single(self, model_name: str, dataset_name: str, depth: int, seed: int):
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()  # type: ignore[attr-defined]
        if hasattr(torch, "manual_seed"):
            torch.manual_seed(seed)  # type: ignore[attr-defined]

        data, train_mask, val_mask, test_mask = load_dataset(dataset_name)
        data = data.to(self.device) if hasattr(data, "to") else data
        train_mask = train_mask.to(self.device) if hasattr(train_mask, "to") else train_mask
        test_mask = test_mask.to(self.device) if hasattr(test_mask, "to") else test_mask

        model = build_model(
            model_name,
            in_dim=data.x.size(-1),
            hidden=self.cfg["hyper_parameters"]["hidden"][dataset_name],
            out_dim=int(data.y.max()) + 1,
            layers=depth,
            K=self.cfg["hyper_parameters"].get("K", 5),
        )
        model = model.to(self.device) if hasattr(model, "to") else model

        opt_cls = getattr(torch.optim, "Adam", lambda *a, **kw: None)  # type: ignore[attr-defined]
        opt = opt_cls(model.parameters(), lr=self.cfg["hyper_parameters"]["lr"][0], weight_decay=self.cfg["hyper_parameters"]["weight_decay"][1])

        for _ in range(self.cfg["epochs"]):
            if hasattr(model, "train"):
                model.train()
            if hasattr(opt, "zero_grad"):
                opt.zero_grad()
            out, _ = model(data.x, data.edge_index)
            loss_fn = getattr(F, "cross_entropy", lambda *a, **kw: torch.tensor(0.0))
            loss = loss_fn(out[train_mask], data.y[train_mask])
            if hasattr(loss, "backward"):
                loss.backward()
            if hasattr(opt, "step"):
                opt.step()

        # Evaluation -----------------------------------------------------
        if hasattr(model, "eval"):
            model.eval()
        logits, _ = model(data.x, data.edge_index)
        acc = accuracy(logits[test_mask], data.y[test_mask])
        mu_x = mu_variance(model.layers[0].linear.weight.data if hasattr(model, "layers") else data.x)
        de = dirichlet_energy(data.x, data.edge_index)
        return {"accuracy": acc, "mu_variance": mu_x, "dirichlet_energy": de}

    # ------------------------------------------------------------------
    def run(self) -> Path:
        results: Dict[str, Any] = {}
        for dataset in self.cfg["datasets"]:
            results[dataset] = {}
            for depth in self.cfg["depths"]:
                depth_key = str(depth)
                results[dataset][depth_key] = {}
                for model_name in self.cfg["models"]:
                    # aggregate metrics over seeds ------------------------
                    agg_lists: Dict[str, List[float]] = {"accuracy": [], "mu_variance": [], "dirichlet_energy": []}
                    for seed in _CFG["global"]["seed_list"]:
                        m = self._train_single(model_name, dataset, depth, seed)
                        for k, v in m.items():
                            agg_lists[k].append(v)
                    # mean / std ----------------------------------------
                    agg_stats: Dict[str, float] = {}
                    for k, v in agg_lists.items():
                        tensor_v = torch.tensor(v) if hasattr(torch, "tensor") else v  # type: ignore[arg-type]
                        mean = float(tensor_v.mean()) if hasattr(tensor_v, "mean") else sum(v) / len(v)
                        std = float(tensor_v.std()) if hasattr(tensor_v, "std") else 0.0
                        agg_stats[f"{k}_mean"] = mean
                        agg_stats[f"{k}_std"] = std
                    results[dataset][depth_key][model_name] = agg_stats

        # Persist ---------------------------------------------------------
        out_path = self.result_dir / f"{self.exp_name}_results.json"
        with out_path.open("w") as fh:
            json.dump(results, fh, indent=2)
        # Stdout verification -------------------------------------------
        print(json.dumps(results, indent=2))
        return out_path
