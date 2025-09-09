"""preprocess.py – configuration helpers & dataset loading.
The actual heavyweight ML libraries (PyTorch, PyG, …) are imported lazily so
that the repository can still be imported when these libs are absent.  For real
training runs the user must install the dependencies listed in *pyproject.toml*
(or provide their own environment).
"""
from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Tuple

import yaml  # PyYAML *is* listed as a hard dependency

# ---------------------------------------------------------------------------
#  Configuration utilities ---------------------------------------------------
_CFG_PATH = Path("config/config.yaml")
if not _CFG_PATH.exists():
    raise FileNotFoundError("Configuration file not found – expected at 'config/config.yaml'")

with _CFG_PATH.open() as fh:
    _CFG = yaml.safe_load(fh)


def get_cfg():
    """Return a deep copy of the full experiment configuration."""
    return copy.deepcopy(_CFG)


# ---------------------------------------------------------------------------
#  Lazy import helper for heavyweight libs -----------------------------------

def _dimport(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:  # pragma: no cover – stub fallback
        import types, sys
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod


# ---------------------------------------------------------------------------
#  Dataset loading -----------------------------------------------------------
_ROOT = Path(get_cfg()["global"]["data_root"]).expanduser()
_ROOT.mkdir(parents=True, exist_ok=True)


def _fail(msg: str):
    raise RuntimeError(f"[STRICT NO-FALLBACK] {msg}")



def load_dataset(name: str) -> Tuple["Data", "Tensor", "Tensor", "Tensor"]:  # type: ignore[name-defined]
    """Return (data, train_mask, val_mask, test_mask) for the requested dataset.
    If the necessary libraries are not installed the function raises with a
    clear message so that users immediately know what to do.
    """
    name = name.lower()

    torch = _dimport("torch")
    pyg_datasets = _dimport("torch_geometric.datasets")
    pyg_utils = _dimport("torch_geometric.utils")

    if name == "cora":
        Planetoid = getattr(pyg_datasets, "Planetoid", None)
        if Planetoid is None:
            _fail("PyG Planetoid dataset class missing – please install torch-geometric")
        ds = Planetoid(str(_ROOT), "Cora", split="full", transform=None)
        data = ds[0]
    elif name == "chameleon":
        WikipediaNetwork = getattr(pyg_datasets, "WikipediaNetwork", None)
        if WikipediaNetwork is None:
            _fail("PyG WikipediaNetwork dataset class missing – please install torch-geometric")
        ds = WikipediaNetwork(str(_ROOT), "Chameleon", geom_gcn_preprocess=False, split="full")
        data = ds[0]
    elif name == "ogbn_arxiv":
        ogb = _dimport("ogb.nodeproppred")
        PygNodePropPredDataset = getattr(ogb, "PygNodePropPredDataset", None)
        if PygNodePropPredDataset is None:
            _fail("Package 'ogb' not installed – cannot load ogbn_arxiv")
        ds = PygNodePropPredDataset("ogbn-arxiv", root=str(_ROOT))
        data = ds[0]
        split_idx = ds.get_idx_split()
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[split_idx["train"]] = True
        val_mask = torch.zeros_like(train_mask)
        val_mask[split_idx["valid"]] = True
        test_mask = torch.zeros_like(train_mask)
        test_mask[split_idx["test"]] = True
        data.y = data.y.squeeze()
        add_self_loops = getattr(pyg_utils, "add_self_loops", lambda *a, **kw: (a[0], None))
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
        data.x = torch.nn.functional.normalize(data.x, p=2.0, dim=-1)  # type: ignore[attr-defined]
        return data, train_mask, val_mask, test_mask
    else:
        _fail(f"Unknown dataset '{name}'")

    # Random 60/20/20 split for Cora & Chameleon -----------------------------
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size : train_size + val_size]] = True
    test_mask[perm[train_size + val_size :]] = True

    add_self_loops = getattr(pyg_utils, "add_self_loops", lambda *a, **kw: (a[0], None))
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=num_nodes)
    data.x = torch.nn.functional.normalize(data.x, p=2.0, dim=-1)  # type: ignore[attr-defined]
    return data, train_mask, val_mask, test_mask
