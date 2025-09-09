"""src/preprocess.py
======================
Data loading & splitting utilities.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import torch
from torch_geometric.datasets import (
    Planetoid,
    WikipediaNetwork,
    WebKB,  # WebKB provides Texas/Wisconsin/Cornell etc.
)

# The LRGB datasets (which include Peptides-func) live behind a dedicated wrapper
try:
    from torch_geometric.datasets import LRGBDataset  # PyG >= 2.3
except ImportError:  # pragma: no cover – extremely old PyG versions
    LRGBDataset = None  # type: ignore

# ogb is optional (large). Import lazily.

def _try_import_ogb(name: str, root: str):
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as e:
        raise RuntimeError("`ogb` not installed – cannot download {name}") from e
    return PygNodePropPredDataset(name=name, root=root)

# --------------------------------------------------------------------------------------
#  Dataset loader
# --------------------------------------------------------------------------------------

def load_dataset(name: str):
    """Download (if needed) and return a torch-geometric dataset."""
    root = Path("data") / name
    root.mkdir(parents=True, exist_ok=True)
    if name == "Cora":
        return Planetoid(root=str(root), name="Cora")
    if name == "Chameleon":
        return WikipediaNetwork(root=str(root), name="chameleon", geom_gcn_preprocess=False)
    if name == "PeptidesFunc":
        if LRGBDataset is None:
            raise RuntimeError("PeptidesFunc dataset requested but this PyG version does not ship LRGBDataset.")
        return LRGBDataset(root=str(root), name="Peptides-func")
    if name == "Texas":
        return WebKB(root=str(root), name="Texas")
    if name == "ogbn-arxiv":
        return _try_import_ogb("ogbn-arxiv", str(root))
    raise ValueError(f"Unsupported dataset {name}")

# --------------------------------------------------------------------------------------
#  Create boolean masks for train/val/test (60/20/20 split)
# --------------------------------------------------------------------------------------

def generate_masks(n_nodes: int, device="cpu") -> Dict[str, torch.Tensor]:
    idx = torch.randperm(n_nodes, device=device)
    n = n_nodes
    splits = {
        "train": idx[: int(0.6 * n)],
        "val": idx[int(0.6 * n) : int(0.8 * n)],
        "test": idx[int(0.8 * n) :],
    }
    masks = {k: torch.zeros(n, dtype=torch.bool, device=device) for k in splits}
    for k, v in splits.items():
        masks[k][v] = True
    return masks
