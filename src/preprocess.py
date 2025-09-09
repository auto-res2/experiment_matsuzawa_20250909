"""src/preprocess.py
Data loading and reproducibility helpers.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from torch_geometric.datasets import (
        Planetoid,
        Coauthor,
        WebKB,
        WikipediaNetwork,
    )
    from ogb.nodeproppred import PygNodePropPredDataset
except Exception as e:
    print("[FATAL] Required dataset packages missing – aborting (STRICT NO-FALLBACK)")
    import sys

    sys.exit(1)

__all__ = ["load_dataset", "set_seed"]

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================
# Global seed for reproducibility
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset loader (full-batch only for this demo)
# ============================================================

def load_dataset(name: str):
    name_l = name.lower()
    root = DATA_DIR / name_l
    if name_l in {"cora", "citeseer", "pubmed"}:
        return Planetoid(root=str(root), name=name.capitalize())
    if name_l == "coauthorcs":
        return Coauthor(root=str(root), name="CS")
    if name_l in {"texas", "wisconsin"}:
        return WebKB(root=str(root), name=name.capitalize())
    if name_l == "chameleon":
        return WikipediaNetwork(root=str(root), name="chameleon")
    if name_l == "ogbn-arxiv":
        return PygNodePropPredDataset(name="ogbn-arxiv", root=str(root))
    if name_l == "ogbn-products":
        return PygNodePropPredDataset(name="ogbn-products", root=str(root))

    print(f"[FATAL] Dataset {name} not recognised/implemented.")
    import sys

    sys.exit(1)