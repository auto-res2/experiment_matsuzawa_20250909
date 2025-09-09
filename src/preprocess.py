"""src/preprocess.py
Dataset loading helpers and reproducibility utilities.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

# Root directories -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Seeding ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def set_seed(seed: int = 11):
    """Seed python, numpy and torch (both CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# -----------------------------------------------------------------------------
# Dataset factory --------------------------------------------------------------
# -----------------------------------------------------------------------------

def _load_cora() -> Data:  # type: ignore
    from torch_geometric.datasets import Planetoid

    ds = Planetoid(
        root=str(DATA_DIR / "Planetoid"),
        name="Cora",
        transform=T.NormalizeFeatures(),
    )
    return ds[0]

_DATASET_FACTORY: Dict[str, callable] = {
    "cora": _load_cora,
}


def load_dataset(dataset_name: str) -> Data:  # type: ignore
    if dataset_name not in _DATASET_FACTORY:
        raise RuntimeError(
            f"Unknown dataset id '{dataset_name}' – abort (NO FALLBACK)"
        )
    return _DATASET_FACTORY[dataset_name]()
