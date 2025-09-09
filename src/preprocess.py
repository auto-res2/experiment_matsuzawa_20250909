from __future__ import annotations

"""src/preprocess.py – utility helpers (I/O, datasets, Laplacian, randomness)
This revision updates the research output directories in compliance with the
project-wide specification:

  • All image artefacts must be saved inside  ``.research/iteration6/images``
  • All JSON artefacts must live directly in ``.research/iteration6``

It also fixes an incorrect import for the *Texas* and *Wisconsin* datasets by
using the generic ``WebKB`` wrapper provided by PyG.  No other functional
changes were introduced.
"""

import json
import random
from pathlib import Path
from typing import Dict

import torch
import torch_sparse
import yaml
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.transforms import ToUndirected
from ogb.nodeproppred import PygNodePropPredDataset

# ---------------------------------------------------------------------------
#                           DIRECTORY CONSTANTS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# NOTE: mandatory path update (specification requirement) -------------------
RESEARCH_DIR = BASE_DIR / ".research" / "iteration6"  # <-- UPDATED to iteration6
IMAGE_DIR = RESEARCH_DIR / "images"
RESULT_DIR = RESEARCH_DIR  # JSON files live directly in iteration6/
# ---------------------------------------------------------------------------

DATA_DIR = BASE_DIR / "data"

for _p in [IMAGE_DIR, RESULT_DIR, DATA_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
#                               MISC I/O
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_yaml(obj: Dict, fp: Path):
    with open(fp, "w") as f:
        yaml.safe_dump(obj, f)


def save_json(obj: Dict, fp: Path):
    with open(fp, "w") as f:
        json.dump(obj, f, indent=2)

# ---------------------------------------------------------------------------
#                               LAPLACIAN
# ---------------------------------------------------------------------------

def get_normalised_laplacian(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
    num_nodes: int | None = None,
):
    """Return symmetric normalised Laplacian  L = I − D^{−1/2} A D^{−1/2}."""
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    row, col = edge_index
    deg = torch_sparse.sum(edge_weight, row, dim=0, dtype=edge_weight.dtype, num_nodes=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    adj = torch_sparse.SparseTensor(row=col, col=row, value=norm, sparse_sizes=(num_nodes, num_nodes))
    I = torch_sparse.SparseTensor.eye(num_nodes, device=edge_index.device)
    return I - adj

# ---------------------------------------------------------------------------
#                               DATASETS
# ---------------------------------------------------------------------------

_DATASET_MAP = {
    "cora": lambda: Planetoid(DATA_DIR.as_posix(), "Cora", transform=ToUndirected()),
    "citeseer": lambda: Planetoid(DATA_DIR.as_posix(), "Citeseer", transform=ToUndirected()),
    "pubmed": lambda: Planetoid(DATA_DIR.as_posix(), "Pubmed", transform=ToUndirected()),
    # WebKB variants --------------------------------------------------------
    "texas": lambda: WebKB(DATA_DIR.as_posix(), name="Texas", transform=ToUndirected()),
    "wisconsin": lambda: WebKB(DATA_DIR.as_posix(), name="Wisconsin", transform=ToUndirected()),
}


def load_dataset(name: str):
    name = name.lower()
    if name in _DATASET_MAP:
        return _DATASET_MAP[name]()[0]

    if name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset("ogbn-arxiv", root=DATA_DIR)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[split_idx["train"]] = True
        data.val_mask = torch.zeros_like(data.train_mask)
        data.val_mask[split_idx["valid"]] = True
        data.test_mask = torch.zeros_like(data.train_mask)
        data.test_mask[split_idx["test"]] = True
        return data

    raise RuntimeError(
        f"Dataset '{name}' is not supported – aborting as per NO-Fallback policy."
    )
