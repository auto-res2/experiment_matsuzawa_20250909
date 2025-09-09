"""Data loading, preprocessing, reproducibility helpers."""
from __future__ import annotations

import pathlib
import random
from typing import Dict

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Planetoid,
    Coauthor,
    WebKB,
    WikipediaNetwork,
)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected

# ---------------------------------------------------------------------------
#  Directory structure -------------------------------------------------------
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent

# Mandatory paths enforced by the evaluation harness ------------------------
# UPDATED: iteration6 as per the latest specification -----------------------
_RESEARCH_ROOT = ROOT / ".research" / "iteration6"  # .research/iteration6
FIG_DIR = _RESEARCH_ROOT / "images"                  # .research/iteration6/images
RES_DIR = _RESEARCH_ROOT                             # .research/iteration6/
DATA_DIR = ROOT / "data"

# Make sure all directories exist ------------------------------------------
for _d in (FIG_DIR, RES_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  Reproducibility -----------------------------------------------------------
# ---------------------------------------------------------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
#  Dataset loader ------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_dataset(name: str):
    """Return an InMemoryDataset from torch_geometric."""
    if name in ("Cora", "Citeseer", "Pubmed"):
        return Planetoid(root=DATA_DIR / name, name=name, transform=NormalizeFeatures())
    if name == "CoauthorCS":
        return Coauthor(root=DATA_DIR / name, name="CS", transform=NormalizeFeatures())
    if name in ("Texas", "Wisconsin"):
        return WebKB(root=DATA_DIR / name, name=name, transform=NormalizeFeatures())
    if name == "Chameleon":
        return WikipediaNetwork(root=DATA_DIR / name, name="chameleon", transform=NormalizeFeatures())
    raise RuntimeError(f"Dataset {name} not supported – STRICT FILE CONSTRAINT VIOLATED")


# ---------------------------------------------------------------------------
#  Curvature computation -----------------------------------------------------
# ---------------------------------------------------------------------------

def graph_curvature(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute Ollivier–Ricci curvature per edge (CPU)."""
    try:
        import GraphRicciCurvature as grc
    except ImportError:
        # Lazy, one-time install if missing.
        import subprocess, sys

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "GraphRicciCurvature==0.5.3.2",
                "--quiet",
            ]
        )
        import GraphRicciCurvature as grc

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    ei = edge_index.cpu().numpy()
    g.add_edges_from(ei.T)
    orc = grc.OllivierRicci(g, alpha=0.5, verbose="ERROR")
    orc.compute_ricci_curvature()
    curvature = [d.get("ricciCurvature", 0.0) for _, _, d in g.edges(data=True)]
    return torch.tensor(curvature, dtype=torch.float)