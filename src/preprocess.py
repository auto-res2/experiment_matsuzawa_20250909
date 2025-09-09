"""src/preprocess.py
Dataset loading, curvature computation/caching, reproducibility helpers.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

# ----------------------------------------------------------------------------
#  Determinism helpers
# ----------------------------------------------------------------------------


def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------------
#  Data & geometry
# ----------------------------------------------------------------------------

def load_dataset(name: str) -> Data:
    """Download (if needed) and return a PyG `Data` object.

    Aborts execution immediately when the dataset cannot be fetched to
    honour the *NO-FALLBACK* principle.
    """

    try:
        if name in {"Cora", "Citeseer", "Pubmed"}:
            from torch_geometric.datasets import Planetoid

            ds = Planetoid(root=f"data/{name}", name=name)
        elif name == "CoauthorCS":
            from torch_geometric.datasets import Coauthor

            ds = Coauthor(root=f"data/{name}", name="CS")
        elif name in {"Texas", "Wisconsin"}:
            from torch_geometric.datasets import WebKB

            ds = WebKB(root=f"data/{name}", name=name)
        elif name == "Chameleon":
            from torch_geometric.datasets import WikipediaNetwork

            ds = WikipediaNetwork(root=f"data/{name}", name=name, geom_gcn_split=True)
        elif name.startswith("ogbn"):
            from ogb.nodeproppred import PygNodePropPredDataset

            ds = PygNodePropPredDataset(name=name, root=f"data/{name}")
        else:
            raise ValueError(f"Unknown dataset: {name}")

        data = ds[0]
        # feature normalisation
        if data.x is None:
            raise RuntimeError("Dataset has no node features – cannot proceed.")
        mean = data.x.mean(dim=0, keepdim=True)
        std = data.x.std(dim=0, keepdim=True) + 1e-12
        data.x = (data.x - mean) / std

        data.edge_index, _ = add_self_loops(data.edge_index)
        return data
    except Exception as e:  # pylint: disable=broad-except
        raise RuntimeError(
            f"Failed to fetch dataset '{name}'. Original error: {e}\n"
            "Execution terminated to comply with NO-FALLBACK rule."
        ) from e


def compute_or_load_curvature(data: Data, name: str, paths: Dict) -> torch.Tensor:  # type: ignore
    """Return vector κ(e) aligned with `data.edge_index`; cached on disk."""
    cache_file = Path(paths["curvature"]) / f"{name}_kappa.pt"
    if cache_file.exists():
        return torch.load(cache_file)

    print(f"[Curvature] Computing Ollivier-Ricci for {name} – this will take a while …")
    try:
        import networkx as nx
        from GraphRicciCurvature.OllivierRicci import OllivierRicci

        g = nx.Graph()
        ei = data.edge_index.cpu().numpy()
        g.add_edges_from(ei.T)
        orc = OllivierRicci(g, alpha=0, verbose="ERROR")
        orc.compute_ricci_curvature()
        k_dict = nx.get_edge_attributes(orc.G, "ricciCurvature")
        kappa = [k_dict.get((u, v), k_dict.get((v, u), 0.0)) for u, v in ei.T]
        kappa = torch.tensor(kappa, dtype=torch.float32)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(kappa, cache_file)
        return kappa
    except Exception as e:  # pylint: disable=broad-except
        raise RuntimeError(
            "Failed to compute Ollivier-Ricci curvature; ensure the "
            "'GraphRicciCurvature' package is installed and the graph fits "
            "into memory.  Aborting per NO-FALLBACK rule."
        ) from e
