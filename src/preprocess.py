# preprocess.py
"""Dataset fetching / preprocessing logic.  Minimal subset of the original helper
functions is retained so that the rest of the codebase can stay unchanged."""

from __future__ import annotations

import pathlib
import shutil
import sys
from typing import Any, Dict

from huggingface_hub import hf_hub_download
import torch_geometric.datasets as tgds

###############################################################################
#                               DATA DOWNLOADERS                              #
###############################################################################

ROOT = pathlib.Path("data/raw")
ROOT.mkdir(parents=True, exist_ok=True)


def _abort(msg: str):
    print("[FATAL]", msg)
    sys.exit(1)


def _hf_planetoid(repo: str, root: str):
    """Planetoid replica hosted on HF – downloads the .pt files or aborts."""

    try:
        local_path = hf_hub_download(repo_id=repo, filename="dataset.pt", repo_type="dataset")
    except Exception as exc:
        _abort(f"Dataset {repo} not accessible – {exc}")

    dest = pathlib.Path(root) / repo.replace("/", "_")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(local_path, dest / "data.pt")

    name = repo.split("/")[-1]
    return tgds.Planetoid(root, name.capitalize())


###############################################################################
#                                REGISTRY                                     #
###############################################################################

_DATASETS: Dict[str, Any] = {}


def get_dataset(name: str, root: str = "data"):
    """Factory that mirrors the original `registry.get_dataset` function."""

    name_l = name.lower()
    if name_l in {"cora", "citeseer", "pubmed"}:
        return tgds.Planetoid(root, name.capitalize())
    if name_l in {"texas", "cornell", "chameleon", "squirrel"}:
        return tgds.WikipediaNetwork(root, name_l, geom_gcn_split="fixed")
    if name_l == "ogbn-arxiv":
        return tgds.OGBNArxiv(root)
    if name_l == "peptides-functional":
        return tgds.LRGBDataset(root, name)
    # fall-back to HF
    return _hf_planetoid(name, root)
