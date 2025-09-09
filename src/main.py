# main.py
"""Entry-point that orchestrates the whole experimental workflow.

USAGE
-----
python -m main   # (the repo root is expected to be on PYTHONPATH)
"""

from __future__ import annotations

import sys
import pathlib
import glob
import importlib

# Optional torch_sparse kernel check – run only if the package is available.
try:
    import torch_sparse  # noqa: F401 – optional runtime dependency for PyG

    try:
        torch_sparse.matmul.dense_sparse(
            torch_sparse.SparseTensor.eye(1), torch_sparse.SparseTensor.eye(1)
        )
    except Exception:
        sys.exit("torch-sparse kernels missing – please rebuild before training.")
except ModuleNotFoundError:
    print("[WARN] torch-sparse not found – certain PyG ops may be slower.")

import yaml

from preprocess import get_dataset
from train import MetaGCN, train_one
from utils import set_seed  # local util – see utils/__init__.py below

###############################################################################
#                                 CONFIG                                     #
###############################################################################

CONFIG_PATH = pathlib.Path("config/config.yaml")
if not CONFIG_PATH.exists():
    sys.exit("Config file config/config.yaml not found – aborting.")

with open(CONFIG_PATH) as f:
    raw_cfg = yaml.safe_load(f)


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


cfg = DotDict(raw_cfg)

###############################################################################
#                           RUN THE EXPERIMENT GRID                           #
###############################################################################

for seed in cfg.seeds:
    set_seed(seed)
    for ds_name in cfg.datasets:
        dataset = get_dataset(ds_name, root=f"data/{ds_name}")
        data = dataset[0]
        for backbone in cfg.backbones:
            depth = cfg.depth[str(backbone)]
            for variant in cfg.variants:
                exp_name = f"{cfg.name}_{ds_name}_{backbone}{depth}_{variant}_seed{seed}"
                cfg.name = exp_name  # inject dynamic name for downstream logs
                model = MetaGCN(
                    dataset.num_features,
                    dataset.num_classes,
                    depth,
                    cfg,
                    variant="meta" if variant.lower() == "meta-mpnn" else variant.lower(),
                )
                train_one(model, data, cfg)
