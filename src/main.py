"""
Main entry-point (src/main.py).  Orchestrates experiments via relative imports.
Usage (as before):
    EXP=1 python -m src.main
"""
from __future__ import annotations

import os, sys, yaml
from pathlib import Path

from .evaluate import run_experiment_1, run_experiment_2, run_experiment_3


# --------------------------------------------------------------
# 1.  Locate / create default YAML config
# --------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_DIR.mkdir(exist_ok=True)
CFG_PATH = CONFIG_DIR / "config.yaml"

_DEFAULT_CONFIG = {
    "hardware": {"num_gpus": 8, "gpu_type": "A100-80GB", "ram": "2048GB"},
    "datasets": {
        "cifar100": "https://huggingface.co/datasets/uoft-cs/cifar100",
        "tiny_imagenet": "https://huggingface.co/datasets/israfelsr/mm_tiny_imagenet",
        "ag_news": "https://huggingface.co/datasets/fancyzhx/ag_news",
    },
    "models": {
        "resnet18": "glasses/resnet18",
        "bert_base": "google-bert/bert-base-uncased",
    },
    "hyperparams": {
        "batch_size_vision": 128,
        "batch_size_text": 64,
        "epochs_per_task": 50,
        "optimizer": {
            "sgd": {"lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4},
            "adam": {"lr": 1e-3, "betas": [0.9, 0.999]},
        },
        "scheduler": {"type": "cosine", "warmup_iters": 500},
    },
    "budgets_mb": [0.25, 0.5, 1, 2, 4, 8],
}

if not CFG_PATH.exists():
    with open(CFG_PATH, "w") as fp:
        yaml.safe_dump(_DEFAULT_CONFIG, fp)


# --------------------------------------------------------------
# 2.  Main
# --------------------------------------------------------------

def main():
    with open(CFG_PATH) as fp:
        config = yaml.safe_load(fp)

    exp = os.environ.get("EXP", "1")
    if exp == "1":
        run_experiment_1(config)
    elif exp == "2":
        run_experiment_2(config)
    elif exp == "3":
        run_experiment_3(config)
    else:
        print("Unknown experiment id; use EXP=1|2|3")
        sys.exit(1)


if __name__ == "__main__":
    main()
