"""Entry-point – run all experiments from YAML config.
Usage:  python -m src.main   (requires Python ≥3.8)"""

import random
import sys
from pathlib import Path

import torch
import yaml

# ---------------------------------------------------------------
#  Project-local imports (relative)
# ---------------------------------------------------------------
from .train import CLTrainer
from .preprocess import split_cifar100, permuted_mnist

# ---------------------------------------------------------------
#  Load configuration
# ---------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError("Configuration file missing: " + str(CONFIG_PATH))

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Ensure mandatory dirs exist
Path("data").mkdir(exist_ok=True)

# ---------------------------------------------------------------
#  Helper – run a single experiment entry
# ---------------------------------------------------------------

def _run_experiment(cfg: dict, exp_name: str):
    random.seed(cfg["seeds"][0])
    torch.manual_seed(cfg["seeds"][0])

    # obtain stream generator
    if cfg["dataset_loader"] == "split_cifar100":
        stream = split_cifar100("data")
    elif cfg["dataset_loader"] == "permuted_mnist":
        stream = permuted_mnist("data", n_tasks=cfg.get("n_tasks", 20))
    else:
        raise ValueError("Unsupported dataset loader: " + cfg["dataset_loader"])

    trainer = CLTrainer(cfg)
    trainer.run_stream(stream, exp_name)


# ---------------------------------------------------------------
#  Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    try:
        _run_experiment(CONFIG["experiment_1"], "experiment1")
        _run_experiment(CONFIG["experiment_2"], "experiment2")
    except KeyboardInterrupt:
        print("Interrupted by user – exiting.")
        sys.exit(130)
    except Exception as e:
        print("Runtime failure:", e)
        sys.exit(1)
