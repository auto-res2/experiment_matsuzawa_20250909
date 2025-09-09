import argparse
import itertools
import json
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW

from .train import build_backbone, Projector, train_one_epoch
from .evaluate import evaluate
from .preprocess import get_dataloader

###############################################################################
# Configuration loader (PyYAML is lightweight but powerful)
###############################################################################

import yaml


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def load_cfg(path: str = "config/config.yaml") -> DotDict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return DotDict(cfg)


###############################################################################
# Utility – fix the random seed for reproducibility
###############################################################################

def set_seed(seed: int):
    import random, numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


###############################################################################
# Main experiment runner
###############################################################################

def run_experiment(exp_name: str, exp_cfg: DotDict, global_cfg: DotDict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for seed in global_cfg.global["seed_list"]:
        set_seed(seed)
        for model_name in exp_cfg["models"]:
            print(f"===== {exp_name} | {model_name} | seed {seed} =====")

            # ------------------------------------------------------------------
            # Data
            # ------------------------------------------------------------------
            train_loader = get_dataloader(
                exp_cfg["dataset"],
                "train",
                batch_size=exp_cfg["batch_size"],
            )
            val_loader = get_dataloader(
                exp_cfg["dataset"],
                "validation",
                batch_size=exp_cfg["batch_size"],
            )

            num_classes = len(train_loader.dataset.features["label"].names)

            # ------------------------------------------------------------------
            # Model & optimiser
            # ------------------------------------------------------------------
            backbone = build_backbone(model_name, num_classes=num_classes).to(device)

            # Determine feature dimension safely
            if hasattr(backbone, "num_features"):
                feat_dim = backbone.num_features
            else:
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 224, 224).to(device)
                    feats = backbone.forward_features(dummy)
                    feat_dim = feats.shape[-1] if feats.ndim == 2 else feats.shape[1]

            projector = Projector(feat_dim).to(device)

            optimiser = AdamW(
                itertools.chain(backbone.parameters(), projector.parameters()),
                lr=exp_cfg["optimiser"]["lr"],
                weight_decay=exp_cfg["optimiser"]["weight_decay"],
            )
            scaler = GradScaler()

            # ------------------------------------------------------------------
            # Training loop
            # ------------------------------------------------------------------
            val_top1 = []
            for epoch in range(exp_cfg["epochs"]):
                train_one_epoch(
                    backbone,
                    projector,
                    train_loader,
                    optimiser,
                    exp_cfg["dcd"],
                    epoch,
                    scaler,
                    device,
                )
                if (epoch + 1) % 5 == 0:
                    metrics = evaluate(
                        backbone,
                        val_loader,
                        device,
                        out_json=Path(".research/iteration2")
                        / f"{exp_name}_{model_name}_seed{seed}_e{epoch+1}.json",
                    )
                    val_top1.append(metrics["top1"])

            results[f"{model_name}_seed{seed}"] = val_top1[-1] if val_top1 else None

    # ----------------------------------------------------------------------
    # Persist per-experiment summary
    # ----------------------------------------------------------------------
    out_dir = Path(".research/iteration2")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{exp_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


###############################################################################
# CLI entry-point
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="DCD experiments orchestrator")
    parser.add_argument("--exp", default="exp1_diff_waterbirds", help="Experiment name from YAML file")
    args = parser.parse_args()

    cfg = load_cfg()

    if args.exp not in cfg["experiments"]:
        raise ValueError(f"Experiment {args.exp} not found in config file.")

    run_experiment(args.exp, cfg["experiments"][args.exp], cfg)


if __name__ == "__main__":
    main()
