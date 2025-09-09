"""src/main.py – entry-point that orchestrates all experiments
Run with:   python -m src.main
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import yaml
import numpy as np

from .preprocess import (RESULT_DIR, set_seed, save_json, load_dataset)
from .train import TrainConfig, Trainer
from .evaluate import plot_metric

# ---------------------------------------------------------------------------
#                       LOAD EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError("Config file not found at 'config/config.yaml'.")

with open(CONFIG_PATH) as f:
    CFG: Dict = yaml.safe_load(f)

SEEDS: List[int] = CFG["seeds"]
DATASETS: List[str] = CFG["datasets"]
BASELINES: List[Dict] = CFG["baselines"]
ADASMOOTH_POLY = int(CFG.get("adasmooth", {}).get("poly_order", 10))
TRAIN_DEFAULTS = CFG["train"]

# ---------------------------------------------------------------------------
#                               EXPERIMENT
# ---------------------------------------------------------------------------

def run_benchmark_suite():
    description = (
        "Experiment-1: Unified benchmark – AdaSmooth-ODE vs fixed-depth baselines"
    )
    print(description)

    results_summary = {}
    for ds_name in DATASETS:
        data = load_dataset(ds_name)
        ds_res = {}

        model_cfgs = BASELINES + [{"model": "adasmooth", "depth": -1}]
        for mc in model_cfgs:
            model_name, depth = mc["model"], mc["depth"]
            key = f"{model_name}{'' if depth < 0 else depth}"

            seed_metrics = []
            for seed in SEEDS:
                set_seed(seed)
                epochs = (
                    TRAIN_DEFAULTS["epochs_small"]
                    if ds_name in ["cora", "citeseer", "pubmed", "texas", "wisconsin"]
                    else TRAIN_DEFAULTS["epochs_large"]
                )
                cfg = TrainConfig(
                    model_name=model_name,
                    depth=depth,
                    hidden_dim=TRAIN_DEFAULTS["hidden_dim"],
                    poly_order=ADASMOOTH_POLY,
                    dropout=TRAIN_DEFAULTS["dropout"],
                    lr=TRAIN_DEFAULTS["lr"],
                    weight_decay=TRAIN_DEFAULTS["weight_decay"],
                    epochs=epochs,
                    patience=TRAIN_DEFAULTS["patience"],
                    seed=seed,
                )
                trainer = Trainer(data, cfg)
                trainer.train()
                acc, erank, _ = trainer.test()
                seed_metrics.append({"acc": acc, "erank": erank})

            accs = [m["acc"] for m in seed_metrics]
            eranks = [m["erank"] for m in seed_metrics]
            ds_res[key] = {
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "erank_mean": float(np.mean(eranks)),
            }

            # ­plot per-model accuracy across seeds
            plot_metric(
                list(range(len(accs))),
                accs,
                xlabel="seed idx",
                ylabel="accuracy (%)",
                title=f"{ds_name}-{key}",
                filename=f"accuracy_{ds_name}_{key}.pdf",
            )

        # ------- save per-dataset json and print to stdout -------- #
        json_path = RESULT_DIR / f"exp1_{ds_name}.json"
        save_json(ds_res, json_path)
        print(f"Results for {ds_name} :\n", json.dumps(ds_res, indent=2))
        results_summary[ds_name] = ds_res

    # global summary
    save_json(results_summary, RESULT_DIR / "exp1_summary.json")


# ---------------------------------------------------------------------------
#                               MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    run_benchmark_suite()
    print(f"Done. Total time: {(time.time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    main()
