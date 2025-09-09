"""src/main.py
Main orchestration script – loads YAML config, runs the experiment grid
and triggers evaluation/visualisation.
Run via:  python -m src.main
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import yaml

# ---------------------------------------------------------------------------
#  Load configuration --------------------------------------------------------
# ---------------------------------------------------------------------------
CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if not CFG_PATH.exists():
    sys.exit("Configuration file missing – cannot proceed.")

with open(CFG_PATH, "r", encoding="utf-8") as fp:
    CONFIG: Dict = yaml.safe_load(fp)

# ensure directories exist
for _p in CONFIG["paths"].values():
    Path(_p).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  Local imports (after sys.path set-up)
# ---------------------------------------------------------------------------
from .preprocess import compute_or_load_curvature, load_dataset  # noqa: E402
from .train import run_one_split  # noqa: E402
from .evaluate import save_results_and_plot  # noqa: E402


def main():
    if CONFIG.get("quick_mode", True):
        datasets = ["Cora"]
        backbones = {"gcn2": {"layers": 2, "hidden": 256}}
        methods = ["vanilla", "grade_full"]
        splits = [0]
        seeds = [0]
    else:
        exp_cfg = CONFIG["experiments"][0]  # full_benchmark
        datasets = exp_cfg["datasets"]
        backbones = exp_cfg["backbones"]
        methods = exp_cfg["methods"]
        splits = list(range(10))
        seeds = list(range(10))

    results: List[Dict] = []
    for dname in datasets:
        data = load_dataset(dname)
        for bname in backbones:
            hyper = {
                "lr": 0.005,
                "dropout": 0.5,
                "weight_decay": 5e-4,
                "lambda_geo": 1e-3,
                "lambda_grad": 1e-3,
                "tau_init": 0.5,
                "hidden": backbones[bname]["hidden"],
            }
            for method in methods:
                for split in splits:
                    for seed in seeds:
                        res = run_one_split(
                            data=data.clone(),
                            model_type=bname,
                            hyper=hyper,
                            dataset_name=dname,
                            method_name=method,
                            compute_or_load_curvature=lambda d, n: compute_or_load_curvature(
                                d, n, CONFIG["paths"]
                            ),
                            split_id=split,
                            seed=seed,
                        )
                        results.append(res)

    save_results_and_plot(results, CONFIG["paths"])


if __name__ == "__main__":
    main()
