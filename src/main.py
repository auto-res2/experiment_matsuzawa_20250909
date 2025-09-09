"""
main.py – orchestrates the full experimental suite via relative imports
Run with:  python -m src.main
"""
from __future__ import annotations

import collections, pathlib, types
from types import SimpleNamespace
from typing import Any, List

import yaml

from .preprocess import DatasetBuilder
from .train import ModelFactory, Trainer, set_seed
from .evaluate import line_plot

# ---------------------------------------------------------------------------
#  Config loading helpers
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  Dataclass-like structures implemented with SimpleNamespace so we avoid a
#  circular import between train.py and main.py.
# ---------------------------------------------------------------------------

def _dict_to_ns(d: Any) -> Any:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_ns(x) for x in d]
    return d

# ---------------------------------------------------------------------------
#  Default configuration (auto-written on first run so users have a template)
# ---------------------------------------------------------------------------
DEFAULT_YAML = {
    "experiments": [
        {
            "name": "quick_demo",
            "dataset": "cifar10",
            "method": "erm",
            "model": "resnet18",
            "seed": 17,
            "correlation": None,
            "optimiser": {
                "epochs": 2,
                "batch_size": 64,
                "lr": 0.0003,
                "weight_decay": 0.05,
                "eta_gc": None,
                "fourier_lambda": None,
            },
            "extra": {},
        }
    ]
}

if not CONFIG_PATH.exists():
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(DEFAULT_YAML, f)
    print(f"[INFO] Configuration file written to {CONFIG_PATH}. Edit it to run full experiments.")

# ---------------------------------------------------------------------------
#  Main entry-point
# ---------------------------------------------------------------------------

def main():
    with open(CONFIG_PATH) as f:
        raw_cfg = yaml.safe_load(f)

    cfg = _dict_to_ns(raw_cfg)

    aggregate_results = collections.defaultdict(list)
    for exp in cfg.experiments:
        set_seed(exp.seed)

        train_ds, val_ds, test_ds = DatasetBuilder(exp.dataset, exp.correlation, exp.seed).get()
        num_classes = len({y for _, y in train_ds}) if exp.dataset != "imagenet1k" else 1000
        model = ModelFactory.get(exp.model, num_classes=num_classes)

        trainer = Trainer(model, train_ds, val_ds, test_ds, exp)
        res = trainer.run()
        aggregate_results[exp.name].append(res)

    # ------------------------------------------------------------------
    #  Example figure: only plotted if quick_demo is swapped for full exp1
    # ------------------------------------------------------------------
    dice_points = []
    for name, res_list in aggregate_results.items():
        if "_dice_" in name:
            rho = float(name.split("rho")[-1]) if "rho" in name else 0.0
            dice_points.append((rho, res_list[0]["test_accuracy"]))

    if dice_points:
        xs, ys = zip(*sorted(dice_points))
        line_plot(xs, ys, "DiCE worst-group accuracy vs correlation", "ρ", "accuracy", "worst_group_accuracy_dice.pdf")
        print("Figure generated: worst_group_accuracy_dice.pdf")


if __name__ == "__main__":
    main()
