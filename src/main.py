# src/main.py
"""Project entry-point – orchestrates the whole C3D pipeline."""
from __future__ import annotations

import yaml
from pathlib import Path

from .preprocess import (
    ExperimentConfig,
    DatasetConfig,
    ModelConfig,
    TrainConfig,
    OptimConfig,
    ROOT,
)
from .train import run_training
from .evaluate import save_results, plot_results

CONFIG_PATH = ROOT / "config" / "config.yaml"

# ────────────────────────────────────────────────────────────────────────────────
# Helper to recursively build dataclasses ---------------------------------------

def _dict_to_dataclass(d, cls):
    if not hasattr(cls, "__annotations__"):
        return d  # primitive
    kwargs = {}
    for k, t in cls.__annotations__.items():
        if k not in d:
            continue
        val = d[k]
        origin = getattr(t, "__origin__", None)
        if origin is list:
            sub_cls = t.__args__[0]
            kwargs[k] = [_dict_to_dataclass(i, sub_cls) for i in val]
        elif origin is dict:
            kwargs[k] = val
        else:
            kwargs[k] = _dict_to_dataclass(val, t)
    return cls(**kwargs)

# ────────────────────────────────────────────────────────────────────────────────
# Main driver -------------------------------------------------------------------

def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config YAML not found at {CONFIG_PATH}")

    with open(CONFIG_PATH) as fp:
        raw_cfg = yaml.safe_load(fp)

    for exp_key, cfg_dict in raw_cfg.items():
        exp_cfg: ExperimentConfig = _dict_to_dataclass(cfg_dict, ExperimentConfig)

        results = run_training(exp_key, exp_cfg)
        json_path = save_results(exp_key, results)
        fig_path = plot_results(exp_key, results, exp_cfg.name)

        # stdout summary --------------------------------------------------------
        print("\n────────────────────────  EXPERIMENT DESCRIPTION  ────────────────────────")
        print(f"Experiment key          : {exp_key}")
        print(f"Human-readable name     : {exp_cfg.name}")
        print(f"Dataset URL             : {exp_cfg.dataset.url}")
        print(f"Backbone                : {exp_cfg.model.classifier_name}")
        print(f"Total epochs            : {exp_cfg.train.epochs}")
        print("──────────────────────────────────────────────────────────────────────────")
        with open(json_path) as fp:
            print(fp.read())
        print("Figures saved:")
        print(fig_path.name)


if __name__ == "__main__":
    main()
