# src/main.py
"""Project entry-point – orchestrates the whole C3D pipeline."""
from __future__ import annotations

import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List

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
# Type registry for forward‐reference resolution ---------------------------------
# ────────────────────────────────────────────────────────────────────────────────
_TYPE_REGISTRY = {
    "ExperimentConfig": ExperimentConfig,
    "DatasetConfig": DatasetConfig,
    "ModelConfig": ModelConfig,
    "TrainConfig": TrainConfig,
    "OptimConfig": OptimConfig,
}

# ────────────────────────────────────────────────────────────────────────────────
# Helper to recursively build dataclasses ---------------------------------------


def _resolve_forward_ref(t):
    """Resolve postponed annotation (string) to the actual type if registered."""
    if isinstance(t, str):
        return _TYPE_REGISTRY.get(t, t)  # fallback to original string if unknown
    return t


def _dict_to_dataclass(d: Any, cls: Any):
    """Recursively convert a (nested) dictionary *d* into an instance of *cls*.

    The implementation is lightweight and purposely avoids external packages. It
    additionally resolves forward references produced by the `from __future__ import
    annotations` directive.
    """
    cls = _resolve_forward_ref(cls)

    # Primitive, list or unsupported target – return as is --------------------
    if not hasattr(cls, "__annotations__"):
        return d

    kwargs: Dict[str, Any] = {}
    for k, t in cls.__annotations__.items():
        if k not in d:
            continue  # use default defined in the dataclass
        val = d[k]
        t_resolved = _resolve_forward_ref(t)
        origin = getattr(t_resolved, "__origin__", None)

        if origin is list:
            sub_cls = t_resolved.__args__[0]
            kwargs[k] = [_dict_to_dataclass(i, sub_cls) for i in val]
        elif origin is dict:
            # dictionary – keep as is (dataclass field default handles it)
            kwargs[k] = val
        else:
            kwargs[k] = _dict_to_dataclass(val, t_resolved)

    try:
        return cls(**kwargs)
    except TypeError as exc:
        print(f"[WARNING] Could not instantiate {cls.__name__}: {exc}. Falling back to raw dict.")
        return d

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
