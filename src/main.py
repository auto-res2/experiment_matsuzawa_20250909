"""src/main.py
Entry-point orchestrating the whole experimental suite.
Can be run via:  python -m src.main
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import yaml

from .preprocess import load_dataset, set_seed, PROJECT_ROOT  # type: ignore
from .train import ExperimentConfig, build_model, train

# -----------------------------------------------------------------------------
# I/O paths --------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Mandatory research directory for this iteration
RESEARCH_DIR = PROJECT_ROOT / ".research" / "iteration13"
IMAGE_DIR = RESEARCH_DIR / "images"
RESULTS_DIR = RESEARCH_DIR  # JSON files live directly here per instruction
for _d in [IMAGE_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Configuration ----------------------------------------------------------------
# -----------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _load_experiment_cfgs() -> List[ExperimentConfig]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    exps = []
    for exp_dict in raw_cfg.get("experiments", []):
        exps.append(ExperimentConfig(**exp_dict))
    return exps


# -----------------------------------------------------------------------------
# Main routine -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    set_seed(11)
    device = (
        "cuda" if (Path("/proc/driver/nvidia").exists() and __import__("torch").cuda.is_available()) else "cpu"
    )
    device = __import__("torch").device(device)

    experiments = _load_experiment_cfgs()

    for cfg in experiments:
        print("=" * 80)
        print(f"Running Experiment: {cfg.name}")
        print("Configuration:")
        print(json.dumps(asdict(cfg), indent=2))

        # ---------------- data ----------------
        data = load_dataset(cfg.dataset_name)
        in_dim = data.num_node_features

        # ---------------- model --------------
        model = build_model(cfg, in_dim)

        # ---------------- train --------------
        metrics = train(model, data, cfg, device, IMAGE_DIR)

        # --------------- save ----------------
        result_path = RESULTS_DIR / f"{cfg.name}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # --------------- print ---------------
        print("Experiment description: Node classification with depth", cfg.depth)
        print("Experimental numerical data:")
        print(json.dumps(metrics, indent=2))
        print("Names of figures summarizing the numerical data:")
        for fig in metrics["figures"]:
            print(fig)
        print("=" * 80)


if __name__ == "__main__":
    main()
