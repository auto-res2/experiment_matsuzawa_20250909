"""
main.py – orchestrates the complete workflow via relative imports.  It reads the
YAML configuration, iterates over seeds/ρ-grid, calls the training routine, and
prints every generated result JSON to stdout for verification as requested.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

import yaml

from train import run_experiment  # relative import (this module lives in src)


# -----------------------------------------------------------------------------
#  1. load configuration -------------------------------------------------------
# -----------------------------------------------------------------------------

CONFIG_PATH = pathlib.Path("config/config.yaml")
if not CONFIG_PATH.exists():
    raise FileNotFoundError("Configuration file 'config/config.yaml' not found.")

with CONFIG_PATH.open() as fp:
    CONFIG: Dict[str, Any] = yaml.safe_load(fp)

# -----------------------------------------------------------------------------
#  2. run the experiment(s) ----------------------------------------------------
# -----------------------------------------------------------------------------

all_results = []
for rho in CONFIG["rho_grid"]:
    for seed in CONFIG["seeds"]:
        res = run_experiment(CONFIG, rho=rho, seed=seed)
        all_results.append(res)

# -----------------------------------------------------------------------------
#  3. print out every JSON result for verification ----------------------------
# -----------------------------------------------------------------------------

print("\n===== RESULTS JSON DUMP =====")
results_root = pathlib.Path(CONFIG["output_dir"])
for json_file in sorted(results_root.glob("*.json")):
    print(f"--- {json_file}")
    print(json_file.read_text())
