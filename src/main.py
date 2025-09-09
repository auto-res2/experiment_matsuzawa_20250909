# src/main.py
"""Entry-point that orchestrates the complete workflow using the refactored
modules (`preprocess`, `train`, `evaluate`).  Execute via

    python -m src.main

No command-line arguments are required – configuration is read from
`config/config.yaml`.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .evaluate import UltraLowFootprintExperiment
from .preprocess import check_and_download_cifar100

# ---------------------------------------------------------------------------
#  CONFIGURATION VALIDATION  -------------------------------------------------
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config" / "config.yaml"
if not CFG_PATH.exists():
    raise FileNotFoundError("config/config.yaml not found – aborting.")
with open(CFG_PATH) as f:
    _ = yaml.safe_load(f)  # simply validate YAML is well-formed


# ---------------------------------------------------------------------------
#  MAIN  --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    # Ensure required datasets are available (strict no-fallback policy)
    check_and_download_cifar100()

    # Run Experiment-1 only (others can be added analogously)
    exp1 = UltraLowFootprintExperiment()
    exp1.run()


if __name__ == "__main__":  # pragma: no cover
    main()
