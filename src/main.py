"""src/main.py
================
Entry-point orchestrating all experiments.
Run with:
    python -m src.main
"""
from __future__ import annotations
import yaml
from pathlib import Path

from .train import Trainer

# --------------------------------------------------------------------------------------
#  Load config YAML
# --------------------------------------------------------------------------------------
CFG_PATH = Path("config/config.yaml")
if not CFG_PATH.exists():
    raise FileNotFoundError("config/config.yaml not found – please create it per instructions.")
with open(CFG_PATH) as fh:
    cfg = yaml.safe_load(fh)

# --------------------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------------------

def main():
    trainer = Trainer(cfg)
    trainer.run_exp1()
    # Future experiments (2,3,...) can be called here.

if __name__ == "__main__":
    main()
