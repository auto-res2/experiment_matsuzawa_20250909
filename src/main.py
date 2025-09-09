"""src/main.py – entry-point executed as `python -m src.main`"""

from __future__ import annotations

import yaml
from pathlib import Path

from .evaluate import run_experiment_1

# ---------------------------------------------------------------------------
#                          LOAD CONFIGURATION
# ---------------------------------------------------------------------------

CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CFG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# ---------------------------------------------------------------------------
#                                MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exp_id = CONFIG.get("EXP_ID", "exp1").lower()
    if exp_id == "exp1":
        run_experiment_1(CONFIG)
    else:
        raise NotImplementedError(
            "Only Experiment 1 has been ported to the multi-file layout."
        )
