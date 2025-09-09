"""main.py
Entry point that orchestrates a single experiment using the refactored
module structure.  It follows the assignment requirements:
  • read all hyper-parameters from config/config.yaml via PyYAML
  • train + evaluate -> JSON result file in .research/iteration1/
  • create figures inside .research/iteration1/images/
  • print the JSON contents to stdout for verification
"""
from __future__ import annotations
import json, sys, traceback
from pathlib import Path

import yaml

from train import Engine
from evaluate import generate_figures

# -----------------------------------------------------------------------------
#  Configuration loading
# -----------------------------------------------------------------------------
CFG_PATH = Path('config/config.yaml')
if not CFG_PATH.exists():
    raise RuntimeError(f"Configuration file {CFG_PATH} missing – aborting.")
with open(CFG_PATH) as fp:
    CFG = yaml.safe_load(fp)

# -----------------------------------------------------------------------------
#  Output directories
# -----------------------------------------------------------------------------
RESEARCH_DIR = Path('.research/iteration1')
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------

def main():
    cfg = CFG
    print('='*80)
    print(f"Starting experiment: {cfg['experiment_name']}")
    print('='*80)
    try:
        engine = Engine(cfg)
        results = engine.run()  # train + eval
        # ---- figures -----------------------------------------------------
        generate_figures(results, cfg)
        # ---- save JSON ---------------------------------------------------
        json_path = RESEARCH_DIR / f"results-{cfg['experiment_name']}.json"
        with open(json_path, 'w') as fp:
            json.dump(results, fp, indent=2)
        # ---- stdout verification ----------------------------------------
        print('\n--- Configuration ---')
        print(json.dumps(cfg, indent=2))
        print('\n--- Results (json) ---')
        print(json.dumps(results, indent=2))
        print(f"\nResults saved to {json_path}")
    except Exception as e:
        print('\n!! Experiment failed – STRICT NO-FALLBACK triggered !!', file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
