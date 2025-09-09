"""main.py – orchestrates the depth-sweep experiment end-to-end."""
from __future__ import annotations

import importlib
import json
import pprint
from pathlib import Path

# Local, dynamic imports (work both as package & as scripts) -----------------
_train = importlib.import_module("src.train" if __name__.startswith("src.") else "train")
ExperimentRunner = _train.ExperimentRunner

_eval = importlib.import_module("src.evaluate" if __name__.startswith("src.") else "evaluate")
plot_depth_sweep = _eval.plot_depth_sweep

_pre = importlib.import_module("src.preprocess" if __name__.startswith("src.") else "preprocess")
get_cfg = _pre.get_cfg


def main():
    runner = ExperimentRunner("depth_sweep")
    json_path = runner.run()

    # ---------------------------------------------------------------------
    description = (
        "\n== EXPERIMENT 1 – DEPTH-SWEEP OVERSMOOTHING STUDY ==\n"
        "Goal: Verify that ADiTi-Net maintains accuracy and feature variance when "
        "scaled to deep architectures, compared against baselines (GCN, PairNorm, "
        "NDLS, DGN, APPNP). Metrics recorded: node-classification accuracy, µ(X) "
        "variance, Dirichlet energy. Each configuration is averaged over 20 seeds "
        "with fixed 500 training epochs (no early stopping).\n"
    )
    print(description)

    # Pretty-print numerical results -------------------------------------
    with json_path.open() as fh:
        results = json.load(fh)
    pprint.pp(results)

    # Figure generation --------------------------------------------------
    plot_depth_sweep(json_path)
    print("Figures saved:")
    for p in Path(get_cfg()["global"]["figure_dir"]).glob("*.pdf"):
        print("  •", p.name)


if __name__ == "__main__":
    main()
