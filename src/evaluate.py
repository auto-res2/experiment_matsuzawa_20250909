"""src/evaluate.py
Utilities for aggregating results and producing the PDF figures + JSON
summary.  No training or model-related code should live here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position


def save_results_and_plot(results: List[Dict], paths: Dict):
    """Persist *results* to `paths["results"]` + generate accuracy bars."""

    grouped: Dict[Tuple[str, str], List[float]] = {}
    for row in results:
        key = (row["dataset"], row["method"])
        grouped.setdefault(key, []).append(row["test_acc"])

    for (dataset, method), accs in grouped.items():
        mean = float(np.mean(accs))
        std = float(np.std(accs))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar([0], [mean], yerr=[std], color="C0", alpha=0.7)
        ax.set_xticks([0])
        ax.set_xticklabels([method])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.annotate(f"{mean:.2%}", xy=(0, mean + 0.01), ha="center")
        ax.set_title(f"{dataset} – {method}")
        fig.tight_layout()
        fname = Path(paths["figures"]) / f"accuracy_{dataset}_{method}.pdf"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"[Figure saved] {fname.relative_to(Path.cwd())}")

    json_path = Path(paths["results"]) / "quick_results.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # stdout dump for CI / human inspection
    print("\n==================== RAW RESULTS ======================")
    print(json.dumps(results, indent=2))
    print("=======================================================")
