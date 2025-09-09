from __future__ import annotations

"""src/evaluate.py
    Experiment orchestration, statistical analysis, and plotting utilities.
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .preprocess import build_stream, coloured, ensure_dir
from .train import HiDeRLearner, ERRingLearner

__all__ = ["run_experiment_1"]


# ---------------------------------------------------------------------------
#                       DIRECTORY & PATH CONSTANTS
# ---------------------------------------------------------------------------

# All artefacts must live under .research/iteration2/
BASE_DIR = Path(".research/iteration2")
IMG_DIR = BASE_DIR / "images"

# Ensure required directories exist at import time so that downstream code
# can safely assume their presence regardless of the execution order.
ensure_dir(BASE_DIR)
ensure_dir(IMG_DIR)


def run_experiment_1(cfg: Dict):
    """Standard Benchmark Suite – Split-CIFAR-100 (20 tasks).

    All JSON results are written to .research/iteration2/ and all figures are
    written to .research/iteration2/images/ in accordance with the mandatory
    path constraints.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = build_stream("split_cifar100", cfg)

    learners = {
        "HiDeR": HiDeRLearner(cfg).to(device),
        "ER-Ring": ERRingLearner(cfg).to(device),
    }
    results: Dict[str, List[float]] = {k: [] for k in learners}

    for task_id, (tr_loader, te_loader) in enumerate(stream):
        for name, learner in learners.items():
            learner.observe(tr_loader, task_id)
            acc = learner.evaluate(te_loader)
            results[name].append(acc)
            print(coloured(f"[Task {task_id:02d}] {name} acc={acc:.2f}", "32"))

    # -------------------------------------------------------------
    # Save JSON results & accuracy plot
    json_path = BASE_DIR / "exp1_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    for name, curve in results.items():
        plt.plot(curve, marker="o", label=name)
        for i, v in enumerate(curve):
            plt.text(i, v + 0.3, f"{v:.1f}", fontsize=6)
    plt.xlabel("Task ID")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.title("Continual Accuracy – Split-CIFAR-100")
    fig_path = IMG_DIR / "accuracy_split_cifar100.pdf"
    plt.savefig(fig_path, bbox_inches="tight")

    # Echo description & results for reproducibility
    print(
        """
Experiment 1 – Standard Benchmark Suite
Datasets : Split-CIFAR-100 (20 tasks)
Backbone : ViT-B/16 + LoRA-16 (frozen encoder)
Learners : HiDeR (lat. replay) vs ER-Ring (raw replay)
Budget   : 1 MB each | Epochs/task : 5 | Batch : 128 | AdamW lr 5e-4
"""
    )
    print(json.dumps(results, indent=2))
    print("Figures saved →", fig_path)
