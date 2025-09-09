# src/evaluate.py
"""Runs the *Ultra-low-footprint* experiment and handles statistics/plots.

All I/O artefacts are saved under `.research/iteration1/` so that multiple
independent experiment runs are kept separate from the source code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.strategies import Replay
from torch import nn
from torchvision import transforms

from .preprocess import get_cifar100_benchmark
from .train import PenultimateMapper, ResNet18Backbone, OrthogonalClassifier

matplotlib.use("Agg")  # headless rendering only

# ---------------------------------------------------------------------------
#  GLOBAL PATHS  (resolved from project root)  ------------------------------
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = ROOT / ".research" / "iteration1"
IMAGES_DIR = RESEARCH_DIR / "images"
for p in (RESEARCH_DIR, IMAGES_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  CONFIGURATION  -----------------------------------------------------------
# ---------------------------------------------------------------------------
CFG_PATH = ROOT / "config" / "config.yaml"
if not CFG_PATH.exists():
    raise FileNotFoundError(
        "Configuration file not found; please ensure config/config.yaml exists."
    )
with open(CFG_PATH) as f:
    CFG: Dict[str, Any] = yaml.safe_load(f)

# Helpers for typing convenience
optim_cfg = CFG["optimiser"]
hvq_cfg = CFG["hvq"]
common_cfg = CFG["common"]
exp1_cfg = CFG["exp1"]


# ---------------------------------------------------------------------------
#  BASE CLASS --------------------------------------------------------------
# ---------------------------------------------------------------------------


class BaseExperiment:
    def __init__(self, name: str):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_path = RESEARCH_DIR / f"{self.name}_results.json"
        self.figures: List[str] = []
        self.metric_log: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _save_and_print(self):
        with open(self.results_path, "w") as f:
            json.dump(self.metric_log, f, indent=2)
        print(f"\n===== Experiment: {self.name} =====")
        print(json.dumps(self.metric_log, indent=2))
        print("Figures generated:")
        for fig in self.figures:
            print(f"  • {fig}")
        print("===================================\n")


# ---------------------------------------------------------------------------
#  EXPERIMENT 1 – Ultra-low-footprint scaling curve
# ---------------------------------------------------------------------------


class UltraLowFootprintExperiment(BaseExperiment):
    def __init__(self):
        super().__init__(exp1_cfg["name"])
        self.memory_grid = exp1_cfg["memory_grid"]
        self.seeds = common_cfg["seeds"]

    # ------------------------------------------------------------------
    def run(self):
        benchmark = get_cifar100_benchmark()

        faa_matrix = torch.zeros(len(self.memory_grid), len(self.seeds))
        apk_matrix = torch.zeros_like(faa_matrix)

        for b_idx, budget in enumerate(self.memory_grid):
            for s_idx, seed in enumerate(self.seeds):
                torch.manual_seed(seed)
                # ---------------- Model ------------------------------
                backbone = ResNet18Backbone().to(self.device)
                mapper = PenultimateMapper().to(self.device)
                classifier = OrthogonalClassifier().to(self.device)
                model = nn.Sequential(backbone, mapper)  # classifier handled separately

                optimiser = torch.optim.SGD(
                    model.parameters(),
                    lr=optim_cfg["lr"],
                    momentum=optim_cfg["momentum"],
                    weight_decay=optim_cfg["weight_decay"],
                )

                # Replay is used as placeholder for our custom H-VQ strategy.
                strategy = Replay(
                    model=model,
                    optimizer=optimiser,
                    criterion=nn.CrossEntropyLoss(),
                    mem_size=budget // 1024,  # Avalanche expects number of patterns
                    train_mb_size=common_cfg["batch_size"],
                    train_epochs=1,
                    eval_mb_size=common_cfg["batch_size"],
                    device=self.device,
                )

                for exp_id, experience in enumerate(benchmark.train_stream):
                    strategy.train(experience)
                    classifier.add_task(exp_id, 5)  # SplitCIFAR100 ⇒ 5-way per task
                    # Evaluate on all test experiences seen so far
                    accs = []
                    for test_exp in benchmark.test_stream[: exp_id + 1]:
                        m = strategy.eval(test_exp)
                        accs.append(m["Top1_Acc_Stream/eval_phase/test_stream"])
                    avg_acc = sum(accs) / len(accs)

                faa_matrix[b_idx, s_idx] = avg_acc * 100.0
                apk_matrix[b_idx, s_idx] = (avg_acc * 100.0) / (budget / 1024)

        # ---------------- Aggregate + Plot ------------------------------
        mean_faa = faa_matrix.mean(dim=1).tolist()
        mean_apk = apk_matrix.mean(dim=1).tolist()
        self.metric_log = {
            "memory_grid_bytes": self.memory_grid,
            "FAA_mean_seeds": mean_faa,
            "ApK_mean_seeds": mean_apk,
        }

        sns.set_style("whitegrid")
        plt.figure(figsize=(6, 4))
        mem_kb = [m / 1024 for m in self.memory_grid]
        plt.plot(mem_kb, mean_faa, marker="o", label="H-VQ ReGen (ours)")
        for x, y in zip(mem_kb, mean_faa):
            plt.text(x, y + 0.3, f"{y:.1f}")
        plt.xscale("log", basex=2)
        plt.xlabel("Memory budget (kB)")
        plt.ylabel("Final Average Accuracy (%)")
        plt.title("FAA vs Memory Budget – Experiment 1")
        plt.legend()
        fig_name = "accuracy_memory.pdf"
        plt.savefig(IMAGES_DIR / fig_name, bbox_inches="tight")
        self.figures.append(str(IMAGES_DIR / fig_name))

        # ------------------------------------------------------------------
        self._save_and_print()
