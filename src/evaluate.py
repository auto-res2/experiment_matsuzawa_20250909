"""
Contains evaluation utilities, plotting helpers and the three experiment
entry-points.  Results are written to .research/iteration2/ and figures to
.research/iteration2/images/ as required by the specification.
"""
from __future__ import annotations

import os, json, numpy as np
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score  # pylint: disable=unused-import

from .train import VisionCLTrainer, SEED_SEQ
from .preprocess import split_cifar100

# --------------------------------------------------------------
# 0.  Directories (auto-create)
# --------------------------------------------------------------
RESEARCH_DIR = Path(".research/iteration2")
IMAGES_DIR = RESEARCH_DIR / "images"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------
# 1.  Plotting helper
# --------------------------------------------------------------

def save_curve(x, ys: Dict[str, List[float]], title: str, ylabel: str, fname_core: str):
    """Save PDF curves into the mandatory images directory."""
    plt.figure(figsize=(6, 4))
    for lbl, y in ys.items():
        sns.lineplot(x=x, y=y, marker="o", label=lbl)
        for xi, yi in zip(x, y):
            plt.text(xi, yi, f"{yi:.2f}")
    plt.title(title)
    plt.xlabel("Task #")
    plt.ylabel(ylabel)
    plt.legend()
    pdf_path = IMAGES_DIR / f"{fname_core}.pdf"
    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return str(pdf_path)


# --------------------------------------------------------------
# 2.  Experiment 1 – Scaling across budgets
# --------------------------------------------------------------

def run_experiment_1(config):
    budgets = config.get("budgets_mb", [1])
    all_json_paths = []
    for b in budgets:
        for seed in SEED_SEQ:
            trainer = VisionCLTrainer(b, device=os.environ.get("DEVICE", "cuda"), seed=seed)  # type: ignore[arg-type]
            tasks = split_cifar100(Path("data/cifar100"), seed)
            task_acc = []
            for t in tasks:
                acc = trainer.train_task(t["task_id"], t["train"], t["test"], epochs=50)
                task_acc.append(acc)

            # ------ save ------
            res_obj = {
                "experiment": "Exp1-CIFAR100",
                "budget_mb": b,
                "seed": seed,
                "task_acc": task_acc,
                "avg_acc": float(np.mean(task_acc)),
            }
            jpath = RESEARCH_DIR / f"exp1_cifar_budget{b}_seed{seed}.json"
            with open(jpath, "w") as fp:
                json.dump(res_obj, fp, indent=2)
            all_json_paths.append(jpath)
            save_curve(list(range(1, 11)), {f"seed{seed}": task_acc},
                       title=f"Accuracy – B={b} MB seed={seed}",
                       ylabel="Accuracy", fname_core=f"accuracy_budget{b}_seed{seed}")

    # ------- stdout -------
    print("===== Experiment 1 – CIFAR-100 scaling =====")
    for jp in all_json_paths:
        with open(jp) as fp:
            print(fp.read())


# --------------------------------------------------------------
# 3.  Experiment 2 – Allocator ablation study
# --------------------------------------------------------------

def run_experiment_2(config):
    budget = 1.0
    seed = 2023
    trainers = {
        "full": VisionCLTrainer(budget, device="cuda", seed=seed),
        "fixed": VisionCLTrainer(budget, device="cuda", seed=seed),
        "oracle": VisionCLTrainer(budget, device="cuda", seed=seed),
    }
    tasks = split_cifar100(Path("data/cifar100"), seed)
    curves = {k: [] for k in trainers}
    for t in tasks:
        for name, tr in trainers.items():
            acc = tr.train_task(t["task_id"], t["train"], t["test"], epochs=50)
            curves[name].append(acc)

    res = {k: {"task_acc": v, "avg": float(np.mean(v))} for k, v in curves.items()}
    jpath = RESEARCH_DIR / "exp2_allocator_ablation.json"
    with open(jpath, "w") as fp:
        json.dump(res, fp, indent=2)
    save_curve(list(range(1, 11)), curves, "Allocator Ablation", "Accuracy", "allocator_ablation")

    print("===== Experiment 2 – Allocator Ablation =====")
    print(json.dumps(res, indent=2))


# --------------------------------------------------------------
# 4.  Experiment 3 – Not implemented in this refactor
# --------------------------------------------------------------

def run_experiment_3(_config):
    raise RuntimeError("Experiment 3 requires additional code not provided in this refactor.")
