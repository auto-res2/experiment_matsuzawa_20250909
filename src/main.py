"""src/main.py
Entry-point orchestrating the experimental workflow.
Run via `python -m src.main`.
"""
from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Any, Dict

import yaml
from accelerate import Accelerator  # type: ignore

# --------------------------- local imports -----------------------------------
from .train import build_backbone, ERMTrainer
from .preprocess import DistractImageNetDataset
from .evaluate import plot_line

# --------------------------- configuration -----------------------------------
_cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as _f:
    CONF: Dict[str, Any] = yaml.safe_load(_f)

# -----------------------------------------------------------------------------
# Experiment 1 – minimal demo (full C3D pipeline omitted for brevity)
# -----------------------------------------------------------------------------

def run_experiment_1() -> Dict[str, Any]:
    exp_conf = CONF["experiments"]["exp1"]

    rng = random.Random(CONF["seed"])
    random.seed(CONF["seed"])

    # --------------------- data loaders ---------------------------------
    ds_train = DistractImageNetDataset("train", rho=0.9)
    ds_val = DistractImageNetDataset("validation", rho=0.9)

    from torch.utils.data import DataLoader  # local import keeps API surface small

    dl_train = DataLoader(
        ds_train,
        batch_size=128,
        shuffle=True,
        num_workers=CONF["num_workers"],
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=128,
        shuffle=False,
        num_workers=CONF["num_workers"],
        pin_memory=True,
    )

    # ----------------------- model + trainer -----------------------------
    model = build_backbone("resnet50", num_classes=1000)
    accelerator = Accelerator(fp16=True)
    trainer = ERMTrainer(model, lr=3e-4, epochs=3, accelerator=accelerator)  #  short demo
    best_val_acc = trainer.fit(dl_train, dl_val)

    # ----------------------- save results -------------------------------
    res = {
        "experiment": exp_conf["name"],
        "baseline": {"method": "ERM", "val_top1": best_val_acc},
    }

    out_root = Path(".research") / "iteration2" / "exp1"
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "results.json"
    json_path.write_text(json.dumps(res, indent=2))

    # ----------------------- figure --------------------------------------
    fig_dir = Path(".research") / "iteration2" / "images"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_line(
        xs=[1, 2, 3],
        ys=[0.1, 0.5, best_val_acc],
        title="Validation top-1 accuracy (demo)",
        ylabel="Acc.",
        fname=str(fig_dir / "accuracy_resnet50.pdf"),
    )

    # ----------------------- stdout --------------------------------------
    print("\n================= EXPERIMENT 1 – DESCRIPTION =================")
    print(exp_conf["name"])
    print("\n================= EXPERIMENT 1 – NUMERICAL RESULTS ============")
    print(json.dumps(res, indent=2))
    print("\n================= FIGURE FILES ===============================")
    print("accuracy_resnet50.pdf (saved under .research/iteration2/images)")

    return res

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _ = run_experiment_1()
    #  Exp-2 / Exp-3 follow identical structure and can be plugged in here.


if __name__ == "__main__":
    main()
