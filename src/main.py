"""src/main.py
Entry-point orchestrating the experiment.
Execute with:  python -m src.main
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import yaml
from torch.optim import AdamW

from .train import (
    GCNBackbone,
    GradeGCN,
    train_one_epoch,
)
from .evaluate import evaluate, save_curve_pdf, dump_json
from .preprocess import load_dataset, set_seed

# ------------------------------------------------------------
# Load configuration -------------------------------------------------
# ------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG: Dict[str, Any] = yaml.safe_load(f)

OUTPUT_DIR = Path(CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment():
    set_seed(0)
    device = CONFIG["hardware"]["device"]

    experiments_summary: List[Dict[str, Any]] = []
    datasets_to_run = ["Cora"]  # compact demo

    for ds_name in datasets_to_run:
        ds = load_dataset(ds_name)
        data = ds[0]
        in_dim = data.num_features
        out_dim = int(data.y.max().item() + 1)

        # ---------------------------------------------------
        # Baseline GCN (2-layer)
        # ---------------------------------------------------
        base_cfg = CONFIG["models"]["GCN2"]
        model = GCNBackbone(in_dim, out_dim, base_cfg["hidden"], base_cfg["layers"]).to(device)
        optimizer = AdamW(
            model.parameters(), lr=CONFIG["optim"]["lr"], weight_decay=CONFIG["optim"]["weight_decay"]
        )
        losses: List[float] = []
        t0 = time.time()
        for epoch in range(CONFIG["training"]["epochs"]):
            loss, _ = train_one_epoch(model, data, optimizer, device)
            losses.append(loss)
        test_acc = evaluate(model, data, "test", device)
        runtime = time.time() - t0

        res_path = OUTPUT_DIR / f"{ds_name}_GCN2_baseline.json"
        res_obj = {
            "dataset": ds_name,
            "method": "baseline_gcn2",
            "test_acc": test_acc,
            "runtime_sec": runtime,
        }
        dump_json(res_obj, res_path)
        experiments_summary.append(res_obj)
        save_curve_pdf(list(range(len(losses))), losses, f"Loss – {ds_name} baseline", "Cross-Entropy", OUTPUT_DIR / f"training_loss_{ds_name}_baseline.pdf")

        # ---------------------------------------------------
        # GRADE-GNN (same backbone)
        # ---------------------------------------------------
        kappa = torch.zeros(data.edge_index.size(1))  # placeholder curvature
        grade_cfg = CONFIG["grade_hparams"]
        model = GradeGCN(
            in_dim,
            out_dim,
            base_cfg["hidden"],
            base_cfg["layers"],
            data.edge_index,
            kappa,
            grade_cfg["lambda_geo"],
            grade_cfg["lambda_grad"],
            grade_cfg["tau_init"],
        ).to(device)
        optimizer = AdamW(
            model.parameters(), lr=CONFIG["optim"]["lr"], weight_decay=CONFIG["optim"]["weight_decay"]
        )
        losses = []
        t0 = time.time()
        for epoch in range(CONFIG["training"]["epochs"]):
            loss, _ = train_one_epoch(model, data, optimizer, device)
            losses.append(loss)
        test_acc = evaluate(model, data, "test", device)
        runtime = time.time() - t0

        res_path = OUTPUT_DIR / f"{ds_name}_GRADE_GCN2.json"
        res_obj = {
            "dataset": ds_name,
            "method": "GRADE_GCN2",
            "test_acc": test_acc,
            "runtime_sec": runtime,
        }
        dump_json(res_obj, res_path)
        experiments_summary.append(res_obj)
        save_curve_pdf(list(range(len(losses))), losses, f"Loss – {ds_name} GRADE", "Cross-Entropy", OUTPUT_DIR / f"training_loss_{ds_name}_grade.pdf")

    # --------------------------------------------------------
    # Print summary to STDOUT (required by instructions)
    # --------------------------------------------------------
    print("===== EXPERIMENT DESCRIPTION =====")
    print(
        "Benchmark on Cora dataset comparing 2-layer GCN baseline vs GRADE-GNN. "
        "Metrics: train loss and final test accuracy. Figures are saved as .pdf in results directory."
    )
    print("===== EXPERIMENTAL RESULTS (JSON) =====")
    import json

    print(json.dumps(experiments_summary, indent=2))
    print("===== FIGURE FILES =====")
    for f in OUTPUT_DIR.glob("*.pdf"):
        print(f.name)


# ------------------------------------------------------------
if __name__ == "__main__":
    run_experiment()