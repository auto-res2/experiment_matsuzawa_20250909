"""Entry-point that orchestrates the full experiment."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Dict, Any

import torch
import yaml
from torch.optim import AdamW
from torch_geometric.utils import to_undirected

from .train import VanillaGCN, GradeGCN, train as train_epoch
from .evaluate import test as evaluate, plot_training_loss
from .preprocess import (
    load_dataset,
    set_seed,
    FIG_DIR,
    RES_DIR,
    graph_curvature,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
#  Dataclass mirrors of YAML schema -----------------------------------------
# ---------------------------------------------------------------------------
@dataclass
class DataConf:
    name: str
    url: str

@dataclass
class ModelConf:
    name: str
    type: str
    layers: int
    hidden: int

@dataclass
class ExpConf:
    exp_id: str
    description: str
    datasets: list[DataConf]
    models: list[ModelConf]
    epochs: int
    lr: float
    weight_decay: float
    dropout: float
    lambda_geo: float
    lambda_grad: float
    tau: float
    patience: int


# ---------------------------------------------------------------------------
#  Config loader -------------------------------------------------------------
# ---------------------------------------------------------------------------
from pathlib import Path

CONFIG_FILE = (Path(__file__).resolve().parent.parent / "config" / "config.yaml").resolve()
if not CONFIG_FILE.exists():
    sys.exit(f"Config file not found at {CONFIG_FILE}")

with open(CONFIG_FILE) as fp:
    cfg_raw: Dict[str, Any] = yaml.safe_load(fp)

# --------------------------- Type coercion ---------------------------------
# Explicitly cast numeric fields to ensure correct dtypes (avoids YAML quirks)
_num_keys = [
    "epochs",
    "lr",
    "weight_decay",
    "dropout",
    "lambda_geo",
    "lambda_grad",
    "tau",
    "patience",
]
for k in _num_keys:
    if k in cfg_raw:
        try:
            # cast ints separately to preserve integer nature where relevant
            if isinstance(cfg_raw[k], str) and cfg_raw[k].isdigit():
                cfg_raw[k] = int(cfg_raw[k])
            else:
                cfg_raw[k] = float(cfg_raw[k]) if "lr" in k or "lambda" in k or k in ("weight_decay", "dropout", "tau") else int(cfg_raw[k])
        except ValueError:
            # leave as is; will error later if truly invalid
            pass

# Convert raw dicts into dataclasses for nicer attribute access
CFG = ExpConf(
    exp_id=str(cfg_raw["exp_id"]),
    description=str(cfg_raw["description"]),
    datasets=[DataConf(**d) for d in cfg_raw["datasets"]],
    models=[ModelConf(**m) for m in cfg_raw["models"]],
    epochs=int(cfg_raw["epochs"]),
    lr=float(cfg_raw["lr"]),
    weight_decay=float(cfg_raw["weight_decay"]),
    dropout=float(cfg_raw["dropout"]),
    lambda_geo=float(cfg_raw["lambda_geo"]),
    lambda_grad=float(cfg_raw["lambda_grad"]),
    tau=float(cfg_raw["tau"]),
    patience=int(cfg_raw["patience"]),
)

# ---------------------------------------------------------------------------
#  Single run (one dataset + one model) --------------------------------------
# ---------------------------------------------------------------------------

def run_single(dataset_name: str, model_conf: ModelConf):
    dataset = load_dataset(dataset_name)[0]
    dataset.edge_index = to_undirected(dataset.edge_index)
    data = dataset.to(DEVICE)

    # ----- Split masks --------------------------------------------------
    if hasattr(data, "train_mask"):
        idx_all = torch.arange(data.num_nodes, device=DEVICE)
        split_idx = {
            "train": idx_all[data.train_mask],
            "val": idx_all[data.val_mask],
            "test": idx_all[data.test_mask],
        }
    else:
        perm = torch.randperm(data.num_nodes, device=DEVICE)
        n = data.num_nodes
        split_idx = {
            "train": perm[: int(0.6 * n)],
            "val": perm[int(0.6 * n) : int(0.8 * n)],
            "test": perm[int(0.8 * n) :],
        }

    # ----- Model selection ---------------------------------------------
    set_seed(0)
    if model_conf.name.startswith("gcn") and model_conf.layers == 64 and model_conf.name.endswith("grade"):
        model = GradeGCN(
            data=data,
            hidden=model_conf.hidden,
            layers=model_conf.layers,
            dropout=CFG.dropout,
            lambda_geo=CFG.lambda_geo,
            tau=CFG.tau,
            graph_curvature_fn=graph_curvature,
        ).to(DEVICE)
    elif model_conf.type.upper() == "GCN":
        model = VanillaGCN(
            in_dim=data.num_features,
            out_dim=int(data.y.max()) + 1,
            hidden=model_conf.hidden,
            layers=model_conf.layers,
            dropout=CFG.dropout,
        ).to(DEVICE)
    else:
        raise NotImplementedError("Model type not implemented in STRICT FILE SET.")

    optimiser = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    best_val = 0.0
    patience_left = CFG.patience
    # Ensure best_state is always initialised to a valid state_dict --------
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    history = {"train_loss": [], "val_acc": [], "test_acc": []}

    for epoch in range(1, CFG.epochs + 1):
        loss = train_epoch(model, data, split_idx["train"], optimiser)
        accs, _ = evaluate(model, data, split_idx)
        history["train_loss"].append(loss)
        history["val_acc"].append(accs["val"])
        history["test_acc"].append(accs["test"])

        if accs["val"] > best_val:
            best_val = accs["val"]
            patience_left = CFG.patience
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
        if patience_left == 0:
            break

    model.load_state_dict(best_state)
    final_accs, _ = evaluate(model, data, split_idx)
    return final_accs, history


# ---------------------------------------------------------------------------
#  Experiment launcher -------------------------------------------------------
# ---------------------------------------------------------------------------

def launch_experiment():
    exp_res = {"description": CFG.description, "per_run": {}}

    for ds in CFG.datasets:
        for mdl in CFG.models:
            tag = f"{ds.name}_{mdl.name}"
            print(f"[RUN] {tag}")
            accs, hist = run_single(ds.name, mdl)
            exp_res["per_run"][tag] = {
                "val_acc": accs["val"],
                "test_acc": accs["test"],
                "epochs": len(hist["train_loss"]),
            }
            plot_training_loss(hist["train_loss"], tag)

    # ------------- Persist ---------------------------------------------
    out_file = RES_DIR / f"{CFG.exp_id}.json"
    json.dump(exp_res, out_file.open("w"), indent=2)

    # ------------- Print ------------------------------------------------
    print("\n\n================ Experiment description ================")
    print(CFG.description)
    print("================ Numerical results ====================")
    print(json.dumps(exp_res, indent=2))
    print("================ Figures saved ========================")
    for pdf in FIG_DIR.glob("*.pdf"):
        print(pdf.name)

    # Also output JSON content for verification as mandated ---------
    print("================ Saved JSON content ==================")
    print(out_file.read_text())


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    launch_experiment()