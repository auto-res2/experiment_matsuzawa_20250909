"""
train.py – model creation and training loop extracted from the monolithic script.
Only logic that existed in the original code is preserved.  No new algorithms are
introduced; heavy Diffusion / SAM parts are intentionally left out of the smoke
experiment exactly as in the source.
"""
from __future__ import annotations

import json
import pathlib
import random
import time
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Helper utilities (ex–utils/io.py) – kept local to obey 6-file restriction
# -----------------------------------------------------------------------------

def ensure_dir(p: pathlib.Path) -> None:
    """Create a directory (including parents) if it does not yet exist."""
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
#  Backbone registry (from models/backbone.py)
# -----------------------------------------------------------------------------

import timm  # noqa: E402  pylint: disable=wrong-import-position

_BACKBONES = {
    "resnet50": dict(model_name="resnet50", num_classes=2),
    "vit_b16": dict(model_name="vit_base_patch16_224", num_classes=1000),
    "resnet18": dict(model_name="resnet18", num_classes=10),
}


def create_backbone(name: str, num_classes: int):
    if name not in _BACKBONES:
        raise ValueError(f"Backbone '{name}' is not registered.")
    cfg = _BACKBONES[name]
    model = timm.create_model(cfg["model_name"], pretrained=True, num_classes=num_classes)
    return model


# -----------------------------------------------------------------------------
#  GC-DRO loss (from models/dro.py – unchanged)
# -----------------------------------------------------------------------------

class GCDROLoss(nn.Module):
    """Group-conditional DRO loss – lightweight version used in the paper."""

    def __init__(self, eta: float = 0.05):
        super().__init__()
        self.eta = eta
        self.register_buffer("group_weight", torch.tensor([0.5, 0.5]))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, groups: torch.Tensor):  # noqa: D401,E501 pylint: disable=arguments-differ
        ce = F.cross_entropy(logits, target, reduction="none")
        # groups ∈ {0,1}
        loss_group: List[torch.Tensor] = []
        for g in [0, 1]:
            mask = groups == g
            if mask.sum() == 0:
                loss_group.append(torch.tensor(0.0, device=logits.device))
            else:
                loss_group.append(ce[mask].mean())
        loss_group = torch.stack(loss_group)
        worst = loss_group.max()
        return worst + self.eta * loss_group.mean()


# -----------------------------------------------------------------------------
#  Public training entry point
# -----------------------------------------------------------------------------


def run_experiment(cfg: Dict[str, Any], *, rho: float, seed: int) -> Dict[str, Any]:
    """Train once with the given (rho, seed) settings and return a result dict."""

    # ---------------- preparation ----------------
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from preprocess import WaterbirdsWrapper, build_correlated_subset  # local import to avoid cycles

    full_ds = WaterbirdsWrapper(split="train")
    subset = build_correlated_subset(full_ds, rho=rho, seed=seed)

    train_loader = DataLoader(
        subset,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    val_ds = WaterbirdsWrapper(split="validation")
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4)

    model = create_backbone(cfg["model"], num_classes=2).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"].get("betas", (0.9, 0.999))),
    )
    loss_fn = GCDROLoss(cfg["dice"].get("gc_eta", 0.05)).to(device)

    history: List[float] = []

    # ---------------- training loop ----------------
    num_epochs: int = int(cfg["optim"]["epochs"])
    for epoch in range(num_epochs):
        model.train()
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimiser.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y, groups=y)  # using label as group as in original script
            loss.backward()
            optimiser.step()

        # ---------------- simple validation ----------------
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        history.append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs} – val_acc={val_acc:.4f}")

    # ---------------- persistence ----------------
    results_root = pathlib.Path(cfg["output_dir"])
    images_root = results_root / "images"
    ensure_dir(results_root)
    ensure_dir(images_root)

    res: Dict[str, Any] = {
        "rho": rho,
        "seed": seed,
        "val_acc": history[-1],
        "history": history,
    }

    json_path = results_root / f"{cfg['name']}_rho{rho:.2f}_seed{seed}.json"
    with json_path.open("w") as fp:
        json.dump(res, fp, indent=2)

    #  simple line plot via evaluate.line_plot
    try:
        from evaluate import line_plot

        pdf_path = images_root / (json_path.stem + ".pdf")
        xs = list(range(1, len(history) + 1))
        line_plot(xs, history, title="Validation accuracy", xlabel="epoch", ylabel="acc", pdf_path=str(pdf_path))
    except Exception as exc:  # safety – plotting failure must not crash training
        print(f"[WARN] Failed to plot training curve: {exc}")

    return res
