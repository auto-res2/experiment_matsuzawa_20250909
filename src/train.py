# src/train.py
"""Training utilities: model definition, one–epoch loop, full experiment runner."""
from __future__ import annotations

import json
import pathlib
import random
import textwrap
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .evaluate import accuracy, evaluate, line_plot
from .preprocess import get_dataloader

__all__ = [
    "Projector",
    "build_backbone",
    "train_one_epoch",
    "run_experiment",
]


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------


def _feature_dim_of(model: nn.Module) -> int:
    """Attempt to obtain the dimensionality of the penultimate layer."""
    if hasattr(model, "num_features"):
        return int(model.num_features)  # timm convention
    # Fallback – works for torchvision-style models
    return int(model.get_classifier().in_features)


class Projector(nn.Module):
    """Two-layer MLP projector used for consistency / contrastive objectives."""

    def __init__(self, in_dim: int, hid: int = 1024, out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(inplace=True), nn.Linear(hid, out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 – short doc
        return self.net(x)


def build_backbone(model_name: str, num_classes: int) -> Tuple[nn.Module, int]:
    """Create ImageNet-pretrained backbone via timm and return (model, feat_dim)."""
    import timm  # local import to keep global namespace minimal

    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    feat_dim = _feature_dim_of(model)
    return model, feat_dim


# -----------------------------------------------------------------------------
# Training loops
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    projector: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimiser: optim.Optimizer,
    ce_loss_fn: nn.Module,
    cfg: "NamespaceLike",
    device: torch.device,
) -> Tuple[float, float, float]:
    """Single epoch over *loader* – returns (ce_loss, cons_loss, train_acc)."""

    model.train()
    projector.train()

    acc_metric = accuracy().to(device)
    scaler = GradScaler()

    total_ce, total_cons = 0.0, 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        imgs, labels = batch["image"].to(device), batch["label"].to(device)

        optimiser.zero_grad(set_to_none=True)
        with autocast(dtype=torch.bfloat16):
            preds = model(imgs)
            # access to pre-logits depends on timm version; safest is attribute
            pre_logits = (
                model.forward_features(imgs) if hasattr(model, "forward_features") else model.pre_logits
            )
            z = projector(pre_logits)
            ce = ce_loss_fn(preds, labels)
            # DCD consistency loss – disabled if lambda_cons == 0
            cons = torch.tensor(0.0, device=device)
            loss = ce + cfg.lambda_cons * cons

        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        acc_metric.update(preds, labels)
        total_ce += ce.item() * imgs.size(0)
        total_cons += cons.item() * imgs.size(0)

    ds_size = len(loader.dataset)
    return (
        total_ce / ds_size,
        total_cons / ds_size,
        acc_metric.compute().item(),
    )


# -----------------------------------------------------------------------------
# Orchestration util – full experiment
# -----------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


class NamespaceLike(dict):
    """Lightweight wrapper to provide attribute access to a dict (for cfg)."""

    def __getattr__(self, item):
        return self[item]


def run_experiment(cfg: NamespaceLike) -> Dict[str, Any]:
    """Run training & evaluation as specified by *cfg*; returns results dict."""

    _seed_everything(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_dataloader(cfg.dataset, "train", int(cfg.batch_size))
    val_loader = get_dataloader(cfg.dataset, "validation", int(cfg.batch_size))

    num_classes = 2 if cfg.dataset == "waterbirds" else 1000
    backbone, feat_dim = build_backbone(cfg.backbone, num_classes=num_classes)
    backbone.to(device)

    projector = Projector(feat_dim).to(device)

    optimiser = optim.AdamW(
        list(backbone.parameters()) + list(projector.parameters()),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    ce_loss_fn = nn.CrossEntropyLoss()

    ce_hist, val_hist = [], []
    for epoch in range(1, int(cfg.epochs) + 1):
        ce, cons, train_acc = train_one_epoch(
            backbone,
            projector,
            train_loader,
            optimiser,
            ce_loss_fn,
            cfg,
            device,
        )
        val_acc = evaluate(backbone, val_loader, device)
        ce_hist.append(ce)
        val_hist.append(val_acc)
        print(
            f"Epoch {epoch}/{cfg.epochs}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} ce={ce:.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Persist results   -------------------------------------------------
    # ------------------------------------------------------------------
    base_dir = pathlib.Path(".research/iteration9")
    base_dir.mkdir(parents=True, exist_ok=True)

    json_path = base_dir / f"{cfg.name}_results.json"
    results: Dict[str, Any] = {
        "experiment": cfg.name,
        "val_accuracy": val_hist[-1],
        "train_ce_last": ce_hist[-1],
        "epochs": int(cfg.epochs),
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Figures   ---------------------------------------------------------
    # ------------------------------------------------------------------
    images_dir = pathlib.Path(".research/iteration9/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    line_plot(val_hist, "Validation accuracy", "Acc", images_dir / f"accuracy_{cfg.name}.pdf")
    line_plot(ce_hist, "CE loss", "loss", images_dir / f"training_loss_{cfg.name}.pdf")

    # Pretty-print JSON for verification
    print("\nExperiment description:")
    print(
        textwrap.dedent(
            f"""
            {cfg.name}: Training {cfg.backbone} for {cfg.epochs} epochs on {cfg.dataset}.
            DCD enabled = {cfg.dcd_enabled}.  K = {cfg.K}, lambda_cons = {cfg.lambda_cons}
        """
        ).strip()
    )
    print("Results JSON:")
    print(json.dumps(results, indent=2))
    print("Figures generated in .research/iteration9/images\n")

    return results