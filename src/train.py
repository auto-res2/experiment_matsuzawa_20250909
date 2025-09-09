"""src/train.py
Model construction and training utilities.
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import timm  # type: ignore
from accelerate import Accelerator  # type: ignore

# -----------------------------------------------------------------------------
# Configuration (read once so every module shares identical values)
# -----------------------------------------------------------------------------
_cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as _f:
    CONF = yaml.safe_load(_f)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_backbone(arch: str, num_classes: int) -> nn.Module:
    """Create a classification backbone using timm with ImageNet weights."""
    model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    return model


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy helper (expects logits)."""
    return (pred.argmax(1) == target).float().mean().item()


class ERMTrainer:
    """Empirical-Risk-Minimisation baseline trainer.

    A lightweight wrapper around AdamW + optional mixed-precision via
    HuggingFace Accelerate so the same code path works on single or multi-GPU
    machines.
    """

    def __init__(self, model: nn.Module, lr: float, epochs: int, accelerator: Accelerator):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.accelerator = accelerator
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.05)
        # prepare handles device placement / DDP / AMP automatically
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> float:
        # Prepare the dataloaders once – this moves tensors to the right device
        train_loader, val_loader = self.accelerator.prepare(train_loader, val_loader)

        best_acc: float = 0.0
        for ep in range(self.epochs):
            self.model.train()
            for xb, yb, *_ in train_loader:
                with self.accelerator.accumulate(self.model):
                    logits = self.model(xb)
                    loss = F.cross_entropy(logits, yb)
                    self.accelerator.backward(loss)
                    self.opt.step()
                    self.opt.zero_grad()
            val_acc = self.evaluate(val_loader)
            if self.accelerator.is_main_process:
                print(f"[ERM] epoch {ep + 1}/{self.epochs}  val_acc = {val_acc:.3f}")
            best_acc = max(best_acc, val_acc)
        return best_acc

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        accs: List[torch.Tensor] = []
        for xb, yb, *_ in loader:
            logits = self.model(xb)
            accs.append((logits.argmax(1) == yb).float())
        # Gather across processes
        acc_tensor = torch.cat(accs)
        acc_tensor = self.accelerator.gather(acc_tensor)
        return acc_tensor.mean().item()
