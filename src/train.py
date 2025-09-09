# src/train.py
"""Training utilities for the C3D experiments.
Splits the original monolithic script into reusable functions so that
src.main can orchestrate the whole pipeline.
"""
from __future__ import annotations

import math
import json
import random
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms

# local modules
from .preprocess import (
    ExperimentConfig,
    discover_masks,
    generate_counterfactuals,
    get_dataset,
    ROOT,
    RESULTS_DIR,
    _fail,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────────
# Helper – cosine LR schedule ----------------------------------------------------

def _cosine_schedule(total_epochs: int):
    return lambda epoch: 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))

# ────────────────────────────────────────────────────────────────────────────────
# Core CER helper ----------------------------------------------------------------

def cer_loss(original_logits: torch.Tensor, cf_logits: torch.Tensor) -> torch.Tensor:
    """Contextual Effect Regularisation loss (L1 difference)."""
    return (original_logits - cf_logits).abs().mean()

# ────────────────────────────────────────────────────────────────────────────────
# Main training routine ----------------------------------------------------------

def run_training(exp_key: str, exp_cfg: ExperimentConfig) -> Dict:
    """Run the complete training loop for a single experiment.
    Returns the *results_all_seeds* dictionary so that src.main can take care of
    saving, plotting and printing.
    """
    # --------------- data ----------------
    train_ds = get_dataset(exp_cfg.dataset, "train")
    val_split = "validation" if "validation" in train_ds.builder_name else "val"
    val_ds = get_dataset(exp_cfg.dataset, val_split)

    if "label" not in train_ds.column_names:
        _fail("Dataset missing `label` column – cannot proceed.")

    num_classes = exp_cfg.model.num_classes

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def _tf(example):
        example["image"] = transform(example["image"])
        return example

    train_ds.set_transform(_tf)
    val_ds.set_transform(_tf)

    # --------------- model ---------------
    if "resnet" in exp_cfg.model.classifier_name:
        weights = (
            models.ResNet50_Weights.IMAGENET1K_V2 if exp_cfg.model.pretrained else None
        )
        model = models.get_model(exp_cfg.model.classifier_name, weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # vit or other timm model
        try:
            import timm  # local import avoids mandatory dependency if unused
        except ImportError as e:
            _fail(f"[timm missing] Ensure timm is installed – {e}")
        model = timm.create_model(
            exp_cfg.model.classifier_name,
            pretrained=exp_cfg.model.pretrained,
            num_classes=num_classes,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --------------- optimiser --------------
    optim_cfg = exp_cfg.train.optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
        betas=optim_cfg.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _cosine_schedule(exp_cfg.train.epochs)
    )

    # --------------- dataloaders ------------
    world_bs = exp_cfg.train.batch_size
    num_gpus = max(1, torch.cuda.device_count())
    per_device_bs = max(1, world_bs // num_gpus)

    train_loader = DataLoader(
        train_ds,
        batch_size=per_device_bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    # --------------- training loop ----------
    results_all_seeds: Dict = {}

    for seed in exp_cfg.seeds:
        torch.manual_seed(seed)
        random.seed(seed)

        best_val_acc = 0.0
        metrics_history = {"epoch": [], "train_loss": [], "val_acc": []}

        for epoch in range(exp_cfg.train.epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                # Stage-1 & 2 --------------------------------------------------
                masks = discover_masks(imgs, exp_cfg)
                cf_imgs = generate_counterfactuals(imgs, masks, exp_cfg)

                # Forward ------------------------------------------------------
                logits_orig = model(imgs)
                logits_cf = model(cf_imgs.detach())

                ce = F.cross_entropy(logits_orig, labels, label_smoothing=0.1)
                cer = cer_loss(logits_orig, logits_cf)
                loss = ce + exp_cfg.train.lambda_cer * cer

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)

            scheduler.step()
            train_loss = running_loss / len(train_loader.dataset)

            # ---------------- validation ----------------
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(device, non_blocking=True)
                    labels = batch["label"].to(device, non_blocking=True)
                    pred = model(imgs).argmax(dim=1)
                    correct += (pred == labels).sum().item()
            val_acc = correct / len(val_loader.dataset)

            metrics_history["epoch"].append(epoch)
            metrics_history["train_loss"].append(train_loss)
            metrics_history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = RESULTS_DIR / f"{exp_key}_seed{seed}.pt"
                torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_path)

            print(
                f"[{exp_key}|seed{seed}] epoch {epoch+1}/{exp_cfg.train.epochs} "
                f"loss={train_loss:.3f}  val_acc={val_acc:.3f}"
            )

        # --------------- summarise ---------------
        results_all_seeds[seed] = {
            "best_val_acc": best_val_acc,
            "history": metrics_history,
        }

    return results_all_seeds
