# src/preprocess.py
"""Data loading and basic preprocessing transforms."""
from __future__ import annotations

import os
from typing import Dict, Any

import torch
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader

__all__ = ["get_dataloader"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------------
# Dataset-specific helpers
# -----------------------------------------------------------------------------

def _waterbirds_split(split: str):
    """Return Waterbirds split via HuggingFace *datasets* package."""
    return load_dataset("grodino/waterbirds", split=split)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int,
    num_workers: int = 8,
) -> DataLoader:
    """Factory for dataloaders.

    Only *waterbirds* is wired-up for this demo but the signature is generic.
    """

    if dataset_name.lower() == "waterbirds":
        ds = _waterbirds_split(split)
        transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
        ds.set_transform(
            lambda batch: {
                "image": [transform(img) for img in batch["image"]],
                "label": batch["label"],
            }
        )
    else:
        raise RuntimeError(f"Unsupported dataset '{dataset_name}'.")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
