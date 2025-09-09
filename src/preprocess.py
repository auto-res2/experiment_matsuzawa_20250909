"""
preprocess.py – data loading & subset builder taken from data/waterbirds.py.
This is the only dataset necessary for the smoke-test refactor; additional
loaders (Celeb-Hair, ImageNet, etc.) are omitted to honour the 6-file rule.
"""
from __future__ import annotations

import random
from typing import Tuple, List

import datasets  # HuggingFace datasets – used in original code
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms


_DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class WaterbirdsWrapper(Dataset):
    """Thin wrapper that mirrors the functionality in the source script."""

    def __init__(self, split: str = "train", transform=None):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split '{split}'.")
        self.ds = datasets.load_dataset("grodino/waterbirds", split=split, cache_dir="./data")
        self.transform = transform or _DEFAULT_TRANSFORM

    def __len__(self) -> int:  # noqa: D401
        return len(self.ds)

    def __getitem__(self, idx: int):  # noqa: D401
        item = self.ds[idx]
        img = item["image"].convert("RGB")
        label = int(item["label"])
        place = int(item["place"])
        if self.transform:
            img = self.transform(img)
        return img, label, place


# -----------------------------------------------------------------------------
#  Correlated subset builder – unchanged from original script
# -----------------------------------------------------------------------------

def build_correlated_subset(wrapper: WaterbirdsWrapper, *, rho: float, seed: int) -> Subset:
    """Return a subset whose label / spurious correlation ≈ rho (identical logic)."""
    rng = np.random.default_rng(seed)
    labels = np.array([wrapper[i][1] for i in range(len(wrapper))])
    spurious = np.array([wrapper[i][2] for i in range(len(wrapper))])

    indices_by_group = {(y, c): [] for y in (0, 1) for c in (0, 1)}
    for idx, (y, c) in enumerate(zip(labels, spurious)):
        indices_by_group[(y, c)].append(idx)

    selected: List[int] = []
    min_size = min(len(v) for v in indices_by_group.values())
    for y in (0, 1):
        same = indices_by_group[(y, y)]
        diff = indices_by_group[(y, 1 - y)]
        k_same = int(rho * min_size)
        k_diff = int((1 - rho) * min_size)
        selected.extend(rng.choice(same, k_same, replace=False))
        selected.extend(rng.choice(diff, k_diff, replace=False))

    return Subset(wrapper, selected)
