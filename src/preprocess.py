"""src/preprocess.py
    Data loading & augmentation utilities used by all experiments.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset, DatasetDict
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
#                          GENERIC UTILITIES
# ---------------------------------------------------------------------------

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def coloured(s: str, colour: str = "33"):
    return f"\033[{colour}m{s}\033[0m"

# ---------------------------------------------------------------------------
#                             DATA WRAPPERS
# ---------------------------------------------------------------------------

class ContinualTaskDataset(Dataset):
    """Wraps a HF dataset to also return task_id."""

    def __init__(self, ds, transform, task_id: int):
        self.ds = ds
        self.transform = transform
        self.task_id = task_id

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[int(idx)]
        # ------------------------------------------------------------------
        # Robust image key resolution: fall back to common alternatives.
        # ------------------------------------------------------------------
        img = sample.get("image", None)
        if img is None:
            img = sample.get("img", None)
        if img is None:
            # Last resort: attempt to fetch the first PIL.Image value in sample
            for v in sample.values():
                if "PIL.Image" in str(type(v)):
                    img = v
                    break
        if img is None:
            raise KeyError(
                "No image found in sample – looked for keys ['image', 'img']."
            )

        # Standardise label retrieval (build_stream already ensured column name)
        label = int(sample["label"]) if "label" in sample else int(sample["labels"])

        return self.transform(img), label, self.task_id

# ---------------------------------------------------------------------------
#                          STREAM CONSTRUCTION
# ---------------------------------------------------------------------------

LABEL_CANDIDATES = [
    "label",  # common
    "labels",  # some datasets
    "fine_label",  # CIFAR100 fine-grained
    "coarse_label",  # CIFAR100 coarse
    "class",  # fallback
]


def _ensure_label_column(ds: DatasetDict) -> DatasetDict:
    """Rename an existing class label column to a unified name 'label'."""

    train_cols = set(ds["train"].column_names)
    if "label" in train_cols:
        return ds  # already standardised

    for cand in LABEL_CANDIDATES:
        if cand in train_cols:
            ds = ds.rename_column(cand, "label")
            return ds

    raise RuntimeError("No suitable label column found in the dataset.")


def build_stream(name: str, cfg: Dict) -> List[Tuple[DataLoader, DataLoader]]:
    """Download dataset `name` and build list of (train, test) loaders."""

    print(coloured(f"Building data stream for {name}", "34"))
    task_cfg = cfg["DATA"][name]

    ds_dict: DatasetDict = load_dataset(task_cfg["hf_repo"])
    if "train" not in ds_dict or "test" not in ds_dict:
        raise RuntimeError("Dataset must provide 'train' and 'test' splits.")

    # ------------------------------------------------------------------
    # Standardise label column name across all datasets
    # ------------------------------------------------------------------
    ds_dict = _ensure_label_column(ds_dict)

    train_ds, test_ds = ds_dict["train"], ds_dict["test"]

    all_classes = sorted(list(set(train_ds["label"])))
    task_size, num_tasks = task_cfg["task_size"], task_cfg["num_tasks"]

    class_per_task: List[List[int]] = [
        all_classes[i * task_size : (i + 1) * task_size] for i in range(num_tasks)
    ]

    img_size = task_cfg["img_size"]
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = T.Compose(
        [
            T.Resize(int(img_size * 1.15)),
            T.RandomResizedCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    test_tf = T.Compose(
        [
            T.Resize(int(img_size * 1.15)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    stream = []
    for t_id, cls_group in enumerate(class_per_task):
        tr_indices = [i for i, y in enumerate(train_ds["label"]) if y in cls_group]
        te_indices = [i for i, y in enumerate(test_ds["label"]) if y in cls_group]
        tr_loader = DataLoader(
            ContinualTaskDataset(Subset(train_ds, tr_indices), train_tf, t_id),
            batch_size=cfg["TRAIN"]["batch_size"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        te_loader = DataLoader(
            ContinualTaskDataset(Subset(test_ds, te_indices), test_tf, t_id),
            batch_size=cfg["TRAIN"]["batch_size"],
            shuffle=False,
            num_workers=4,
        )
        stream.append((tr_loader, te_loader))
    return stream
