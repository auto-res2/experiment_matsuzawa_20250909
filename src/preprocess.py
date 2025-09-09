"""
Data-loading & preprocessing utilities used by all experiments.
Only CIFAR-100 split is implemented as per the original monolithic script.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torchvision.transforms as T
import torchvision.datasets as tvds
from torch.utils.data import Subset

from .train import set_global_seed


# --------------------------------------------------------------
# 1.  10×10 Split CIFAR-100 generator
# --------------------------------------------------------------

def split_cifar100(root: Path, seed: int) -> List[Dict[str, Any]]:
    """Return a list of task-dicts for the 10-task split CIFAR-100 stream."""
    set_global_seed(seed)
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    full_train = tvds.CIFAR100(root=str(root), train=True, download=True, transform=transform)
    full_test = tvds.CIFAR100(root=str(root), train=False, download=True, transform=transform)

    classes_per_task = 10
    task_orders = [list(range(i * classes_per_task, (i + 1) * classes_per_task)) for i in range(10)]

    tasks = []
    for task_id, cls_idxs in enumerate(task_orders):
        train_idx = [i for i, lbl in enumerate(full_train.targets) if lbl in cls_idxs]
        test_idx = [i for i, lbl in enumerate(full_test.targets) if lbl in cls_idxs]
        tasks.append({
            "task_id": task_id,
            "train": Subset(full_train, train_idx),
            "test": Subset(full_test, test_idx),
            "classes": cls_idxs,
        })
    return tasks
