# src/preprocess.py
"""Dataset preparation and benchmark helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from avalanche.benchmarks.classic import SplitCIFAR100
from torchvision import transforms

__all__ = ["get_cifar100_benchmark"]


def get_cifar100_benchmark(data_root: str | Path | None = None, n_experiences: int = 20):
    """Return the standard 20-task SplitCIFAR100 benchmark.

    Parameters
    ----------
    data_root: str | Path | None, optional
        Directory used to download/store the CIFAR-100 dataset. Defaults to
        ``<project-root>/.data``.
    n_experiences: int, optional
        Number of experiences (tasks). The canonical SplitCIFAR100 uses 20.
    """

    if data_root is None:
        # Keep raw datasets out of repo – store under project-root/.data
        data_root = Path(__file__).resolve().parent.parent / ".data"
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    # Normalisation constants from CIFAR-100 statistics
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2761)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    benchmark = SplitCIFAR100(
        n_experiences=n_experiences,
        return_task_id=True,
        train_transform=train_transform,
        eval_transform=test_transform,
        dataset_root=str(data_root),
    )
    return benchmark
