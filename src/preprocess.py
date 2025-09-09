# src/preprocess.py
"""Data downloading / preprocessing utilities extracted from the prototype."""
from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import torch

# ---------------------------------------------------------------------------
#  TEMPORARY MONKEY-PATCH ----------------------------------------------------
# ---------------------------------------------------------------------------
# See detailed explanation in `src/evaluate.py` – the same workaround is
# required here because Avalanche is imported at module import-time.
import torch.utils.data.dataset as _torch_dataset  # noqa: E402  (import after torch)
if not hasattr(_torch_dataset, "T_co"):
    _torch_dataset.T_co = TypeVar("T_co", covariant=True)  # type: ignore[attr-defined]

from torchvision import transforms, datasets  # noqa: E402
from avalanche.benchmarks.classic import SplitCIFAR100  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
#  CIFAR-100 (Split)  --------------------------------------------------------
# ---------------------------------------------------------------------------

def check_and_download_cifar100() -> None:
    """Makes sure CIFAR-100 is available locally; raises RuntimeError otherwise."""

    try:
        _ = datasets.CIFAR100(root=str(DATA_DIR), download=True)
    except Exception as e:  # pragma: no cover – must abort hard
        raise RuntimeError("Dataset download failed – STRICT NO-FALLBACK.") from e


def get_cifar100_benchmark():
    """Returns 20-task SplitCIFAR100 benchmark identical to the original script."""

    check_and_download_cifar100()
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )
    benchmark = SplitCIFAR100(
        n_experiences=20,
        seed=0,
        train_transform=transform,
        eval_transform=transform,
    )
    return benchmark
