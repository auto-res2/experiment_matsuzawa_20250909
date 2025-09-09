# src/preprocess.py
"""Data downloading / preprocessing utilities extracted from the prototype."""
from __future__ import annotations

from pathlib import Path
from typing import TypeVar, Any

import torch

# ---------------------------------------------------------------------------
#  TEMPORARY MONKEY-PATCHES --------------------------------------------------
# ---------------------------------------------------------------------------
# 1. Restore missing `T_co` symbol for Avalanche compatibility (see evaluate.py)
import torch.utils.data.dataset as _torch_dataset  # noqa: E402  (import after torch)
if not hasattr(_torch_dataset, "T_co"):
    _torch_dataset.T_co = TypeVar("T_co", covariant=True)  # type: ignore[attr-defined]

# 2. Restore `DwsConvBlock` required by Avalanche's MobileNetV1 definition
try:
    import pytorchcv.models.common as _pc_common  # noqa: E402
    import pytorchcv.models.mobilenet as _pc_mobilenet  # noqa: E402
    import torch.nn as _nn  # noqa: E402

    if not hasattr(_pc_common, "DwsConvBlock") or not hasattr(_pc_mobilenet, "DwsConvBlock"):

        class DwsConvBlock(_nn.Sequential):  # type: ignore[misc]
            def __init__(
                self,
                in_channels: int,
                out_channels: int,
                kernel_size: int | tuple[int, int] = 3,
                stride: int | tuple[int, int] = 1,
                padding: int | tuple[int, int] | None = None,
                **_: Any,
            ) -> None:
                if padding is None:
                    padding = kernel_size // 2 if isinstance(kernel_size, int) else kernel_size[0] // 2
                layers = [
                    _nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size,
                        stride,
                        padding,
                        groups=in_channels,
                        bias=False,
                    ),
                    _nn.BatchNorm2d(in_channels),
                    _nn.ReLU6(inplace=True),
                    _nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                    _nn.BatchNorm2d(out_channels),
                    _nn.ReLU6(inplace=True),
                ]
                super().__init__(*layers)

        _pc_common.DwsConvBlock = DwsConvBlock  # type: ignore[attr-defined]
        _pc_mobilenet.DwsConvBlock = DwsConvBlock  # type: ignore[attr-defined]
except ModuleNotFoundError:
    raise

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
