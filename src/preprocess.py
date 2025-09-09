from typing import Any

from torchvision import transforms

# Avalanche imports are local so that users who only want preprocessing do not
# need the heavy avalanche dependency.
try:
    from avalanche.benchmarks.classic import SplitCIFAR100
except ImportError as _err:  # pragma: no cover
    raise RuntimeError(
        "'avalanche-lib' is required but not installed. Install via `pip install avalanche-lib`."
    ) from _err

from .train import DATA_DIR

__all__ = ["run_split_cifar100"]


def run_split_cifar100(**kwargs: Any):
    """Create a 20-task Split-CIFAR100 benchmark with common train/eval transforms."""

    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )
    benchmark = SplitCIFAR100(
        20,
        dataset_root=DATA_DIR,
        return_task_id=True,
        train_transform=transform,
        eval_transform=transform,
        **kwargs,
    )
    return benchmark
