import warnings
from pathlib import Path
from typing import Literal

import torch
import torchvision as tv
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    Normalize,
    InterpolationMode,
)

###############################################################################
# ImageNet normalisation
###############################################################################

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


###############################################################################
# Transforms
###############################################################################

def transform_train(img_size: int = 224):
    return tv.transforms.Compose(
        [
            RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def transform_val(img_size: int = 224):
    return tv.transforms.Compose(
        [
            Resize(int(img_size * 1.15), interpolation=InterpolationMode.BICUBIC),
            CenterCrop(img_size),
            ToTensor(),
            Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


###############################################################################
# Data loader
###############################################################################

def get_dataloader(
    name: str,
    split: Literal["train", "validation", "test"],
    *,
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 8,
):
    """Return a PyTorch `DataLoader` backed by a HuggingFace dataset."""

    cache_dir = Path("data")
    cache_dir.mkdir(exist_ok=True)

    try:
        dataset = load_dataset(name, split=split, cache_dir=str(cache_dir))
    except Exception as e:
        warnings.warn(f"Dataset {name} could not be loaded – {e}")
        raise

    tfm = transform_train(img_size) if split == "train" else transform_val(img_size)

    def _apply(examples):
        # `examples["image"]` is a list of PIL images. Apply `tfm` to each and
        # return a list of tensors to keep the correspondence.
        pixel_values = [tfm(img) for img in examples["image"]]
        return {"pixel_values": pixel_values, "label": examples["label"]}

    # `with_transform` expects the transform to work on *batches* of examples.
    dataset = dataset.with_transform(_apply)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
