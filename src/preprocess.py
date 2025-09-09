"""
preprocess.py – dataset downloading / preprocessing logic
"""
from __future__ import annotations

import os, pathlib, random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageFolder
from datasets import load_dataset

# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_CACHE = ROOT / "data"
DATA_CACHE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
class DatasetBuilder:
    """Central access point for every dataset used in the paper."""

    def __init__(self, name: str, correlation: float | None, seed: int):
        self.name = name
        self.correlation = correlation
        self.seed = seed

    # ------------------------------------------------------------------
    def get(
        self,
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        if self.name == "waterbirds":
            return self._waterbirds()
        if self.name == "celeba_hair":
            return self._celeba_hair()
        if self.name == "imagenet1k":
            return self._imagenet()
        if self.name == "cifar10":
            return self._cifar10()
        raise RuntimeError(f"Dataset {self.name} is not supported.")

    # ------------------------------------------------------------------
    def _waterbirds(self):
        ds = load_dataset("grodino/waterbirds")  # train / validation / test

        def split(ds_split, train=True):
            tfms = self._img_transforms(train)
            return ds_split.with_format("torch").map(
                lambda x: {
                    "image": tfms(x["image"]),
                    "label": torch.tensor(x["label"]),
                    "place": torch.tensor(x["place"]),
                }
            )

        train_raw, val_raw, test_raw = ds["train"], ds["validation"], ds["test"]
        if self.correlation is not None:
            train_raw = self._subsample_by_corr(train_raw, self.correlation)
        return split(train_raw, True), split(val_raw, False), split(test_raw, False)

    # ------------------------------------------------------------------
    def _celeba_hair(self):
        ds = load_dataset("cpuimage/CelebAHairMask-HQ", split="train")
        rng = np.random.RandomState(self.seed)
        idx = np.arange(len(ds))
        rng.shuffle(idx)
        val_sz = int(0.1 * len(ds))
        test_sz = val_sz
        val_idx, test_idx, train_idx = idx[:val_sz], idx[val_sz : val_sz + test_sz], idx[val_sz + test_sz :]
        subsets = {"train": train_idx, "validation": val_idx, "test": test_idx}

        def subset(which):
            sub = ds.select(subsets[which])
            tf = self._img_transforms(which == "train")
            return sub.with_format("torch").map(lambda x: {"image": tf(x["image"]), "label": x["label"]})

        if self.correlation is not None:
            train_split_raw = subset("train")
            train_split = self._subsample_by_corr(train_split_raw, self.correlation, label_key="label", place_key="mask")
        else:
            train_split = subset("train")
        return train_split, subset("validation"), subset("test")

    # ------------------------------------------------------------------
    def _imagenet(self):
        impath = os.environ.get("IMAGENET_DIR")
        if not impath or not pathlib.Path(impath).exists():
            raise RuntimeError(
                "Set IMAGENET_DIR environment variable to a local ImageNet directory. Auto-download is not allowed."
            )
        tf_train = self._img_transforms(True)
        tf_eval = self._img_transforms(False)
        train = ImageFolder(os.path.join(impath, "train"), transform=tf_train)
        val = ImageFolder(os.path.join(impath, "val"), transform=tf_eval)
        return train, val, val  # validation doubles as test for in-domain

    # ------------------------------------------------------------------
    def _cifar10(self):
        tf_train = self._img_transforms(True, size=224)
        tf_eval = self._img_transforms(False, size=224)
        train = CIFAR10(DATA_CACHE, train=True, download=True, transform=tf_train)
        val = CIFAR10(DATA_CACHE, train=True, download=True, transform=tf_eval)
        test = CIFAR10(DATA_CACHE, train=False, download=True, transform=tf_eval)
        return train, val, test

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _img_transforms(train: bool, size: int = 224):
        if train:
            return T.Compose(
                [
                    T.RandomResizedCrop(size),
                    T.RandAugment(2, 9),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        return T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    # ------------------------------------------------------------------
    def _subsample_by_corr(self, dataset, rho: float, label_key: str = "label", place_key: str = "place"):
        from collections import defaultdict

        random.seed(self.seed)
        indices_by_group = defaultdict(list)
        for i, x in enumerate(dataset):
            y = x[label_key]
            g = x[place_key]
            indices_by_group[(y, g)].append(i)

        new_indices = []
        for y in [0, 1]:
            maj = len(indices_by_group[(y, y)])
            min_sz = int((1 - rho) / rho * maj)
            random.shuffle(indices_by_group[(y, y)])
            random.shuffle(indices_by_group[(y, 1 - y)])
            new_indices += indices_by_group[(y, y)][:maj]
            new_indices += indices_by_group[(y, 1 - y)][:min_sz]
        return dataset.select(new_indices)
