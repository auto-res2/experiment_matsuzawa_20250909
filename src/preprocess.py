import itertools
import random
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

__all__ = ["split_cifar100", "permuted_mnist"]

# ----------------------------------------------------------------------------------
#  20 × 5-way Split CIFAR-100
# ----------------------------------------------------------------------------------

def split_cifar100(root: str, batch: int = 128):
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)

    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.486, 0.441), (0.267, 0.256, 0.276)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.486, 0.441), (0.267, 0.256, 0.276)),
    ])

    train_set = datasets.CIFAR100(root, train=True, download=True, transform=tf_train)
    test_set = datasets.CIFAR100(root, train=False, download=True, transform=tf_test)

    classes = list(range(100))
    tasks = [classes[i : i + 5] for i in range(0, 100, 5)]

    for tid, cls in enumerate(tasks):
        idx_tr = [i for i, y in enumerate(train_set.targets) if y in cls]
        idx_te = [i for i, y in enumerate(test_set.targets) if y in cls]

        yield (
            tid,
            DataLoader(Subset(train_set, idx_tr), batch_size=batch, shuffle=True, num_workers=4),
            DataLoader(Subset(test_set, idx_te), batch_size=batch, shuffle=False, num_workers=4),
        )


# ----------------------------------------------------------------------------------
#  Permuted-MNIST  (20 tasks by default)
# ----------------------------------------------------------------------------------

def permuted_mnist(root: str, n_tasks: int = 20, batch: int = 128, seed: int = 0):
    torch.manual_seed(seed)
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)

    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(root, train=True, download=True, transform=base_tf)
    test_set = datasets.MNIST(root, train=False, download=True, transform=base_tf)

    pixels = 28 * 28
    permutations = [torch.randperm(pixels) for _ in range(n_tasks)]

    for tid in range(n_tasks):
        p = permutations[tid]

        def _permute(img):
            img = img.view(-1)[p].view(1, 28, 28)
            return img

        tf_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(_permute),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(_permute),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_set.transform = tf_train
        test_set.transform = tf_test

        yield (
            tid,
            DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=2),
            DataLoader(test_set, batch_size=batch, shuffle=False, num_workers=2),
        )
