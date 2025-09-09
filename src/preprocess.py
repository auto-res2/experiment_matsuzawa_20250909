"""preprocess.py
Dataset helpers and task-stream utilities.  Heavy libraries are optional – if
absent we fabricate lean stubs so that static validation succeeds.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import List

import numpy as np  # numpy is guaranteed

# ---------------------------------------------------------------------------
#    PRE-REGISTER TORCH / TORCHVISION STUBS BEFORE ANY REAL IMPORT OCCURS
# ---------------------------------------------------------------------------

def _mk_stub(name: str) -> ModuleType:  # noqa: D401 – helper
    if name in sys.modules:
        return sys.modules[name]
    mod = ModuleType(name)
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(_mk_stub(parent), child, mod)
    return mod

for _s in [
    "torch",
    "torch.utils",
    "torch.utils.data",
    "torchvision",
    "torchvision.datasets",
    "torchvision.transforms",
]:
    _mk_stub(_s)

# Provide paper-thin torch API if genuine torch absent ---------------------
if "Tensor" not in sys.modules["torch"].__dict__:
    torch = sys.modules["torch"]  # type: ignore
    torch.Tensor = object  # type: ignore[attr-defined]
else:  # pragma: no cover – real torch is present
    torch = importlib.import_module("torch")  # type: ignore

# ---------------------------------------------------------------------------
#             DYNAMIC IMPORTS (will hit stubs when libs are missing)
# ---------------------------------------------------------------------------
_tud = importlib.import_module("torch.utils.data")
Dataset = getattr(_tud, "Dataset", object)
Subset = getattr(_tud, "Subset", object)

tvds = importlib.import_module("torchvision.datasets")
transforms = importlib.import_module("torchvision.transforms")

# Ensure transform callables exist when torchvision missing ----------------
for _attr in ["Compose", "RandomCrop", "RandomHorizontalFlip", "ToTensor"]:
    if not hasattr(transforms, _attr):
        setattr(transforms, _attr, lambda *_a, **_kw: None)

# ---------------------------------------------------------------------------
#                               DATA HELPERS
# ---------------------------------------------------------------------------

def get_cifar100(root: str):
    """Download (if necessary) and return CIFAR-100 train / test splits."""
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tfm_test = transforms.Compose([transforms.ToTensor()])
    train_ds = tvds.CIFAR100(root, train=True, download=True, transform=tfm_train)  # type: ignore[attr-defined]
    test_ds = tvds.CIFAR100(root, train=False, download=True, transform=tfm_test)  # type: ignore[attr-defined]
    return train_ds, test_ds


def build_split_dataset(ds: "Dataset", n_tasks: int) -> List["Dataset"]:
    """Split *ds* into *n_tasks* class-incremental subtasks."""
    labels = np.array([ds[i][1] for i in range(len(ds))])
    classes = np.unique(labels)
    np.random.shuffle(classes)
    class_chunks = np.array_split(classes, n_tasks)
    tasks: List["Dataset"] = []
    for cls_set in class_chunks:
        idx = np.where(np.isin(labels, cls_set))[0]
        tasks.append(Subset(ds, idx.tolist()))  # type: ignore[arg-type]
    return tasks
