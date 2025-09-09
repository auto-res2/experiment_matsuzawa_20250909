"""preprocess.py
Data downloading / preprocessing logic extracted from the original
`src/datamodules.py`.
"""
from __future__ import annotations
from typing import Tuple, Dict, Any
from pathlib import Path

import torch.utils.data
import torchvision.transforms as T
from torchvision.datasets import ImageFolder, CIFAR10
from datasets import load_dataset
from PIL import Image

# -----------------------------------------------------------------------------
#  Constants & default transforms
# -----------------------------------------------------------------------------

DATA_ROOT = Path('data')
DATA_ROOT.mkdir(exist_ok=True)

_DEF_TRAIN_TRANSF = T.Compose([
    T.RandomResizedCrop(224),
    T.RandAugment(num_ops=2, magnitude=9),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_DEF_EVAL_TRANSF = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------
#  Wrapper for HuggingFace datasets → PyTorch Dataset
# -----------------------------------------------------------------------------

class _HFWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform):
        self.ds = hf_ds; self.tf = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return {"image": self.tf(img), "label": sample["label"]}

# -----------------------------------------------------------------------------
#  Public API – returns the three datasets
# -----------------------------------------------------------------------------

def make_dataset(cfg: Dict[str, Any], accel) -> Tuple[torch.utils.data.Dataset, ...]:
    name = cfg["dataset"].lower()

    def _download(hf_name: str, split: str):
        try:
            return load_dataset(hf_name, split=split)
        except Exception as e:
            raise RuntimeError(f"Failed to download {hf_name}:{split} – {e}")

    if name == 'waterbirds':
        tr = _HFWrapper(_download('grodino/waterbirds', 'train'), _DEF_TRAIN_TRANSF)
        va = _HFWrapper(_download('grodino/waterbirds', 'validation'), _DEF_EVAL_TRANSF)
        te = _HFWrapper(_download('grodino/waterbirds', 'test'), _DEF_EVAL_TRANSF)
    elif name == 'imagenet':
        im_dir = DATA_ROOT / 'imagenet'
        if not im_dir.exists():
            raise RuntimeError('ImageNet directory missing at data/imagenet/')
        tr = ImageFolder(im_dir/'train', _DEF_TRAIN_TRANSF)
        va = ImageFolder(im_dir/'val',   _DEF_EVAL_TRANSF)
        te = va  # reuse val as test
    elif name == 'cifar10':
        tr = CIFAR10(DATA_ROOT, train=True,  transform=_DEF_TRAIN_TRANSF, download=True)
        va = CIFAR10(DATA_ROOT, train=False, transform=_DEF_EVAL_TRANSF,  download=True)
        te = va
    else:
        raise ValueError(f"Unsupported dataset {cfg['dataset']}")

    return tr, va, te
