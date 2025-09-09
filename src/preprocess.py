"""src/preprocess.py
Dataset wrappers and preprocessing utilities.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from datasets import load_dataset  # type: ignore
from torchvision import transforms  # type: ignore
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt  # type: ignore

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
_cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as _f:
    CONF: Dict[str, Any] = yaml.safe_load(_f)

# -----------------------------------------------------------------------------
# Helper – availability check
# -----------------------------------------------------------------------------

def _assert_dataset_available(dataset_name: str) -> None:
    try:
        _ = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as exc:
        raise RuntimeError(
            f"Required dataset '{dataset_name}' is not accessible. Strict abort.\n{exc}"
        ) from exc

# -----------------------------------------------------------------------------
# Waterbirds wrapper (used in Exp-2 but included here for completeness)
# -----------------------------------------------------------------------------

class WaterbirdsDataset(Dataset):
    """HF wrapper that yields (image, label, group_tag). Group tag is only used
    for evaluation; never exposed during training.
    """

    def __init__(self, split: str):
        _assert_dataset_available("grodino/waterbirds")
        self.ds = load_dataset(
            "grodino/waterbirds", split=split, cache_dir=CONF["data_root"]
        )
        self.tfm = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = self.tfm(sample["image"])
        label = int(sample["label"])
        group = int(sample["place"])  #  land / water (evaluation only)
        return img, label, group

# -----------------------------------------------------------------------------
# Synthetic DistractImageNet++
# -----------------------------------------------------------------------------

class DistractImageNetDataset(Dataset):
    """On-the-fly creation of DistractImageNet++ images with correlated texture
    patches. Returns (image, class_label, mask).
    """

    TEXTURE_URLS = [
        #  Example subset of 40 public-domain textures (full list omitted)
        "https://huggingface.co/datasets/ayaji/textures/resolve/main/texture00.png",
        "https://huggingface.co/datasets/ayaji/textures/resolve/main/texture01.png",
    ]

    def __init__(self, split: str, rho: float = 0.9, cache_root: str | Path | None = None):
        if cache_root is None:
            cache_root = Path(CONF["data_root"]) / "distract_imagenet" / split
        cache_root = Path(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        self.cache_root = cache_root
        self.rho = rho

        base_name = "benjamin-paine/imagenet-1k-256x256"
        _assert_dataset_available(base_name)
        self.base = load_dataset(base_name, split=split, cache_dir=str(cache_root))

        self._ensure_textures()

        self.tfm_img = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.tfm_tex = transforms.ToTensor()

        self.rng = random.Random(CONF["seed"] + hash(split))

    # ------------------------------------------------------------------
    # Texture utilities
    # ------------------------------------------------------------------
    def _ensure_textures(self) -> None:
        tex_dir = Path(CONF["data_root"]) / "textures"
        tex_dir.mkdir(parents=True, exist_ok=True)
        for url in self.TEXTURE_URLS:
            fname = tex_dir / Path(url).name
            if not fname.exists():
                import requests  #  local import avoids unconditional dependency

                r = requests.get(url, timeout=60)
                if r.status_code != 200:
                    raise RuntimeError(f"Failed to download texture '{url}'. Strict abort.")
                fname.write_bytes(r.content)

    # ------------------------------------------------------------------
    # Core logic: paste class-correlated patch
    # ------------------------------------------------------------------
    def _paste_texture(self, pil_img, class_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        use_patch = self.rng.random() < self.rho
        img_rgb = pil_img.convert("RGB")
        mask = torch.zeros((224, 224), dtype=torch.uint8)

        if use_patch:
            tex_idx = class_id % len(self.TEXTURE_URLS)
            tex_path = (
                Path(CONF["data_root"]) / "textures" / f"texture{tex_idx:02d}.png"
            )
            tex_np = plt.imread(str(tex_path))  #  H×W×C, in [0,1]

            # Random 10% window (≈70×70 on 224×224 crop)
            x0 = self.rng.randint(0, 224 - 70)
            y0 = self.rng.randint(0, 224 - 70)

            img_np = plt.imread(pil_img)
            img_np[y0 : y0 + 70, x0 : x0 + 70, :3] = tex_np[y0 : y0 + 70, x0 : x0 + 70, :3]
            mask[y0 : y0 + 70, x0 : x0 + 70] = 1

            from PIL import Image  #  late import

            img_rgb = Image.fromarray((img_np * 255).astype("uint8"))

        return self.tfm_img(img_rgb), mask

    # ------------------------------------------------------------------
    # Public Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        x, m = self._paste_texture(sample["image"], int(sample["label"]))
        return x, int(sample["label"]), m
