# src/preprocess.py
"""Dataset utilities, mask discovery and counterfactual generation."""
from __future__ import annotations

import random
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionInpaintPipeline

warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────────────────────────────────────────────────────────────────────
# Paths -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
for _d in (DATA_DIR, CACHE_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# Dataclass based config ---------------------------------------------------------
@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 5e-2
    betas: Tuple[float, float] = (0.9, 0.999)

@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    optimiser: OptimConfig
    warmup_epochs: int
    lambda_cer: float
    refresh: int
    prompts: int

@dataclass
class DatasetConfig:
    hf_name: str
    url: str
    split_files: Dict[str, str] = field(default_factory=dict)

@dataclass
class ModelConfig:
    classifier_name: str
    pretrained: bool = True
    num_classes: int = 1000
    clip_name: str = "openai/clip-vit-large-patch14"
    diffusion_name: str = "stabilityai/stable-diffusion-2-inpainting"

@dataclass
class ExperimentConfig:
    name: str
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig
    seeds: List[int]

# ════════════════════════════════════════════════════════════════════════════════
# Utility helpers ----------------------------------------------------------------

def _fail(msg: str):
    print(msg)
    sys.exit(1)

# ----------------------------------------------------------------------------
# Dataset loader --------------------------------------------------------------

def get_dataset(cfg: DatasetConfig, split: str):
    """Download (if needed) and return a HuggingFace *datasets* object for *split*."""
    try:
        ds = load_dataset(cfg.hf_name, split=split, cache_dir=str(DATA_DIR))
    except Exception as e:
        _fail(
            f"[ERROR] Unable to download or access dataset '{cfg.hf_name}'.\nReason: {e}\nStrict NO-FALLBACK engaged – terminating."
        )
    return ds

# ----------------------------------------------------------------------------
# Stage-1 – Mask discovery ----------------------------------------------------

def discover_masks(images: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    """Return binary masks (B×1×H×W) highlighting candidate spurious factors."""
    device = images.device
    try:
        clip_model = CLIPModel.from_pretrained(cfg.model.clip_name).to(device)
        clip_processor = CLIPProcessor.from_pretrained(cfg.model.clip_name)
    except Exception as e:
        _fail(f"Failed to load CLIP backbone: {e}")

    B, _, H, W = images.shape
    agg_masks = torch.zeros((B, 1, H, W), device=device)

    random_prompts = [
        "grass",
        "water",
        "snow",
        "logo",
        "sky",
        "animals",
        "person",
        "texture",
        "wood",
        "numbers",
        "metal",
        "fabric",
        "text",
        "flower",
        "road",
    ]
    prompts = random.sample(random_prompts, cfg.train.prompts)

    for p in prompts:
        text_inputs = clip_processor(text=p, images=None, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_embeds = clip_model.get_image_features(images)
        text_embeds = clip_model.get_text_features(**text_inputs)
        sims = F.cosine_similarity(vision_embeds, text_embeds)
        sims_map = sims.view(B, 1, 1, 1).expand(-1, 1, H, W)
        agg_masks += sims_map

    agg_masks /= cfg.train.prompts
    masks = (agg_masks > agg_masks.mean()).float()
    return masks

# ----------------------------------------------------------------------------
# Stage-2 – Diffusion counterfactuals ----------------------------------------

def generate_counterfactuals(
    images: torch.Tensor, masks: torch.Tensor, cfg: ExperimentConfig
) -> torch.Tensor:
    """Generate counterfactual images with in-painting guided by masks."""
    device = images.device
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            cfg.model.diffusion_name, torch_dtype=dtype, cache_dir=str(CACHE_DIR)
        ).to(device)
    except Exception as e:
        _fail(f"Failed to load diffusion model '{cfg.model.diffusion_name}': {e}")

    pipe.enable_attention_slicing()

    cf_images = []
    for img, msk in zip(images, masks):
        img_pil = transforms.ToPILImage()(img.cpu())
        mask_pil = transforms.ToPILImage()(msk.repeat(3, 1, 1).cpu())
        cf = pipe(prompt="photo", image=img_pil, mask_image=mask_pil).images[0]
        cf_images.append(TF.to_tensor(cf))

    return torch.stack(cf_images).to(device)
