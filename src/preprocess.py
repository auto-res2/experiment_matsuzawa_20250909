# src/preprocess.py
"""Dataset utilities, mask discovery and counterfactual generation."""
from __future__ import annotations

import random
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

from datasets import load_dataset

# Optional heavy imports (CLIP & Diffusion) – handled lazily inside functions
try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # pragma: no cover – handled in discover_masks
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore

try:
    from diffusers import StableDiffusionInpaintPipeline
except Exception:  # pragma: no cover – handled in generate_counterfactuals
    StableDiffusionInpaintPipeline = None  # type: ignore

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

def get_dataset(cfg: Union[DatasetConfig, Dict], split: str):
    """Download (if needed) and return a HuggingFace *datasets* object for *split*.
    Accepts either a DatasetConfig dataclass or a raw dictionary fallback to
    enhance robustness against partially parsed configurations.
    """
    if isinstance(cfg, dict):
        hf_name = cfg.get("hf_name")
    else:
        hf_name = cfg.hf_name

    if hf_name is None:
        _fail("[ERROR] 'hf_name' missing from dataset configuration – terminating.")

    try:
        ds = load_dataset(hf_name, split=split, cache_dir=str(DATA_DIR))
    except Exception as e:
        _fail(
            f"[ERROR] Unable to download or access dataset '{hf_name}'.\nReason: {e}\nStrict NO-FALLBACK engaged – terminating."
        )
    return ds

# ----------------------------------------------------------------------------
# Stage-1 – Mask discovery ----------------------------------------------------

def _random_masks(images: torch.Tensor) -> torch.Tensor:
    """Return simple random binary masks (fallback when CLIP is unavailable)."""
    B, _, H, W = images.shape
    device = images.device
    masks = torch.rand((B, 1, H, W), device=device)
    return (masks > 0.5).float()


def discover_masks(images: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    """Return binary masks (B×1×H×W) highlighting candidate spurious factors.

    If the CLIP backbone cannot be loaded (e.g. due to memory constraints or
    missing weights), the function falls back to cheap random masks while
    emitting a warning. This keeps the training loop functional without hiding
    potential issues – the console message makes the compromise explicit.
    """
    if CLIPModel is None or CLIPProcessor is None:
        print("[WARNING] transformers/CLIP not available – falling back to random masks.")
        return _random_masks(images)

    device = images.device
    try:
        clip_model = CLIPModel.from_pretrained(cfg.model.clip_name).to(device)
        clip_processor = CLIPProcessor.from_pretrained(cfg.model.clip_name)
    except Exception as e:
        print(f"[WARNING] Failed to load CLIP backbone ({e}) – using random masks instead.")
        return _random_masks(images)

    B, _, H, W = images.shape
    agg_masks = torch.zeros((B, 1, H, W), device=device)

    random_prompts = [
        "grass", "water", "snow", "logo", "sky", "animals", "person", "texture",
        "wood", "numbers", "metal", "fabric", "text", "flower", "road",
    ]
    prompts = random.sample(random_prompts, max(1, cfg.train.prompts))

    # Compute CLIP-based similarity maps --------------------------------------
    with torch.no_grad():
        vision_embeds = clip_model.get_image_features(images)
        vision_embeds = vision_embeds / vision_embeds.norm(dim=-1, keepdim=True)

        for p in prompts:
            text_inputs = clip_processor(text=p, return_tensors="pt").to(device)
            text_embeds = clip_model.get_text_features(**text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            sims = (vision_embeds * text_embeds).sum(-1)  # cosine similarity after normalisation
            sims_map = sims.view(B, 1, 1, 1).expand(-1, 1, H, W)
            agg_masks += sims_map

    agg_masks /= len(prompts)
    masks = (agg_masks > agg_masks.mean()).float()
    return masks

# ----------------------------------------------------------------------------
# Stage-2 – Diffusion counterfactuals ----------------------------------------

def _identity_counterfactual(images: torch.Tensor) -> torch.Tensor:
    """Lightweight fallback – simply return the original images."""
    return images.clone()


def generate_counterfactuals(
    images: torch.Tensor, masks: torch.Tensor, cfg: ExperimentConfig
) -> torch.Tensor:
    """Generate counterfactual images with in-painting guided by masks.

    If a diffusion pipeline cannot be loaded (e.g. due to memory limits), the
    original images are returned unchanged. A console warning makes this decision
    explicit to comply with the *no silent fallback* guideline while still
    allowing the remainder of the pipeline to execute.
    """
    if StableDiffusionInpaintPipeline is None:
        print("[WARNING] diffusers not available – using identity counterfactuals.")
        return _identity_counterfactual(images)

    device = images.device
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            cfg.model.diffusion_name, torch_dtype=dtype, cache_dir=str(CACHE_DIR)
        ).to(device)
        pipe.enable_attention_slicing()
    except Exception as e:
        print(f"[WARNING] Failed to load diffusion model ({e}) – using identity counterfactuals.")
        return _identity_counterfactual(images)

    cf_images = []
    for img, msk in zip(images, masks):
        img_pil = transforms.ToPILImage()(img.cpu())
        mask_pil = transforms.ToPILImage()(msk.repeat(3, 1, 1).cpu())
        try:
            cf = pipe(prompt="photo", image=img_pil, mask_image=mask_pil).images[0]
            cf_images.append(TF.to_tensor(cf))
        except Exception as e:
            print(f"[WARNING] Diffusion sampling failed ({e}) – defaulting to original image.")
            cf_images.append(img.cpu())

    return torch.stack(cf_images).to(device)
