import time
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms.functional import to_pil_image
import timm

from .preprocess import transform_train

###############################################################################
# Model helpers
###############################################################################

def build_backbone(name: str, num_classes: int) -> nn.Module:
    """Create a timm backbone initialised with ImageNet weights."""
    model = timm.create_model(name, pretrained=True, num_classes=num_classes)
    return model


class Projector(nn.Module):
    """Two–layer MLP projector that maps backbone features to a 128-d representation."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


###############################################################################
# Losses
###############################################################################

class ConsistencyLoss(nn.Module):
    """L2 distance between original and counter-factual features."""

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, z_cf: torch.Tensor) -> torch.Tensor:
        return ((z - z_cf).pow(2).sum(dim=1)).mean()


class SupConLoss(nn.Module):
    """Supervised contrastive loss – https://arxiv.org/abs/2004.11362"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        b = feats.size(0)
        feats = F.normalize(feats, dim=1)
        sim = torch.div(torch.matmul(feats, feats.T), self.T)
        eye = torch.eye(b, dtype=torch.bool, device=feats.device)
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        sim_exp = torch.exp(sim) * (~eye)
        pos_exp = sim_exp * pos_mask
        loss = -torch.log((pos_exp.sum(1) + 1e-6) / (sim_exp.sum(1) + 1e-6)).mean()
        return loss


###############################################################################
# Diffusion-based counter-factual editor
###############################################################################

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

_PIPE = None


def _load_pipe(device: str = "cuda", torch_dtype=torch.float16):
    global _PIPE
    if _PIPE is None:
        _PIPE = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            safety_checker=None,
            torch_dtype=torch_dtype,
        )
        _PIPE.scheduler = DPMSolverMultistepScheduler.from_config(_PIPE.scheduler.config)
        _PIPE = _PIPE.to(device)
        _PIPE.enable_attention_slicing()
    return _PIPE


@torch.no_grad()
def edit_image_pnp(
    pil_img: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance: float = 7.5,
    steps: int = 20,
):
    pipe = _load_pipe()
    edited = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_img,
        guidance_scale=guidance,
        num_inference_steps=steps,
    ).images[0]
    return edited


###############################################################################
# Training loop (single epoch)
###############################################################################

def train_one_epoch(
    model: nn.Module,
    projector: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    dcd_cfg: Dict[str, Any],
    epoch: int,
    scaler: GradScaler,
    device: torch.device,
    img_size: int = 224,
):
    """One training epoch with ERM + consistency + supervised contrastive losses."""

    model.train()
    projector.train()

    ce = nn.CrossEntropyLoss()
    cons = ConsistencyLoss()
    conloss = SupConLoss()

    total, correct = 0, 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        imgs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # --------------------------------------------------------------
        # Counter-factual generation via diffusion editing (on CPU/GPU)
        # --------------------------------------------------------------
        cf_imgs = []
        for img in imgs:
            pil = to_pil_image(img.cpu())
            edited = edit_image_pnp(
                pil_img=pil,
                prompt="a photo of a bird",
                negative_prompt="background",
                guidance=dcd_cfg.get("guidance", 7.5),
                steps=20,
            )
            cf_imgs.append(transform_train(img_size)(edited))

        cf_imgs = torch.stack(cf_imgs).to(device, non_blocking=True)

        with autocast(dtype=getattr(torch, dcd_cfg.get("amp_dtype", "bfloat16"))):
            # Forward pass for original images
            logits = model(imgs)
            feats = model.forward_features(imgs)
            # Forward pass for counterfactual images
            logits_cf = model(cf_imgs)
            feats_cf = model.forward_features(cf_imgs)

            z, z_cf = projector(feats), projector(feats_cf)
            loss = ce(logits, labels) + cons(z, z_cf) + conloss(z, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()

    acc = 100.0 * correct / total
    dt = (time.time() - t0) / 60.0
    print(f"Epoch {epoch:03d} | acc = {acc:5.2f}% | loss = {loss.item():.3f} | time = {dt:.1f} m")
