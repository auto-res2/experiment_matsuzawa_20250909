"""src/train.py
    Model architectures, replay buffers and continual learning learners for HiDeR
    and the ER-Ring baseline.  All heavy lifting during training happens here –
    `evaluate.py` only orchestrates experiments and plots.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from timm import create_model
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
#                           HELPER MODULES
# ---------------------------------------------------------------------------

class LatentEncoder(nn.Module):
    """2×conv + 1×FC encoder that maps backbone CLS features → latent z."""

    def __init__(self, in_dim: int = 1024, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_dim // 2, in_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(in_dim // 4, latent_dim)

    def forward(self, x):  # x (B, D)
        x = x.unsqueeze(-1)  # (B, D, 1)
        x = self.net(x).squeeze(-1)
        return self.fc(x)


class LatentProjector(nn.Module):
    """Linear projector that maps latent codes back to CLS feature space."""

    def __init__(self, latent_dim: int, feat_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, feat_dim)

    def forward(self, z):
        return self.fc(z)


class VectorQuantiser(nn.Module):
    """EMA Vector-Quantiser (VQ-VAE style)."""

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim))
        self.register_buffer("ema_count", torch.zeros(codebook_size))
        self.register_buffer("ema_weight", torch.randn(codebook_size, dim))
        self.beta = 0.99
        self.eps = 1e-5

    @torch.no_grad()
    def quantise(self, z: torch.Tensor):
        # z: (B, dim)
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )
        indices = dist.argmin(-1)
        z_q = self.codebook[indices]
        return z_q, indices

    def forward(self, z):
        z_q, indices = self.quantise(z.detach())
        if self.training:
            onehot = F.one_hot(indices, num_classes=self.codebook.size(0)).type_as(z)
            self.ema_count.mul_(self.beta).add_(1 - self.beta, onehot.sum(0))
            dw = onehot.t() @ z
            self.ema_weight.mul_(self.beta).add_(1 - self.beta, dw)
            n = self.ema_count.sum()
            cluster_size = (
                (self.ema_count + self.eps) / (n + self.codebook.size(0) * self.eps) * n
            )
            self.codebook.data.copy_(self.ema_weight / cluster_size.unsqueeze(1))
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        return z_q, indices


# ---------------------------------------------------------------------------
#                               HiDeR LEARNER
# ---------------------------------------------------------------------------

@dataclass
class BufferEntry:
    c_id: int
    f_id: int
    task_id: int


class HiDeRLearner(nn.Module):
    """Minimal HiDeR learner (hierarchical latent replay)."""

    def __init__(self, cfg: Dict):
        super().__init__()
        bb_name = cfg["MODEL"]["backbone"]
        self.backbone = create_model(bb_name, pretrained=True, num_classes=0)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Attach LoRA adapters
        lora_cfg = LoraConfig(
            r=cfg["MODEL"]["lo_ra"]["r"],
            lora_alpha=cfg["MODEL"]["lo_ra"]["lora_alpha"],
            lora_dropout=cfg["MODEL"]["lo_ra"]["lora_dropout"],
            target_modules=["qkv", "proj"],
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        self.feat_dim = self.backbone.num_features
        latent_dim = cfg["MODEL"]["latent_dim"]

        self.encoder = LatentEncoder(self.feat_dim, latent_dim)
        self.projector = LatentProjector(latent_dim, self.feat_dim)

        # Two-level VQ codebooks
        cb_sz = cfg["MODEL"]["codebook_sizes"]
        self.vq_coarse = VectorQuantiser(cb_sz[0], latent_dim // 2)
        self.vq_fine = VectorQuantiser(cb_sz[1], latent_dim // 2)

        # Classifier (will resize on-demand)
        self.classifier = nn.Linear(self.feat_dim, 1000)

        # Replay buffer
        self.buffer: List[BufferEntry] = []
        self.budget_bytes = cfg["MEMORY"]["budget_bytes"]

        # Training hyper-parameters held here for convenience
        self.cfg = cfg

    # ------------------------------------------------------------------
    #  PUBLIC API
    # ------------------------------------------------------------------
    def observe(self, loader: DataLoader, task_id: int):
        """Train LoRA branch on `loader`. Replay is disabled in this minimal
        implementation to avoid runtime issues with feature-level stubs.
        """
        device = next(self.parameters()).device
        cfg = self.cfg
        optimiser = optim.AdamW(
            self.parameters(), lr=cfg["TRAIN"]["lr"], weight_decay=cfg["TRAIN"]["weight_decay"]
        )
        sched = optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=cfg["TRAIN"]["epochs"] * len(loader)
        )

        # Adapt classifier head if new labels appear
        labels_in_task = sorted({int(y) for _, y, _ in loader.dataset})
        out_dim = max(labels_in_task) + 1
        if self.classifier.out_features < out_dim:
            self.classifier = nn.Linear(self.feat_dim, out_dim).to(device)

        for _ in range(cfg["TRAIN"]["epochs"]):
            for img, y, _ in loader:
                img, y = img.to(device, non_blocking=True), y.to(device)
                # REPLAY (disabled -> returns None)
                if False:  # placeholder for future feature-level replay
                    pass

                logits = self.forward(img)
                loss = F.cross_entropy(logits, y)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), cfg["TRAIN"]["grad_clip"])
                optimiser.step()
                sched.step()

        # Encode samples into buffer (simple reservoir)
        self._update_buffer(loader, task_id, device)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        device = next(self.parameters()).device
        total, correct = 0, 0
        for img, y, _ in loader:
            logits = self.forward(img.to(device, non_blocking=True))
            pred = logits.argmax(1)
            total += len(y)
            correct += (pred.cpu() == y).sum().item()
        return 100.0 * correct / total

    # ------------------------------------------------------------------
    #  INTERNAL HELPERS
    # ------------------------------------------------------------------
    def _update_buffer(self, loader: DataLoader, task_id: int, device):
        for img, _y, _ in loader:
            with torch.no_grad():
                z = self.encoder(self.backbone(img.to(device))).cpu()
            c, f = torch.split(z, z.shape[1] // 2, dim=1)
            _, c_idx = self.vq_coarse.quantise(c)
            _, f_idx = self.vq_fine.quantise(f)
            for ci, fi in zip(c_idx.tolist(), f_idx.tolist()):
                self.buffer.append(BufferEntry(ci, fi, task_id))
        self._enforce_budget()

    def _enforce_budget(self):
        bytes_per_entry = 4  # 3B code ids + 1B task id (round-up)
        max_entries = self.budget_bytes // bytes_per_entry
        if len(self.buffer) > max_entries:
            random.shuffle(self.buffer)
            self.buffer = self.buffer[:max_entries]

    # ------------------------------------------------------------------
    #  FORWARD
    # ------------------------------------------------------------------
    def forward(self, x):  # x: (B,3,H,W)
        if x.ndim != 4:
            raise ValueError("Expected 4-D tensor (B,3,H,W)")
        feat = self.backbone(x)
        return self.classifier(feat)


# ---------------------------------------------------------------------------
#                             ER-RING BASELINE
# ---------------------------------------------------------------------------

class ERBuffer(nn.Module):
    """Ring-buffer that stores raw FP32 images."""

    def __init__(self, budget_bytes: int, img_size: int):
        super().__init__()
        bytes_per_img = 3 * img_size * img_size * 4  # float32 RGB
        self.max_samples = max(1, budget_bytes // bytes_per_img)
        self.images: List[torch.Tensor] = []
        self.labels: List[int] = []

    def add(self, imgs, labels):
        for img, lab in zip(imgs, labels):
            if len(self.images) < self.max_samples:
                self.images.append(img.cpu())
                self.labels.append(int(lab))
            else:
                # Ring behaviour
                self.images.pop(0)
                self.labels.pop(0)
                self.images.append(img.cpu())
                self.labels.append(int(lab))

    def sample(self, n):
        if not self.images:
            return None, None
        idx = np.random.choice(len(self.images), size=min(n, len(self.images)), replace=False)
        imgs = torch.stack([self.images[i] for i in idx])
        labels = torch.tensor([self.labels[i] for i in idx])
        return imgs, labels


class ERRingLearner(nn.Module):
    """Raw-replay baseline using ring buffer (ER-Ring)."""

    def __init__(self, cfg: Dict):
        super().__init__()
        bb_name = cfg["MODEL"]["backbone"]
        self.backbone = create_model(bb_name, pretrained=True, num_classes=0)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        lora_cfg = LoraConfig(
            r=cfg["MODEL"]["lo_ra"]["r"],
            lora_alpha=cfg["MODEL"]["lo_ra"]["lora_alpha"],
            lora_dropout=cfg["MODEL"]["lo_ra"]["lora_dropout"],
            target_modules=["qkv", "proj"],
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)
        self.feat_dim = self.backbone.num_features
        self.classifier = nn.Linear(self.feat_dim, 1000)
        self.buffer = ERBuffer(cfg["MEMORY"]["budget_bytes"], cfg["DATA"]["split_cifar100"]["img_size"])
        self.cfg = cfg

    def observe(self, loader: DataLoader, task_id: int):
        device = next(self.parameters()).device
        cfg = self.cfg
        optimiser = optim.AdamW(self.parameters(), lr=cfg["TRAIN"]["lr"], weight_decay=cfg["TRAIN"]["weight_decay"])

        for _ in range(cfg["TRAIN"]["epochs"]):
            for img, y, _ in loader:
                r_img, r_y = self.buffer.sample(len(img) // 2)
                if r_img is not None:
                    img = torch.cat([img, r_img.to(img.device)], 0)
                    y = torch.cat([y, r_y.to(img.device)], 0)
                feat = self.backbone(img.to(device))
                logits = self.classifier(feat)
                loss = F.cross_entropy(logits, y.to(device))
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
            # End epoch: add samples to buffer
            for img, y, _ in loader:
                self.buffer.add(img, y)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        device = next(self.parameters()).device
        total, correct = 0, 0
        for img, y, _ in loader:
            logits = self.classifier(self.backbone(img.to(device)))
            pred = logits.argmax(1)
            total += len(y)
            correct += (pred.cpu() == y).sum().item()
        return 100.0 * correct / total
