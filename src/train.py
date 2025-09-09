import os
import sys
import json
import random
import math
import time
import pathlib
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Global paths that are shared across all modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "figures"

# Create directories if they do not exist.
for _p in (DATA_DIR, RESULTS_DIR, FIG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Make results reproducible across `random`, `torch` (CPU/GPU)."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Replay-buffer implementation that respects a strict byte cap
# ---------------------------------------------------------------------------
class ByteCappedBuffer:
    """Simple list-based buffer that respects a byte budget using `sys.getsizeof`."""

    def __init__(self, byte_budget: int):
        self.byte_budget = byte_budget
        self.storage: List[Any] = []
        self.bytes_used: int = 0

    def _sizeof(self, obj: Any) -> int:
        """Wrapper around `sys.getsizeof` so we can monkey-patch in tests if needed."""
        return sys.getsizeof(obj)

    def add(self, item: Any) -> None:
        item_bytes = self._sizeof(item)
        if item_bytes > self.byte_budget:
            raise RuntimeError("Single item exceeds buffer budget")
        # Evict oldest items until the new item fits.
        while self.bytes_used + item_bytes > self.byte_budget:
            removed = self.storage.pop(0)
            self.bytes_used -= self._sizeof(removed)
        self.storage.append(item)
        self.bytes_used += item_bytes

    def sample(self, k: int):
        k = min(k, len(self.storage))
        return random.sample(self.storage, k)

    def __len__(self):
        return len(self.storage)


# ---------------------------------------------------------------------------
# Core components of H-VQ-ReGen (minimal viable implementation)
# ---------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int = 256, code_dim: int = 32):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.register_buffer("beta", torch.tensor(0.25))
        nn.init.uniform_(self.codebook.weight,
                         -1 / math.sqrt(num_codes),
                         1 / math.sqrt(num_codes))

    def forward(self, z: torch.Tensor):  # type: ignore[override]
        flat_z = z.view(-1, z.size(-1))  # (B*H, D)
        dist = (
            flat_z.pow(2).sum(-1, keepdim=True)
            - 2 * flat_z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(-1)
        )
        idx = dist.argmin(-1)
        z_q = self.codebook(idx).view_as(z)
        # Straight-through estimator
        z_q_st = (z_q.detach() - z).detach() + z
        loss = ((z_q_st.detach() - z.detach()) ** 2).mean() + self.beta * (
            (z_q - z.detach()) ** 2
        ).mean()
        return z_q_st, idx.view(z.shape[0], -1), loss


class TinyEncoder(nn.Module):
    def __init__(self, code_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, code_dim, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        return self.net(x)


class TinyDecoder(nn.Module):
    def __init__(self, latent_dim: int = 32, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, z: torch.Tensor):  # type: ignore[override]
        return self.net(z)


class HVQReGen(nn.Module):
    """Hierarchical VQ based regenerative memory (high-level, simplified version)."""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.encoder1 = TinyEncoder(32)
        self.vq1 = VectorQuantizer(256, 32)
        # Tier-2 quantises the spatial average of tier-1 codes → single vector
        self.vq2 = VectorQuantizer(4096, 32)
        self.decoder = TinyDecoder(32, feature_dim)

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        z = self.encoder1(x)
        z_q1, idx1, _ = self.vq1(z)
        z_mean = z_q1.mean(dim=[2, 3], keepdim=True)
        z_q2, idx2, _ = self.vq2(z_mean)
        return idx1.cpu(), idx2.cpu()

    @torch.no_grad()
    def decode(self, idx1: torch.Tensor, idx2: torch.Tensor):
        z2 = self.vq2.codebook(idx2.to(self.vq2.codebook.weight.device))
        feat = self.decoder(z2.unsqueeze(-1).unsqueeze(-1))
        return feat


# ---------------------------------------------------------------------------
# Backbone & strategy wrappers (Avalanche-lib)
# ---------------------------------------------------------------------------

from torchvision import models  # noqa: E402  (delayed to avoid heavy import at top level)


def build_backbone() -> nn.Module:
    model = models.resnet18(weights=None)  # Shared backbone across experiments
    model.fc = nn.Linear(model.fc.in_features, 256)
    return model


try:
    from avalanche.training import strategies as cl_strategies  # noqa: E402
    from avalanche.logging import InteractiveLogger  # noqa: E402
    from avalanche.training.plugins import EvaluationPlugin  # noqa: E402
    from avalanche.evaluation.metrics import (
        accuracy_metrics,
        loss_metrics,  # noqa: F401 (imported for completeness)
    )
except ImportError as _err:
    raise RuntimeError(
        "'avalanche-lib' is required but not installed. Install via `pip install avalanche-lib`."
    ) from _err


class DERPPStrategy:
    """Thin wrapper around Avalanche's Replay strategy configured as DER++."""

    def __init__(self, model: nn.Module, buffer_size: int, cfg: Dict[str, Any]):
        self.model = model
        optimizer = torch.optim.SGD(model.parameters(), **cfg["optimizer"])
        self.strategy = cl_strategies.Replay(
            model,
            optimizer,
            criterion=nn.CrossEntropyLoss(),
            replay_size=buffer_size,
            train_mb_size=cfg["batch_size"],
            evaluator=EvaluationPlugin(
                accuracy_metrics(epoch=True, stream=True),
                loggers=[InteractiveLogger()],
            ),
        )
