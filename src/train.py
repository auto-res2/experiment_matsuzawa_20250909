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
# Global paths that are shared across all modules – UPDATED TO ITERATION20
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
# All JSON artefacts must live under “.research/iteration20/”
RESULTS_DIR = PROJECT_ROOT / ".research" / "iteration20"
# All figure artefacts must live under “.research/iteration20/images”
FIG_DIR = RESULTS_DIR / "images"
# Keep the original data dir unchanged
DATA_DIR = PROJECT_ROOT / "data"

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
        while self.bytes_used + item_bytes > self.byte_budget and self.storage:
            removed = self.storage.pop(0)
            self.bytes_used -= self._sizeof(removed)
        # Only append if it now fits (guard against empty-budget edge-case)
        if self.bytes_used + item_bytes <= self.byte_budget:
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
    """Straightforward VQ layer with EMA-free codebook updates (no commitment loss)."""

    def __init__(self, num_codes: int = 256, code_dim: int = 32):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)
        # Commitment loss weight – kept as a buffer so it travels with .to()
        self.register_buffer("beta", torch.tensor(0.25))
        nn.init.uniform_(
            self.codebook.weight,
            -1 / math.sqrt(num_codes),
            1 / math.sqrt(num_codes),
        )

    def forward(self, z: torch.Tensor):  # type: ignore[override]
        """Quantise last-dimension vectors irrespective of spatial rank (2D or 4D)."""
        original_shape = z.shape

        # Bring channel/code dimension to the end so that the last dim is `D`.
        if z.dim() == 4:  # (B, D, H, W) → (B, H, W, D)
            z_perm = z.permute(0, 2, 3, 1).contiguous()
        elif z.dim() == 2:  # (B, D)
            z_perm = z
        else:
            raise ValueError("Unsupported tensor rank for VectorQuantizer")

        flat_z = z_perm.view(-1, z_perm.size(-1))  # (N, D) where D = code_dim

        # Efficient pair-wise L2 distance to every codebook vector
        dist = (
            flat_z.pow(2).sum(-1, keepdim=True)
            - 2 * flat_z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(-1)
        )  # (N, num_codes)

        idx = dist.argmin(-1)  # (N,)
        z_q = self.codebook(idx).view_as(flat_z)  # (N, D)

        # Straight-through estimator – gradients flow to encoder, codebook gets none
        z_q_st = (z_q.detach() - flat_z).detach() + flat_z

        # Commitment loss (MSE between encoder output and its quantised version)
        loss = ((z_q_st.detach() - flat_z.detach()) ** 2).mean() + self.beta * (
            (z_q - flat_z.detach()) ** 2
        ).mean()

        # Reshape back to original layout
        if z.dim() == 4:
            z_q_st = z_q_st.view(*z_perm.shape).permute(0, 3, 1, 2).contiguous()
        else:  # 2-D
            z_q_st = z_q_st.view(*z_perm.shape)

        return z_q_st, idx.view(original_shape[0], -1), loss


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
            nn.Linear(latent_dim, 128),  # adjusted for 1×1 latent
            nn.ReLU(),
            nn.Linear(128, out_dim),
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
        z_mean = z_q1.mean(dim=[2, 3])  # (B, D)
        z_q2, idx2, _ = self.vq2(z_mean)
        return idx1.cpu(), idx2.cpu()

    @torch.no_grad()
    def decode(self, idx1: torch.Tensor, idx2: torch.Tensor):
        # Decoding only uses the tier-2 indices for the toy sanity-check
        device = self.vq2.codebook.weight.device
        z2 = self.vq2.codebook(idx2.to(device)).view(idx2.shape[0], -1)  # (B, D)
        feat = self.decoder(z2)
        return feat


# ---------------------------------------------------------------------------
# Backbone & strategy wrappers (Avalanche-lib)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# *** Compatibility patches BEFORE importing Avalanche ***
# ---------------------------------------------------------------------------

# 1. PyTorch ≥2.1 removed `T_co` – reintroduce it so older Avalanche versions import.
import types as _types  # noqa: E402
import typing as _typing  # noqa: E402

_torch_dataset_mod = sys.modules.get("torch.utils.data.dataset")
if _torch_dataset_mod is None:
    import torch.utils.data.dataset as _torch_dataset_mod  # type: ignore

if not hasattr(_torch_dataset_mod, "T_co"):
    _torch_dataset_mod.T_co = _typing.TypeVar("T_co", covariant=True)  # type: ignore

# 2. Older Avalanche versions expect `DwsConvBlock` in pytorchcv but recent
#    pytorchcv dropped it. We create a lightweight dummy to satisfy the import.
try:
    from pytorchcv.models.common import DwsConvBlock  # noqa: F401
except (ImportError, AttributeError):

    class _DummyDwsConvBlock(nn.Identity):
        """Minimal no-op replacement for deprecated DwsConvBlock."""

        def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
            super().__init__()

    import importlib
    import sys as _sys

    # Ensure the common submodule exists
    try:
        _common_mod = importlib.import_module('pytorchcv.models.common')
    except ModuleNotFoundError:
        _common_mod = _types.ModuleType('pytorchcv.models.common')
        _sys.modules['pytorchcv.models.common'] = _common_mod  # type: ignore
    setattr(_common_mod, 'DwsConvBlock', _DummyDwsConvBlock)

    # Also register within mobilenet submodule because Avalanche tries both locations
    try:
        _mobile_mod = importlib.import_module('pytorchcv.models.mobilenet')
        setattr(_mobile_mod, 'DwsConvBlock', _DummyDwsConvBlock)
    except ModuleNotFoundError:
        pass

# Heavy torchvision import after the patches
from torchvision import models  # noqa: E402 (delayed heavy import)


def build_backbone() -> nn.Module:
    model = models.resnet18(weights=None)  # Shared backbone across experiments
    model.fc = nn.Linear(model.fc.in_features, 256)
    return model


try:
    from avalanche.training import strategies as cl_strategies  # noqa: E402
    from avalanche.logging import InteractiveLogger  # noqa: E402
    from avalanche.training.plugins import EvaluationPlugin  # noqa: E402
    from avalanche.evaluation.metrics import accuracy_metrics  # noqa: F401,E402
except ImportError as _err:  # pragma: no cover
    raise RuntimeError(
        "'avalanche-lib' is required but not installed. Install via `pip install avalanche-lib`."
    ) from _err


class DERPPStrategy:
    """Thin wrapper around Avalanche's Replay strategy configured as DER++."""

    def __init__(self, model: nn.Module, buffer_size: int, cfg: Dict[str, Any]):
        self.model = model

        # -------------------------------------------------------------------
        # Optimiser instantiation with defensive handling of extra keys
        # -------------------------------------------------------------------
        opt_cfg = dict(cfg.get("optimizer", {}))
        opt_name = opt_cfg.pop("name", "SGD").upper()
        if opt_name != "SGD":
            raise ValueError(f"Only SGD optimiser supported for DER++ baseline, got {opt_name}.")
        optimizer = torch.optim.SGD(model.parameters(), **opt_cfg)

        self.strategy = cl_strategies.Replay(
            model,
            optimizer,
            criterion=nn.CrossEntropyLoss(),
            replay_size=buffer_size,
            train_mb_size=cfg.get("batch_size", 128),
            evaluator=EvaluationPlugin(
                accuracy_metrics(epoch=True, stream=True),
                loggers=[InteractiveLogger()],
            ),
        )
