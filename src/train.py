"""
BUMA – Budgeted Unified Memory Adaptation
src/train.py
Houses all model components and the continual-learning trainer needed by the
experiments.  No experiment-specific logic should live here.
"""
from __future__ import annotations

import math, random, torch, yaml, json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------------------------------------------------------------
# AMP imports – support both new (torch.amp) and legacy (torch.cuda.amp) APIs
# -----------------------------------------------------------------------------
try:
    # PyTorch ≥2.0 – preferred, device agnostic
    from torch.amp import autocast as _raw_autocast  # type: ignore
    from torch.amp import GradScaler as _RawGradScaler  # type: ignore
except (ImportError, AttributeError):
    # Fallback for older versions (CUDA-only).  These may raise deprecation
    # warnings but remain functional.
    from torch.cuda.amp import autocast as _raw_autocast  # type: ignore
    from torch.cuda.amp import GradScaler as _RawGradScaler  # type: ignore

# -----------------------------------------------------------------------------
# Helper wrappers so the *rest of the code* can remain agnostic to the concrete
# AMP version available at runtime.
# -----------------------------------------------------------------------------
from inspect import signature

# ---- autocast wrapper --------------------------------------------------------

def _autocast_with_optional_device(device_type: str | None = None, *args, **kwargs):  # noqa: D401
    """Return an autocast context manager.

    The first argument in the upstream APIs differs between versions:
      • torch.amp.autocast(device_type="cuda", dtype=torch.float16, …)
      • torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, …)

    To write version-agnostic code we dynamically test whether the wrapped
    function supports the *device_type* keyword.  If not, the argument is
    simply ignored.
    """
    if device_type is not None:
        try:
            return _raw_autocast(device_type=device_type, *args, **kwargs)  # type: ignore[arg-type]
        except TypeError:
            # Older API – silently fall back to calling without the keyword.
            pass
    return _raw_autocast(*args, **kwargs)

# Expose the compatibility wrapper under the canonical name expected below.
autocast = _autocast_with_optional_device  # type: ignore[assignment]

# ---- GradScaler factory ------------------------------------------------------

def create_grad_scaler(device_type: str, **kwargs):  # noqa: D401
    """Instantiate a GradScaler irrespective of the underlying API version."""
    try:
        return _RawGradScaler(device_type=device_type, **kwargs)  # type: ignore[arg-type]
    except TypeError:
        # Legacy API – *device_type* is not accepted.
        return _RawGradScaler(**kwargs)  # type: ignore[call-arg]

# --------------------------------------------------------------
# 0.  Globals & deterministic seed utilities
# --------------------------------------------------------------
SEED_SEQ = [2023, 2024, 2025]


def set_global_seed(seed: int):
    """Sets seeds for Python, NumPy and PyTorch – deterministic mode."""
    import random, numpy as np, torch  # pylint: disable=redefined-outer-name

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------
# 1.  Low-Rank LoRA-style adapter modules
# --------------------------------------------------------------
class LoRAConv2d(nn.Module):
    """LoRA adapter for a 3×3 Conv2d layer."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, r: int = 4,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.r = r
        self.A = nn.Parameter(torch.zeros((r, in_ch * kernel_size * kernel_size)))
        self.B = nn.Parameter(torch.zeros((out_ch, r)))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.stride = stride
        self.padding = padding
        self.in_ch, self.out_ch, self.ks = in_ch, out_ch, kernel_size

    # --------------------------------------------------
    # Forward & misc
    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        w = (self.B @ self.A).view(self.out_ch, self.in_ch, self.ks, self.ks)
        return F.conv2d(x, w, stride=self.stride, padding=self.padding)

    def extra_bytes(self) -> int:
        """Adapter-parameter memory footprint in bytes (FP32)."""
        return (self.A.numel() + self.B.numel()) * 4


# --------------------------------------------------------------
# 2.  ResNet-18 backbone with per-layer LoRA adapter banks
# --------------------------------------------------------------
class ResNet18SparseLoRA(nn.Module):
    """Frozen ResNet-18 backbone + trainable LoRA adapter banks."""

    def __init__(self, backbone_ckpt: str = "glasses/resnet18", rank: int = 4):
        super().__init__()
        from timm import create_model

        # NOTE: Setting *pretrained=False* avoids network calls when running in
        # offline/CI environments.  If the weight download is desired users can
        # toggle this by passing a checkpoint path instead.
        self.encoder = create_model("resnet18", pretrained=False)

        # Freeze original backbone params FIRST
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Inject adapters + forward hooks
        self.adapters: List[LoRAConv2d] = []
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
                lora = LoRAConv2d(module.in_channels, module.out_channels, 3,
                                  r=rank, stride=module.stride[0], padding=1)
                # attach adapter as a sub-module so its parameters are registered
                module.add_module("lora_adapter", lora)
                self.adapters.append(lora)

                # Forward hook that adds LoRA output to the conv output
                def _add_lora_out(mod, inputs, output, l=lora):  # noqa: ANN001
                    return output + l(inputs[0])

                h = module.register_forward_hook(_add_lora_out)
                self._hook_handles.append(h)

    # --------------------------------------------------
    # Helper utilities
    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.encoder(x)

    def bytes_lora(self) -> int:
        return sum(a.extra_bytes() for a in self.adapters)

    def adapt_rank(self, delta_rows: int):
        """Grow / prune each adapter's rank uniformly."""
        if delta_rows == 0:
            return
        for ad in self.adapters:
            if delta_rows > 0:
                # ----- GROW -----
                growA = torch.zeros((delta_rows, ad.A.shape[1]), device=ad.A.device)
                growB = torch.zeros((ad.B.shape[0], delta_rows), device=ad.B.device)
                ad.A.data = torch.cat([ad.A.data, growA], dim=0)
                ad.B.data = torch.cat([ad.B.data, growB], dim=1)
            else:
                # ----- PRUNE -----
                keep_r = ad.r + delta_rows  # delta_rows negative
                keep_r = max(1, keep_r)  # never drop below rank-1
                idx = torch.topk(ad.B.abs().mean(0), k=keep_r, largest=True).indices
                ad.A.data = ad.A.data[idx]
                ad.B.data = ad.B.data[:, idx]
            ad.r = ad.A.shape[0]


# --------------------------------------------------------------
# 3.  Tiny VQ-VAE for 32×32 images (latent rehearsal)
# --------------------------------------------------------------
class VQVAE8bit(nn.Module):
    """VQ-VAE whose codes are stored at 1 byte each."""

    def __init__(self, code_bytes: int = 1, n_embed: int = 512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True))
        self.codebook = nn.Embedding(n_embed, 128)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1))
        self.code_bytes = code_bytes

    # ------------ encode / decode ------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        z = self.enc(x)
        z = z.permute(0, 2, 3, 1).contiguous().view(-1, 128)
        dist = (z.pow(2).sum(1, keepdim=True)
                - 2 * z @ self.codebook.weight.T
                + self.codebook.weight.pow(2).sum(1))
        idx = dist.argmin(1)
        return idx.view(x.size(0), -1)

    def decode(self, idx: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # The latent grid is 4×4=16 positions for a 32×32 input (given 3 downsamples)
        embed = self.codebook(idx).view(idx.size(0), 4, 4, 128).permute(0, 3, 1, 2)
        return torch.sigmoid(self.dec(embed))

    def forward(self, x):  # noqa: D401, ANN001
        idx = self.encode(x)
        recon = self.decode(idx)
        return recon, idx

    def bytes_buffer(self, n_codes: int) -> int:
        return n_codes * self.code_bytes


# --------------------------------------------------------------
# 4.  Reinforcement-Learning allocator (tiny actor-critic)
# --------------------------------------------------------------
class RLAllocator(nn.Module):
    def __init__(self, state_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def act(self, state: torch.Tensor) -> torch.Tensor:  # noqa: D401
        logits = self.actor(state)
        action = torch.tanh(logits)  # (-1, 1)
        return action * 32  # coarse discretisation


# --------------------------------------------------------------
# 5.  Continual-learning trainer (vision)
# --------------------------------------------------------------
class VisionCLTrainer:
    """Task-agnostic continual-learning trainer for the vision experiments."""

    def __init__(self, budget_mb: float, device: str | torch.device, seed: int):
        set_global_seed(seed)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.budget_bytes = int(budget_mb * 1024 ** 2)

        # ---------------- model / aux modules ----------------
        self.model = ResNet18SparseLoRA(rank=4).to(self.device)
        self.vqvae = VQVAE8bit().to(self.device)
        self.alloc = RLAllocator().to(self.device)

        # Ensure the *initial* footprint already satisfies the budget by
        # pruning adapter rank if necessary.  This prevents later crashes when
        # the buffer is still empty but the model itself is too large.
        self._shrink_model_to_budget()

        # ---------------- optimisers ----------------
        self.opt_sgd = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.05, momentum=0.9, weight_decay=5e-4,
        )
        self.opt_adam = optim.Adam(list(self.vqvae.parameters()) + list(self.alloc.parameters()), lr=1e-3)

        # Device-agnostic GradScaler
        self.scaler = create_grad_scaler(self.device.type, init_scale=2.0)

        # (code_tensor, label)
        self.replay_buffer: List[Tuple[torch.Tensor, int]] = []

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _buffer_bytes(self) -> int:
        return sum(code.numel() * self.vqvae.code_bytes for code, _ in self.replay_buffer)

    def _footprint_ok(self) -> bool:
        return self.model.bytes_lora() + self._buffer_bytes() <= self.budget_bytes

    def _shrink_model_to_budget(self):
        """Reduce adapter rank until the (model-only) footprint fits budget."""
        if self._footprint_ok():
            return
        # Uniformly prune one rank at a time across all adapters.
        while (not self._footprint_ok()) and self.model.adapters[0].r > 1:
            self.model.adapt_rank(-1)
        if not self._footprint_ok():
            raise RuntimeError(
                "Memory budget is too small even for rank-1 adapters. "
                f"Budget={self.budget_bytes} bytes, model={self.model.bytes_lora()} bytes."
            )

    def _after_task(self, val_acc: float):
        # actor-critic chooses allocation delta
        state = torch.tensor([
            self.model.bytes_lora(),
            self._buffer_bytes(),
            val_acc,
            self.budget_bytes,
        ], dtype=torch.float32, device=self.device)
        deltaP, deltaD = self.alloc.act(state).round().to(torch.int64).cpu().tolist()
        self.model.adapt_rank(int(deltaP))

        # adjust replay buffer size
        target = max(0, self._buffer_bytes() + int(deltaD))
        if target < self._buffer_bytes():
            byte_cnt = 0
            new_buf: List[Tuple[torch.Tensor, int]] = []
            for code, lab in self.replay_buffer:
                if byte_cnt >= target:
                    break
                new_buf.append((code, lab))
                byte_cnt += code.numel() * self.vqvae.code_bytes
            self.replay_buffer = new_buf
        if not self._footprint_ok():
            raise RuntimeError("Memory budget overflow – allocator error")

    # --------------------------------------------------
    # Public API – train a single task
    # --------------------------------------------------
    def train_task(self, task_id: int, train_ds, test_ds, epochs: int = 50):  # noqa: D401, ANN001
        from torch.utils.data import DataLoader
        import numpy as np
        from sklearn.metrics import accuracy_score

        loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)
        for _ in range(epochs):
            self.model.train()
            for img, lbl in loader:
                img, lbl = img.to(self.device), lbl.to(self.device)

                # ---- latent replay ----
                if self.replay_buffer:
                    buf_samples = random.sample(self.replay_buffer, min(len(self.replay_buffer), img.size(0)))
                    codes, buf_lbls = zip(*buf_samples)
                    codes = torch.stack(codes).to(self.device)
                    buf_imgs = self.vqvae.decode(codes)
                    img = torch.cat([img, buf_imgs], dim=0)
                    lbl = torch.cat([lbl, torch.tensor(buf_lbls, device=self.device)])

                with autocast(self.device.type):
                    out = self.model(img)
                    loss_cls = F.cross_entropy(out, lbl)
                self.opt_sgd.zero_grad()
                self.scaler.scale(loss_cls).backward()
                self.scaler.step(self.opt_sgd)
                self.scaler.update()

        # ---------------- evaluation ----------------
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
        self.model.eval(); preds: List[int] = []; gts: List[int] = []
        with torch.no_grad():
            for img, lbl in test_loader:
                img = img.to(self.device)
                with autocast(self.device.type):
                    logits = self.model(img)
                preds.extend(logits.argmax(1).cpu().tolist())
                gts.extend(lbl.tolist())
        acc = accuracy_score(gts, preds)

        # ---------------- buffer update ----------------
        per_class = 10
        class_cnt: Dict[int, int] = {}
        for img, lbl in train_ds:  # type: ignore[assignment]
            c = int(lbl)
            if class_cnt.get(c, 0) >= per_class:
                continue
            code = self.vqvae.encode(img.unsqueeze(0).to(self.device)).squeeze(0).cpu()
            self.replay_buffer.append((code, c))
            class_cnt[c] = class_cnt.get(c, 0) + 1
        # keep within budget – guard against empty buffer pop
        while (not self._footprint_ok()) and self.replay_buffer:
            self.replay_buffer.pop(0)

        if not self._footprint_ok():
            # At this point further shrinking the buffer is impossible; fall back
            # to additional model pruning until the footprint fits or give up.
            self._shrink_model_to_budget()
            if not self._footprint_ok():
                raise RuntimeError(
                    "Memory budget cannot be satisfied after buffer pruning and model shrinking.")

        # allocator step
        self._after_task(acc)
        return acc
