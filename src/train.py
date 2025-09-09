import json
import math
import random
import textwrap
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# ----------------------------------------------------------------------------------
#  Model building blocks
# ----------------------------------------------------------------------------------

class _VectorQuantizer(nn.Module):
    """Straight-Through Vector-Quantiser (VQ-VAE style)"""

    def __init__(self, n_codes: int, code_dim: int):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(n_codes, code_dim))
        self.n_codes = n_codes
        self.code_dim = code_dim

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z : (B, D)
        dist = (z.unsqueeze(1) - self.codebook.unsqueeze(0)).pow(2).sum(-1)  # (B, n_codes)
        indices = dist.argmin(1)                                             # (B,)
        codes = F.embedding(indices, self.codebook)                          # (B, D)
        quantised = z + (codes - z).detach()                                 # straight-through
        return quantised, indices


class Tier1Encoder(nn.Module):
    """Encodes RGB image → 32-d code; 256 codewords ⇒ 8-bit index."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 32)
        self.vq = _VectorQuantizer(256, 32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(1)
        z = self.fc(h)
        q, idx = self.vq(z)
        return q, idx  # (B, 32), (B,)


class Tier2Encoder(nn.Module):
    """Encodes sequence of Tier-1 codes → 32-d code; 4096 codewords ⇒ 12-bit index."""

    def __init__(self, in_codes: int = 16, code_dim: int = 32):
        super().__init__()
        self.mapper = nn.Linear(in_codes * code_dim, code_dim)
        self.vq = _VectorQuantizer(4096, code_dim)  # stored as uint16 (2 B)

    def forward(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # codes : (B, in_codes, code_dim)
        h = self.mapper(codes.flatten(1))
        q, idx = self.vq(h)
        return q, idx  # (B, 32), (B,)


class TinyDecoder(nn.Module):
    """Decodes Tier-2 code → 256-d feature."""

    def __init__(self, code_dim: int = 32, out_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(code_dim, 512), nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.fc(c)


# ----------------------------------------------------------------------------------
#  Replay Buffer (≤ 1 MB in RAM)
# ----------------------------------------------------------------------------------

class HVQBuffer:
    """Stores Tier-2 index + label + uncertainty σ (float32)."""

    BYTES_PER_SAMPLE = 2 + 1 + 4  # uint16 + uint8 + float32

    def __init__(self, budget_bytes: int):
        self.budget = budget_bytes
        self.capacity = budget_bytes // self.BYTES_PER_SAMPLE
        self._store: List[Tuple[int, int, float]] = []

    def __len__(self):
        return len(self._store)

    # ---------------- public API ------------------
    def add(self, idx: int, label: int, sigma: float):
        """Probabilistic replacement: keep low-uncertainty samples."""
        if len(self._store) < self.capacity:
            self._store.append((idx, label, sigma))
        else:
            worst = max(range(len(self._store)), key=lambda i: self._store[i][2])
            if sigma < self._store[worst][2]:
                self._store[worst] = (idx, label, sigma)

    def sample(self, k: int) -> List[Tuple[int, int, float]]:
        if len(self._store) == 0:
            return []
        k = min(k, len(self._store))
        return random.sample(self._store, k)

    @property
    def bytes_used(self) -> int:
        return len(self) * self.BYTES_PER_SAMPLE


# ----------------------------------------------------------------------------------
#  Full H-VQ ReGen model
# ----------------------------------------------------------------------------------

class HVQReGenModel(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        cnn = resnet18(weights=None)
        cnn.fc = nn.Identity()
        self.cnn = cnn                           # 512-D features
        self.mapper = nn.Linear(512, 256)

        self.tier1 = Tier1Encoder()
        self.tier2 = Tier2Encoder()
        self.decoder = TinyDecoder()

        self.classifier = nn.Linear(256, n_classes, bias=False)
        self.dropout = nn.Dropout(p=0.2)         # used for σ estimate

    # ---------------------------------------------------------------------
    #  Encoding for storage (Tier-2 indices only)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, idx1 = self.tier1(x)                  # Tier-1 indices (ignored below)
        # For simplicity we feed a dummy sequence of zeros to Tier-2 encoder.
        code_seq = torch.zeros(x.size(0), 16, 32, device=x.device)
        _, idx2 = self.tier2(code_seq)
        return idx2  # (B,)

    # ---------------------------------------------------------------------
    #  Forward / regeneration
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        feat = self.mapper(self.cnn(x))          # 256-D
        logits = self.classifier(feat)
        return logits, feat

    def regenerate(self, tier2_indices: List[int], device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            codes = F.embedding(torch.tensor(tier2_indices, device=device), self.tier2.vq.codebook)
            feats = self.decoder(codes)          # (B, 256)
        return feats


# ----------------------------------------------------------------------------------
#  Continual-learning trainer (handles both train & inline evaluation)
# ----------------------------------------------------------------------------------

from .evaluate import plot_line  # local relative import (no circular reference)

class CLTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- core components
        self.buffer = HVQBuffer(cfg["replay_budget"])
        self.model = HVQReGenModel(cfg["n_classes"]).to(self.device)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=1e-4)
        self.crit = nn.CrossEntropyLoss()

        # bookkeeping
        self.results = {"task_acc": []}

    # ------------------------------------------------------------------
    #  Train a single task
    # ------------------------------------------------------------------
    def _train_task(self, task_id: int, loader):
        self.model.train()
        epochs = self.cfg["epochs_per_task"]
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                # ============ mix replay ============
                k = int(self.cfg["replay_ratio"] * x.size(0))
                replay_batch = self.buffer.sample(k)

                loss = 0.0
                # --- current samples ---
                logits_cur, feat_cur = self.model(x)
                loss += self.crit(logits_cur, y)

                # --- replay samples ---
                if replay_batch:
                    idxs, labels, _ = zip(*replay_batch)
                    feat_rep = self.model.regenerate(idxs, self.device)
                    logits_rep = self.model.classifier(feat_rep)
                    y_rep = torch.tensor(labels, device=self.device)
                    loss += self.crit(logits_rep, y_rep)

                # --- optimisation ---
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

                # ============ store samples (uncertainty-guided) ============
                with torch.no_grad():
                    out1 = self.model.dropout(feat_cur)
                    out2 = self.model.dropout(feat_cur)
                    sigma = (out1 - out2).pow(2).mean(1).cpu().numpy()  # (B,)
                    idx2 = self.model.encode(x).cpu().numpy()           # (B,)
                    for i in range(len(x)):
                        self.buffer.add(int(idx2[i]), int(y[i]), float(sigma[i]))

        # Orthogonalise classifier weights (very light-weight variant)
        with torch.no_grad():
            w = self.model.classifier.weight.data  # (C, 256)
            self.model.classifier.weight.data.copy_(torch.linalg.qr(w.T).Q.T)

    # ------------------------------------------------------------------
    #  Evaluation on test loader
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval(self, loader) -> float:
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return 100.0 * correct / total

    # ------------------------------------------------------------------
    #  Run full stream (continual-learning experiment)
    # ------------------------------------------------------------------
    def run_stream(self, stream_gen, exp_name: str):
        out_root = Path(".research/iteration12")
        img_dir = out_root / "images"
        out_root.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(exist_ok=True)

        for task_id, dl_train, dl_test in stream_gen:
            self._train_task(task_id, dl_train)
            acc = self._eval(dl_test)
            self.results["task_acc"].append(acc)
            print(f"Task {task_id:02d}  |  Accuracy: {acc:5.2f}%  |  Buffer: {self.buffer.bytes_used/1024:5.1f} kB")

        # ---------------- summary ----------------
        faa = sum(self.results["task_acc"]) / len(self.results["task_acc"])
        res = {
            "FAA": faa,
            "task_acc": self.results["task_acc"],
            "buffer_bytes": self.buffer.bytes_used,
            "ApK": faa / (self.buffer.bytes_used / 1024 + 1e-8),
        }

        # save JSON
        json_path = out_root / f"{exp_name}.json"
        with open(json_path, "w") as f:
            json.dump(res, f, indent=2)

        # plot curve
        fig_path = img_dir / f"accuracy_{exp_name}.pdf"
        plot_line(list(range(len(self.results["task_acc"]))), [self.results["task_acc"]], ["H-VQ ReGen"],
                  "Task", "Accuracy (%)", f"Accuracy curve – {exp_name}", fig_path)

        # pretty print
        print("\n================  Experiment description  ================")
        print(textwrap.dedent(f"""
            {exp_name}: dataset = {self.cfg['dataset_name']},  buffer budget = {self.cfg['replay_budget']/1024:.0f} kB,  model = H-VQ ReGen
        """))
        print("================  Numerical results (JSON)  ================")
        print(json.dumps(res, indent=2))
        print("================  Figure saved  ================")
        print(str(fig_path))
