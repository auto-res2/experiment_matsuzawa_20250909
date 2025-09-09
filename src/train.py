# src/train.py
"""Model architectures, vector-quantisation layers and the ultra-compact
memory container used by H-VQ ReGen.  All classes are extracted verbatim from
src/main.py of the monolithic prototype and only lightly adapted so that they
can be imported from other project files (no logic changes).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BasicBlock",
    "ResNet18Backbone",
    "PenultimateMapper",
    "OrthogonalClassifier",
    "TinyDecoder",
    "VectorQuantizer",
    "HVQMemory",
]

# ---------------------------------------------------------------------------
#  RESNET-18 BACKBONE (slightly simplified to avoid BN / mixed-precision bugs)
# ---------------------------------------------------------------------------


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18Backbone(nn.Module):
    """Standard (imagenette-sized) ResNet-18 that returns the 512-D global-avg-pooled
    feature vector.  Copy/paste from torchvision to make the project fully
    self-contained.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_dim = 512

    # ---------------------------------------------------------------------
    def _make_layer(self, planes: int, blocks: int, stride: int):
        strides = [stride] + [1] * (blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


# ---------------------------------------------------------------------------
#  PENULTIMATE MAPPER + ORTHOGONAL CLASSIFIER
# ---------------------------------------------------------------------------


class PenultimateMapper(nn.Module):
    """Maps 512-D backbone features → 256-D vector used by the task-specific head."""

    def __init__(self, in_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.fc(x)


class OrthogonalClassifier(nn.Module):
    """Classifier with task-specific weight matrices kept mutually orthogonal.

    Each task *t* gets its own weight matrix *W_t*.  During the forward pass the
    corresponding weight is selected via the `task_id` argument.
    """

    def __init__(
        self, feat_dim: int = 256, n_classes_per_task: int = 5, rank: int = 32
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.rank = rank
        self.n_classes_per_task = n_classes_per_task
        self.subspaces: Dict[str, nn.Parameter] = nn.ParameterDict()

    # ------------------------------------------------------------------
    def add_task(self, task_id: int, n_classes: int):
        weight = nn.Parameter(torch.randn(n_classes, self.feat_dim))
        nn.init.orthogonal_(weight)
        self.subspaces[str(task_id)] = weight

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, task_id: int):  # noqa: D401
        key = str(task_id)
        if key not in self.subspaces:
            raise RuntimeError(f"Task {task_id} not initialised in classifier")
        return F.linear(x, self.subspaces[key])

    # ------------------------------------------------------------------
    def gram_schmidt(self):
        """Re-orthogonalise every sub-space in-place (rarely needed)."""

        for w in self.subspaces.values():
            with torch.no_grad():
                q, _ = torch.linalg.qr(w.data.T)
                w.data.copy_(q.T)


# ---------------------------------------------------------------------------
#  DECODER  +  VECTOR-QUANTISER  +  BYTE-LEVEL MEMORY
# ---------------------------------------------------------------------------


class TinyDecoder(nn.Module):
    """Two-layer MLP that reconstructs a 256-D feature vector from discrete codes."""

    def __init__(self, code_dim: int = 32, feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, feat_dim)
        )

    def forward(self, code: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # No gradient flows back through the decoder as per the H-VQ ReGen spec.
        return self.net(code).detach()


class VectorQuantizer(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, commitment_cost: float
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(
            self.embeddings.weight, -1.0 / num_embeddings, 1.0 / num_embeddings
        )

    # ------------------------------------------------------------------
    def forward(self, inputs: torch.Tensor):  # noqa: D401
        flat = inputs.view(-1, self.embedding_dim)
        distances = torch.cdist(flat, self.embeddings.weight)
        encoding_indices = distances.argmin(dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat.dtype)
        quantised = torch.matmul(encodings, self.embeddings.weight).view_as(inputs)

        # Losses
        e_latent = F.mse_loss(quantised.detach(), inputs)
        q_latent = F.mse_loss(quantised, inputs.detach())
        loss = q_latent + self.commitment_cost * e_latent

        # Straight-through estimator
        quantised = inputs + (quantised - inputs).detach()
        return quantised, encoding_indices.view(inputs.shape[0], -1), loss


# ---------------------------------------------------------------------------
#  CONSTANT-FOOTPRINT MEMORY MANAGER
# ---------------------------------------------------------------------------


class HVQMemory:
    """Stores ≤20 bytes per sample (tier-2 code + bookkeeping).

    A FIFO replacement policy is used for simplicity – plug-in a better one
    (e.g. uncertainty-based) if needed.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.bytes_per_sample = 20
        self.capacity = max_bytes // self.bytes_per_sample
        self.storage: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    def add(self, tier2_code: torch.Tensor):
        tier2_code = tier2_code.cpu()
        if len(self.storage) < self.capacity:
            self.storage.append(tier2_code)
        else:
            self.storage.pop(0)
            self.storage.append(tier2_code)

    # ------------------------------------------------------------------
    def sample(self, k: int) -> torch.Tensor:
        idx = torch.randint(0, len(self.storage), (k,))
        return torch.stack([self.storage[i] for i in idx])

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.storage)
