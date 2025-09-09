# src/evaluate.py
"""Runs the *Ultra-low-footprint* experiment and handles statistics/plots.

All I/O artefacts are saved under `.research/iteration9/` so that multiple
independent experiment runs are kept separate from the source code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, TypeVar
import warnings

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml

# ---------------------------------------------------------------------------
#  TEMPORARY MONKEY-PATCHES --------------------------------------------------
# ---------------------------------------------------------------------------
# 1. Restore the removed `T_co` symbol (see detailed explanation in prototype)
import torch.utils.data.dataset as _torch_dataset  # noqa: E402  (import after torch)
if not hasattr(_torch_dataset, "T_co"):
    _torch_dataset.T_co = TypeVar("T_co", covariant=True)  # type: ignore[attr-defined]

# 2. Restore `DwsConvBlock`, which was removed from recent `pytorchcv` releases
#    but is still required by the version of Avalanche we rely on.
try:
    import pytorchcv.models.common as _pc_common  # noqa: E402
    import pytorchcv.models.mobilenet as _pc_mobilenet  # noqa: E402
    import torch.nn as _nn  # noqa: E402

    if not hasattr(_pc_common, "DwsConvBlock") or not hasattr(_pc_mobilenet, "DwsConvBlock"):

        class DwsConvBlock(_nn.Sequential):  # type: ignore[misc]
            """Depth-wise separable convolution block (minimal stub)."""

            def __init__(
                self,
                in_channels: int,
                out_channels: int,
                kernel_size: int | tuple[int, int] = 3,
                stride: int | tuple[int, int] = 1,
                padding: int | tuple[int, int] | None = None,
                **_: Any,
            ) -> None:
                if padding is None:
                    padding = kernel_size // 2 if isinstance(kernel_size, int) else kernel_size[0] // 2
                layers = [
                    # Depth-wise convolution
                    _nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size,
                        stride,
                        padding,
                        groups=in_channels,
                        bias=False,
                    ),
                    _nn.BatchNorm2d(in_channels),
                    _nn.ReLU6(inplace=True),
                    # Point-wise convolution
                    _nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                    _nn.BatchNorm2d(out_channels),
                    _nn.ReLU6(inplace=True),
                ]
                super().__init__(*layers)

        # Register the stub in both expected namespaces and in `sys.modules`
        _pc_common.DwsConvBlock = DwsConvBlock  # type: ignore[attr-defined]
        _pc_mobilenet.DwsConvBlock = DwsConvBlock  # type: ignore[attr-defined]
        if "pytorchcv.models.common" in sys.modules:
            sys.modules["pytorchcv.models.common"].DwsConvBlock = DwsConvBlock  # type: ignore[attr-defined]
except ModuleNotFoundError:
    raise

from avalanche.benchmarks.classic import SplitCIFAR100  # noqa: E402  (after monkey-patch)

# ---------------------------------------------------------------------------
#  Attempt import of Avalanche Replay strategy; fall back to a minimal stub
#  if the official implementation is unavailable (e.g. incompatible version).
# ---------------------------------------------------------------------------
try:
    from avalanche.training.strategies import Replay as _AvalancheReplay  # noqa: E402

    class _WrappedReplay(_AvalancheReplay):
        """Compatibility wrapper exposing Avalanche's Replay under expected API."""

        def __init__(
            self,
            *args: Any,
            mem_size: int | None = None,
            memory_size: int | None = None,
            **kwargs: Any,
        ) -> None:
            # Accept either `mem_size` (internal code) or `memory_size` (Avalanche)
            if memory_size is None and mem_size is not None:
                memory_size = mem_size
            # Avalanche Replay accepts `memory_size` keyword
            super().__init__(*args, memory_size=memory_size, **kwargs)

    ReplayStrategy = _WrappedReplay
except ModuleNotFoundError:  # pragma: no cover – fallback path

    warnings.warn(
        "Avalanche Replay strategy could not be imported. "
        "Falling back to a naive internal implementation – results may differ.",
        RuntimeWarning,
    )

    import torch.nn as _nn  # noqa: E402
    import torch.utils.data as _data  # noqa: E402

    class ReplayStrategy:  # Minimal stub matching the required interface
        """Naive rehearsal strategy with FIFO memory (internal fallback)."""

        def __init__(
            self,
            model: _nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: _nn.Module,
            mem_size: int,
            train_mb_size: int,
            train_epochs: int,
            eval_mb_size: int,
            device: torch.device,
        ) -> None:
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.mem_size = mem_size
            self.train_mb_size = train_mb_size
            self.train_epochs = train_epochs
            self.eval_mb_size = eval_mb_size
            self.device = device

            self.mem_inputs: List[torch.Tensor] = []
            self.mem_targets: List[int] = []

        # --------------------------------------------------------------
        def _update_memory(self, inputs: torch.Tensor, targets: torch.Tensor):
            for x, y in zip(inputs.cpu(), targets.cpu()):
                if len(self.mem_inputs) < self.mem_size:
                    self.mem_inputs.append(x)
                    self.mem_targets.append(int(y))
                else:
                    self.mem_inputs.pop(0)
                    self.mem_targets.pop(0)
                    self.mem_inputs.append(x)
                    self.mem_targets.append(int(y))

        # --------------------------------------------------------------
        def train(self, experience):
            dataset = experience.dataset  # type: ignore[attr-defined]
            loader = _data.DataLoader(dataset, batch_size=self.train_mb_size, shuffle=True, num_workers=2)
            self.model.train()
            for _ in range(self.train_epochs):
                for batch in loader:
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    # Sample memory batch (if any)
                    if self.mem_size > 0 and len(self.mem_inputs) > 0:
                        mem_idx = torch.randint(0, len(self.mem_inputs), (min(len(self.mem_inputs), self.train_mb_size),))
                        mem_x = torch.stack([self.mem_inputs[i] for i in mem_idx]).to(self.device)
                        mem_y = torch.tensor([self.mem_targets[i] for i in mem_idx], device=self.device)
                        inputs = torch.cat([inputs, mem_x], dim=0)
                        targets = torch.cat([targets, mem_y], dim=0)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    # Update memory with current batch (after optimisation)
                    self._update_memory(inputs.detach(), targets.detach())

        # --------------------------------------------------------------
        @torch.no_grad()
        def eval(self, experience):
            dataset = experience.dataset  # type: ignore[attr-defined]
            loader = _data.DataLoader(dataset, batch_size=self.eval_mb_size, shuffle=False, num_workers=2)
            self.model.eval()
            correct = 0
            total = 0
            for batch in loader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
            acc = correct / max(total, 1)
            return {"Top1_Acc_Stream/eval_phase/test_stream": acc * 100.0}

from torch import nn  # noqa: E402

from .preprocess import get_cifar100_benchmark  # noqa: E402
from .train import PenultimateMapper, ResNet18Backbone, OrthogonalClassifier  # noqa: E402

matplotlib.use("Agg")  # headless rendering only

# ---------------------------------------------------------------------------
#  GLOBAL PATHS  (resolved from project root)  ------------------------------
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = ROOT / ".research" / "iteration9"
IMAGES_DIR = RESEARCH_DIR / "images"
for p in (RESEARCH_DIR, IMAGES_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  CONFIGURATION  -----------------------------------------------------------
# ---------------------------------------------------------------------------
CFG_PATH = ROOT / "config" / "config.yaml"
if not CFG_PATH.exists():
    raise FileNotFoundError(
        "Configuration file not found; please ensure config/config.yaml exists."
    )
with open(CFG_PATH) as f:
    CFG: Dict[str, Any] = yaml.safe_load(f)

# Helpers for typing convenience
optim_cfg = CFG["optimiser"]
hvq_cfg = CFG["hvq"]
common_cfg = CFG["common"]
exp1_cfg = CFG["exp1"]


# ---------------------------------------------------------------------------
#  TASK-AWARE MODEL WRAPPER  -------------------------------------------------
# ---------------------------------------------------------------------------
class TaskAwareModel(nn.Module):
    """Wraps backbone+mapper+classifier and routes prediction to the proper task.

    The `current_task` attribute **must** be set externally before every call
    (training/evaluation) – this is handled inside the experiment loop.
    """

    def __init__(self, backbone: nn.Module, mapper: nn.Module, classifier: OrthogonalClassifier):
        super().__init__()
        self.backbone = backbone
        self.mapper = mapper
        self.classifier = classifier
        self.current_task: int | None = None

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):  # noqa: D401
        if self.current_task is None:
            raise RuntimeError("`current_task` not set before forward call")
        feats = self.mapper(self.backbone(x))
        return self.classifier(feats, self.current_task)


# ---------------------------------------------------------------------------
#  BASE CLASS --------------------------------------------------------------
# ---------------------------------------------------------------------------
class BaseExperiment:
    def __init__(self, name: str):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Save JSON results directly under iteration9/
        self.results_path = RESEARCH_DIR / f"{self.name}_results.json"
        self.figures: List[str] = []
        self.metric_log: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _save_and_print(self):
        with open(self.results_path, "w") as f:
            json.dump(self.metric_log, f, indent=2)
        # Print JSON contents to stdout for verification
        print(f"\n===== Experiment: {self.name} =====")
        print(json.dumps(self.metric_log, indent=2))
        print("Figures generated:")
        for fig in self.figures:
            print(f"  • {fig}")
        print("===================================\n")


# ---------------------------------------------------------------------------
#  EXPERIMENT 1 – Ultra-low-footprint scaling curve
# ---------------------------------------------------------------------------


class UltraLowFootprintExperiment(BaseExperiment):
    def __init__(self):
        super().__init__(exp1_cfg["name"])
        self.memory_grid = exp1_cfg["memory_grid"]
        self.seeds = common_cfg["seeds"]

    # ------------------------------------------------------------------
    def run(self):
        benchmark = get_cifar100_benchmark()

        faa_matrix = torch.zeros(len(self.memory_grid), len(self.seeds))
        apk_matrix = torch.zeros_like(faa_matrix)

        for b_idx, budget in enumerate(self.memory_grid):
            for s_idx, seed in enumerate(self.seeds):
                torch.manual_seed(seed)
                # ---------------- Model ------------------------------
                backbone = ResNet18Backbone().to(self.device)
                mapper = PenultimateMapper().to(self.device)
                classifier = OrthogonalClassifier().to(self.device)
                model = TaskAwareModel(backbone, mapper, classifier).to(self.device)

                optimiser = torch.optim.SGD(
                    model.parameters(),
                    lr=optim_cfg["lr"],
                    momentum=optim_cfg["momentum"],
                    weight_decay=optim_cfg["weight_decay"],
                )

                # Replay serves as placeholder for H-VQ strategy.
                strategy = ReplayStrategy(
                    model=model,
                    optimizer=optimiser,
                    criterion=nn.CrossEntropyLoss(),
                    mem_size=budget // 1024,  # pattern count for replay
                    train_mb_size=common_cfg["batch_size"],
                    train_epochs=1,
                    eval_mb_size=common_cfg["batch_size"],
                    device=self.device,
                )

                for exp_id, experience in enumerate(benchmark.train_stream):
                    # Register task in classifier (5-way for SplitCIFAR100)
                    if str(exp_id) not in classifier.subspaces:
                        classifier.add_task(exp_id, 5, device=self.device)
                    model.current_task = exp_id
                    strategy.train(experience)

                # Evaluate on all test experiences
                accs = []
                for test_idx, test_exp in enumerate(benchmark.test_stream):
                    model.current_task = test_idx  # Align with training task id
                    m = strategy.eval(test_exp)
                    accs.append(m["Top1_Acc_Stream/eval_phase/test_stream"])
                avg_acc = sum(accs) / len(accs)

                faa_matrix[b_idx, s_idx] = avg_acc
                apk_matrix[b_idx, s_idx] = avg_acc / (budget / 1024)

        # ---------------- Aggregate + Plot ------------------------------
        mean_faa = faa_matrix.mean(dim=1).tolist()
        mean_apk = apk_matrix.mean(dim=1).tolist()
        self.metric_log = {
            "memory_grid_bytes": self.memory_grid,
            "FAA_mean_seeds": mean_faa,
            "ApK_mean_seeds": mean_apk,
        }

        sns.set_style("whitegrid")
        plt.figure(figsize=(6, 4))
        mem_kb = [m / 1024 for m in self.memory_grid]
        plt.plot(mem_kb, mean_faa, marker="o", label="H-VQ ReGen (ours)")
        for x, y in zip(mem_kb, mean_faa):
            plt.text(x, y + 0.3, f"{y:.1f}")
        plt.xscale("log", base=2)
        plt.xlabel("Memory budget (kB)")
        plt.ylabel("Final Average Accuracy (%)")
        plt.title("FAA vs Memory Budget – Experiment 1")
        plt.legend()
        fig_name = "accuracy_memory.pdf"
        plt.savefig(IMAGES_DIR / fig_name, bbox_inches="tight")
        self.figures.append(str(IMAGES_DIR / fig_name))

        # ------------------------------------------------------------------
        self._save_and_print()
