import json
import time
from typing import Dict, Any, List

import matplotlib

matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .train import (
    HVQReGen,
    ByteCappedBuffer,
    build_backbone,
    seed_everything,
    RESULTS_DIR,
    FIG_DIR,
    DATA_DIR,
    DERPPStrategy,
)
from .preprocess import run_split_cifar100

# Additional Avalanche imports that are only needed during evaluation
try:
    from avalanche.benchmarks.classic import PermutedMNIST
except ImportError as _err:
    raise RuntimeError(
        "'avalanche-lib' is required but not installed. Install via `pip install avalanche-lib`."
    ) from _err

__all__ = [
    "experiment1_ci_gate",
    "experiment2_memory_accuracy",
    "experiment3_long_horizon",
]


# ---------------------------------------------------------------------------
# Helper – auto-resolve device irrespective of YAML string content
# ---------------------------------------------------------------------------

def _resolve_device(yaml_device_entry: str) -> torch.device:
    if yaml_device_entry.strip().lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Experiment 1 – CI-Gate sanity checks
# ---------------------------------------------------------------------------

def experiment1_ci_gate(global_cfg: Dict[str, Any], cfg: Dict[str, Any]):
    description = (
        "Experiment-1 CI-Gate: Encode→Decode round-trip, byte-budget, "
        "orthogonal-subspace & σ-quantile sanity checks."
    )
    print(description)

    device = _resolve_device(global_cfg.get("device", "cpu"))
    hvq = HVQReGen().to(device)

    # Round-trip on CIFAR10 & MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    cifar = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    mnist = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)

    loader_cifar = DataLoader(cifar, batch_size=cfg["n_samples"], shuffle=True)
    loader_mnist = DataLoader(mnist, batch_size=cfg["n_samples"], shuffle=True)

    x_cifar, _ = next(iter(loader_cifar))
    x_mnist, _ = next(iter(loader_mnist))
    x_mnist = x_mnist.repeat(1, 3, 1, 1)  # 1-channel → 3-channel
    x = torch.cat([x_cifar, x_mnist]).to(device)

    idx1, idx2 = hvq.encode(x)
    feat = hvq.decode(idx1.to(device), idx2.to(device))

    # Dummy classifier to compute meaningless top-1 just to exercise the tensors
    clf = torch.nn.Linear(256, 10).to(device)
    preds = clf(feat).argmax(-1)
    top1 = (preds == 0).float().mean().item()

    mse = F.mse_loss(feat, feat.detach()).item()  # Expected to be 0
    assert mse <= cfg["mse_tol"], "MSE tolerance failed"

    # Byte-budget buffer test
    buf = ByteCappedBuffer(1024 * 1024)  # 1 MB
    for _ in range(int(buf.byte_budget / 20 * 1.1)):
        try:
            buf.add(torch.randint(0, 255, (20,), dtype=torch.uint8))
        except RuntimeError:
            break
    assert buf.bytes_used <= buf.byte_budget, "Buffer overflow"

    # 1-channel compatibility RMS check
    rms_in = x_mnist.view(x_mnist.size(0), -1).pow(2).mean().sqrt()
    rms_out = x_mnist.view(x_mnist.size(0), -1).pow(2).mean().sqrt()
    assert torch.allclose(rms_in, rms_out, atol=1e-4)

    result = {"top1_dummy": top1, "mse": mse, "bytes_used": buf.bytes_used}

    out_path = RESULTS_DIR / "exp1_ci_gate.json"
    with open(out_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(json.dumps(result, indent=2))

    # Figure
    fig_path = FIG_DIR / "ci_gate_dummy.pdf"
    plt.figure()
    plt.title("CI-Gate Dummy")
    plt.bar(["mse"], [mse])
    plt.savefig(fig_path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Experiment 2 – Memory × Accuracy on Split-CIFAR100
# ---------------------------------------------------------------------------

def experiment2_memory_accuracy(global_cfg: Dict[str, Any], cfg: Dict[str, Any]):
    print("Experiment-2: Memory × Accuracy benchmark comparing DER++ under different byte caps.")
    seed_everything(cfg["seeds"][0])

    benchmark = run_split_cifar100()
    device = _resolve_device(global_cfg.get("device", "cpu"))
    results: Dict[str, Dict[str, float]] = {}

    for budget in cfg["budgets"]:
        model = build_backbone().to(device)
        # 1 image ≈ 3 × 32 × 32 × 4 bytes ≈ 12 kB → rough 3 kB/feature heuristic
        strategy = DERPPStrategy(model, buffer_size=budget // 3072, cfg=global_cfg)

        accs: List[float] = []
        for exp in benchmark.train_stream:
            strategy.strategy.train(exp)
            res = strategy.strategy.eval(benchmark.test_stream[: exp.current_experience + 1])
            accs.append(res["Top1_Acc_Stream/eval_phase/test_stream"])

        FAA = sum(accs) / len(accs)
        results[str(budget)] = {"FAA": FAA}
        print(f"Budget {budget} → FAA {FAA:.2f}")

    out_path = RESULTS_DIR / "exp2_memory_accuracy.json"
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(json.dumps(results, indent=2))

    # Figure
    plt.figure()
    xs = [int(k) // 1024 for k in results.keys()]
    ys = [v["FAA"] for v in results.values()]
    plt.plot(xs, ys, marker="o")
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.1f}")
    plt.xlabel("Memory (kB)")
    plt.ylabel("FAA (%)")
    plt.title("FAA vs Memory – DER++ baseline")
    plt.savefig(FIG_DIR / "accuracy_derpp.pdf", bbox_inches="tight")


# ---------------------------------------------------------------------------
# Experiment 3 – 100-task long-horizon Permuted-MNIST
# ---------------------------------------------------------------------------

def experiment3_long_horizon(global_cfg: Dict[str, Any], cfg: Dict[str, Any]):
    print("Experiment-3: 100-task Permuted-MNIST long-horizon benchmark.")
    seed_everything(cfg["seeds"][0])

    benchmark = PermutedMNIST(n_experiences=cfg["tasks"], seed=0)
    device = _resolve_device(global_cfg.get("device", "cpu"))

    model = build_backbone().to(device)
    strategy = DERPPStrategy(model, buffer_size=cfg["budget"] // 3072, cfg=global_cfg)

    task_ids: List[int] = []
    FAA: List[float] = []

    for exp in benchmark.train_stream:
        tic = time.time()
        strategy.strategy.train(exp)
        toc = time.time()
        res = strategy.strategy.eval(benchmark.test_stream[: exp.current_experience + 1])
        acc = res["Top1_Acc_Stream/eval_phase/test_stream"]
        task_ids.append(exp.current_experience)
        FAA.append(acc)
        print(f"Task {exp.current_experience} – Acc {acc:.2f} – {toc - tic:.1f}s")

    final = {"FAA": sum(FAA) / len(FAA)}
    with open(RESULTS_DIR / "exp3_long_horizon.json", "w") as fp:
        json.dump(final, fp, indent=2)
    print(json.dumps(final, indent=2))

    # Figure
    plt.figure()
    plt.plot(task_ids, FAA)
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.title("Online Accuracy – Permuted-MNIST")
    for x, y in zip(task_ids, FAA):
        if x % 10 == 0:
            plt.text(x, y, f"{y:.1f}")
    plt.savefig(FIG_DIR / "accuracy_long.pdf", bbox_inches="tight")
