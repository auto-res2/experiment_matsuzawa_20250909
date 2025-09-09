"""evaluate.py – lightweight metric helpers & plotting utilities.
All heavy plotting dependencies are imported via `_dimport` so the module can be
imported in a minimal environment.  When matplotlib is not present the plotting
functions become no-ops (they print a short notice instead of raising).
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
#  Lazy imports --------------------------------------------------------------

def _dimport(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:  # pragma: no cover – stub fallback
        import types, sys
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod


plt = _dimport("matplotlib.pyplot")
_dimport("seaborn")  # optional – sets default style when available

torch = _dimport("torch")

# Internal config helper -----------------------------------------------------
_pre = importlib.import_module("src.preprocess" if __name__.startswith("src.") else "preprocess")
get_cfg = _pre.get_cfg

# ---------------------------------------------------------------------------
#  Metrics -------------------------------------------------------------------

def accuracy(logits, labels) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def mu_variance(x) -> float:
    return x.var(dim=0).mean().item()


def dirichlet_energy(x, edge_index) -> float:
    diff = x[edge_index[0]] - x[edge_index[1]]
    return diff.pow(2).sum(dim=-1).mean().item()

# ---------------------------------------------------------------------------
#  Plotting helper -----------------------------------------------------------


def _annotate(ax):
    for line in ax.get_lines():
        x, y = line.get_xdata(), line.get_ydata()
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6)


def plot_depth_sweep(json_path: Path):
    """Create accuracy-versus-depth figures and save them under
    `.research/iteration1/images/` as mandated by the task.  When matplotlib is
    missing, the function degrades gracefully and prints a short notice.
    """
    if not hasattr(plt, "figure"):
        print("[plot_depth_sweep] matplotlib not available – skipping figure generation.")
        return

    with json_path.open() as fh:
        data: Dict = json.load(fh)

    cfg = get_cfg()
    fig_dir = Path(cfg["global"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    depths_cfg = cfg["experiments"]["depth_sweep"]["depths"]

    for dataset, depths in data.items():
        plt.figure(figsize=(6, 4))
        for model_name in next(iter(depths.values())).keys():
            ys = [depths[str(d)][model_name]["accuracy_mean"] for d in depths_cfg]
            plt.plot(depths_cfg, ys, marker="o", label=model_name)
        _annotate(plt.gca())
        plt.xlabel("Depth (layers)")
        plt.ylabel("Accuracy")
        plt.title(f"Depth-sweep accuracy – {dataset}")
        plt.legend()
        fname = fig_dir / f"accuracy_{dataset}.pdf"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
