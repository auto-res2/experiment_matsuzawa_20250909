# src/main.py
"""Entry-point script – read YAML config, run experiment, save artefacts."""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from typing import Any, Dict

import yaml

from .train import run_experiment

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _parse_config(path: str) -> SimpleNamespace:
    """Load YAML config at *path* → SimpleNamespace for convenient dot access."""
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict: Dict[str, Any] = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diffusion-based Counterfactual Debiasing")
    parser.add_argument(
        "--config",
        "-c",
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    return parser


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – standard main function
    parser = _build_parser()
    args = parser.parse_args()

    cfg = _parse_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
