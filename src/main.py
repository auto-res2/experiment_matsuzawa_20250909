# src/main.py
"""Entry point for the ultra-low-footprint experiments.

Executing `python -m src.main` will run Experiment 1 (Ultra-low-footprint
scaling curve) and save all artefacts – JSON logs and figures – under
`.research/iteration10/`.
"""
from __future__ import annotations

import torch

from .evaluate import UltraLowFootprintExperiment


def main():
    # Initialise CUDA early to avoid fork issues on certain platforms
    _ = torch.cuda.is_available()
    experiment = UltraLowFootprintExperiment()
    experiment.run()


if __name__ == "__main__":
    main()
