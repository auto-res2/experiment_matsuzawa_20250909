import yaml

from .evaluate import (
    experiment1_ci_gate,
    experiment2_memory_accuracy,
    experiment3_long_horizon,
)
from .train import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Load configuration (single YAML file for the whole project)
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
with open(CONFIG_PATH) as fp:
    CFG = yaml.safe_load(fp)


def main():
    # Experiment 1
    experiment1_ci_gate(CFG["global"], CFG["exp1_ci_gate"])

    # Experiment 2
    experiment2_memory_accuracy(CFG["global"], CFG["exp2_memory_accuracy"])

    # Experiment 3
    experiment3_long_horizon(CFG["global"], CFG["exp3_long_horizon"])


if __name__ == "__main__":
    main()
