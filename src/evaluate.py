import json
from pathlib import Path

import torch
from torchmetrics import Accuracy

###############################################################################
# Evaluation helpers
###############################################################################

def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_json: str | Path,
):
    """Compute top-1 accuracy (and persist results to JSON)."""

    model.eval()
    acc = Accuracy(task="multiclass", num_classes=len(loader.dataset.features["label"].names)).to(device)

    with torch.no_grad():
        for batch in loader:
            imgs = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(imgs)
            acc.update(logits, labels)

    res = {"top1": float(acc.compute() * 100)}

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)

    # Print to stdout for verification
    print(json.dumps(res, indent=2))

    return res
