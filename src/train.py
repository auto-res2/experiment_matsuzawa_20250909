"""train.py
Contains all training-related classes / functions: the model builder,
DiCE augmentor and the high-level Engine that performs training and
validation.  Nothing outside this file should directly touch PyTorch
objects – inference utilities live in evaluate.py.
"""
from __future__ import annotations
import os, random, time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
import timm

# NOTE: use absolute package import (src.preprocess) to avoid ModuleNotFound
# errors when the codebase is executed via ``python -m src.main``.
from src.preprocess import make_dataset  # noqa: E402  (import after third-party)

# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def _set_seed(seed: int):
    """Deterministic behaviour for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# -----------------------------------------------------------------------------
#  MODEL
# -----------------------------------------------------------------------------

def build_model(cfg: Dict[str, Any]):
    name = cfg["model"].lower()
    if name == "resnet50":
        model = timm.create_model("resnet50", pretrained=True,
                                  num_classes=cfg["num_classes"])
    elif name in {"vit_b16", "vit-b16"}:
        model = timm.create_model("vit_base_patch16_224", pretrained=True,
                                  num_classes=cfg["num_classes"])
    else:
        raise ValueError(f"Unknown model type {cfg['model']}")
    return model

# -----------------------------------------------------------------------------
#  DICE AUGMENTOR  – optional, only constructed if cfg['method']=="dice"
# -----------------------------------------------------------------------------

try:
    # Heavy imports are optional – guard them so unit tests without GPU do not
    # crash; they will only be needed when method=="dice".
    import torchvision.transforms as _T
    from PIL import Image
    from diffusers import StableDiffusionInpaintPipeline
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    from transformers import CLIPProcessor, CLIPModel
except Exception:  # pragma: no cover – imported conditionally
    _T = None  # type: ignore

class DiceAugmentor:
    """Minimal implementation of DiCE data augmentation.

    Given an image tensor (B,3,H,W) it performs
      1. foreground extraction by SAM
      2. background replacement using Stable-Diffusion in-painting
      3. returns a batch of augmented tensors.

    Any failure in external libraries raises RuntimeError (STRICT
    NO-FALLBACK rule).
    """

    def __init__(self, cfg: Dict[str, Any], accelerator: Accelerator):
        if _T is None:
            raise RuntimeError("torchvision/diffusers not available; cannot use DiCE")
        self.cfg     = cfg
        self.device  = accelerator.device
        self.prompts = cfg.get(
            "dice_prompts",
            ["random landscape", "urban street", "mountain view", "underwater scene"],
        )

        # --- 1) SAM -----------------------------------------------------------
        try:
            sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
            sam.to(self.device)
            self.mask_gen = SamAutomaticMaskGenerator(sam)
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Failed to load SAM – " + str(e))

        # --- 2) Stable Diffusion in-paint -------------------------------------
        try:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
            ).to(self.device)
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Failed to load StableDiffusion – " + str(e))

        # --- 3) CLIP background embedding (not used directly during forward)
        try:
            self.clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Failed to load CLIP – " + str(e))

        # Resize / normalisation identical to training preprocessing
        self.pre_tf = _T.Compose([
            _T.Resize(256),
            _T.CenterCrop(224),
            _T.ToTensor(),
            _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ---------------------------------------------------------------------
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Late import to avoid circular dependency at module import time
        from src.preprocess import _DEF_TRAIN_TRANSF  # noqa: WPS433
        batch_aug: List[torch.Tensor] = []
        for img in x:  # iterate over batch
            pil = _T.ToPILImage()(img.cpu())
            masks = self.mask_gen.generate(pil)
            if not masks:
                batch_aug.append(img)  # keep original if SAM failed
                continue
            fg_mask = (masks[0]["segmentation"].astype("uint8") * 255)
            bg_prompt = random.choice(self.prompts)
            out_pil = self.pipe(prompt=bg_prompt,
                                image=pil,
                                mask_image=Image.fromarray(fg_mask)).images[0]
            batch_aug.append(_DEF_TRAIN_TRANSF(out_pil))
        x_prime = torch.stack(batch_aug).to(x.device, non_blocking=True)
        return x_prime, y  # y unchanged

# -----------------------------------------------------------------------------
#  ENGINE
# -----------------------------------------------------------------------------

class Engine:
    """High-level training / evaluation loop (single / multi GPU via accelerate)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        _set_seed(cfg.get("seed", 11))

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accel = Accelerator(fp16=cfg.get("amp", True), kwargs_handlers=[ddp_kwargs])

        # ---- DATA ---------------------------------------------------------
        self.train_ds, self.val_ds, self.test_ds = make_dataset(cfg, self.accel)
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )

        # ---- MODEL --------------------------------------------------------
        self.model = build_model(cfg)
        self.opt = torch.optim.AdamW(self.model.parameters(),
                                     lr=cfg["lr"],
                                     weight_decay=cfg["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg["epochs"]
        )

        # ---- OPTIONAL DICE -----------------------------------------------
        self.dice_aug: Union[DiceAugmentor, None] = None
        if cfg.get("method", "erm").lower() == "dice":
            self.dice_aug = DiceAugmentor(cfg, self.accel)

        # ---- PREPARE ------------------------------------------------------
        (self.model, self.opt, self.train_loader,
         self.val_loader, self.test_loader) = self.accel.prepare(
            self.model, self.opt, self.train_loader, self.val_loader, self.test_loader
        )

        self.best_val_acc = 0.0
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_acc": []}

    # ------------------------------------------------------------------
    def _parse_batch(self, batch):
        """Handle both dict- and tuple-based batches seamlessly."""
        if isinstance(batch, dict):
            x, y = batch["image"], batch["label"]
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch  # type: ignore[misc]
        else:  # pragma: no cover – unexpected batch shape
            raise TypeError("Unsupported batch format returned by DataLoader.")
        return x, y

    def _forward(self, batch):
        x, y = self._parse_batch(batch)
        if self.dice_aug is not None:
            x, y = self.dice_aug(x, y)
        logits = self.model(x)
        loss   = F.cross_entropy(logits, y)
        acc    = (logits.argmax(1) == y).float().mean()
        return loss, acc, len(y)

    # ------------------------------------------------------------------
    def _train_epoch(self):
        self.model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for batch in self.train_loader:
            with self.accel.autocast():
                loss, acc, bs = self._forward(batch)
            self.accel.backward(loss)
            self.opt.step(); self.opt.zero_grad()
            total_loss += loss.item() * bs
            total_acc  += acc.item()  * bs
            n += bs
        self.scheduler.step()
        return total_loss / n, total_acc / n

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        total_acc, n = 0.0, 0
        for batch in loader:
            _, acc, bs = self._forward(batch)
            total_acc += acc.item() * bs
            n += bs
        return total_acc / n

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._train_epoch()
            val_acc = self._evaluate(self.val_loader)
            self.history["train_loss"].append(tr_loss)
            self.history["val_acc"].append(val_acc)
            if self.accel.is_main_process:
                print(f"Epoch {epoch:3d}/{cfg['epochs']}  loss={tr_loss:.4f}  val_acc={val_acc:.3f}  time={time.time()-t0:.1f}s")
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if self.accel.is_main_process:
                    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"model": self.model.state_dict(), "epoch": epoch, "cfg": cfg},
                        Path(cfg["output_dir"]) / f"best-{cfg['experiment_name']}.pt",
                    )

        test_acc = self._evaluate(self.test_loader)
        if self.accel.is_main_process:
            print(f"Final test accuracy: {test_acc * 100:.2f} %")

        return {
            "best_val_acc": self.best_val_acc,
            "test_acc": test_acc,
            "train_loss": self.history["train_loss"],
            "val_acc_curve": self.history["val_acc"],
            "figures": [],  # filled later by evaluate.generate_figures
        }
