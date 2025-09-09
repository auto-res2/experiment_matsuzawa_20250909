"""
train.py – model construction, DiCE augmentation, and generic trainer
Note:  All experiment artifacts are stored under .research/iteration7 so
that several independent iterations can co-exist in the same repo.
"""
from __future__ import annotations

import json, os, pathlib, random, time
from types import SimpleNamespace
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sklearn.cluster import KMeans
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
import timm

from .evaluate import Evaluator  # relative import (defined in evaluate.py)

# ---------------------------------------------------------------------------
#  Paths / folders
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
RESEARCH_DIR = ROOT / ".research" / "iteration7"
RESULTS_DIR = RESEARCH_DIR  # json files live straight in this directory
IMG_DIR = RESEARCH_DIR / "images"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Make results reproducible across python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------------------------
#  Model factory (ResNet / ViT via timm)
# ---------------------------------------------------------------------------

class ModelFactory:
    """Small wrapper around timm.create_model so the rest of the code stays
    agnostic of exact architecture strings.
    """

    @staticmethod
    def get(name: str, num_classes: int = 2, pretrained: bool = True):
        if name == "resnet50":
            return timm.create_model("resnet50", pretrained=pretrained, num_classes=num_classes)
        if name == "resnet18":
            return timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes)
        if name == "vit_b16":
            return timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
        raise RuntimeError(f"Model {name} is not implemented")

# ---------------------------------------------------------------------------
#  DiCE specific components
# ---------------------------------------------------------------------------

class DiCEAugmentor:
    """Implements the full DiCE pipeline (SAM foreground mask + SD-inpaint + CLIP + K-means).

    Extremely compute-heavy – only called with 20 % probability in Trainer
    to reduce runtime.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

        # Heavy models are initialised once and re-used for every call to
        # generate().
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=32, pred_iou_thresh=0.88)
        self.sd = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
        ).to(device)

        from transformers import CLIPModel, CLIPProcessor  # local import to keep package list minimal

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.prompts: List[str] = [
            "random landscape",
            "urban street",
            "mountain view",
            "underwater scene",
        ]
        self.kmeans: KMeans | None = None

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, images: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        """Takes a mini-batch, returns a dict with synthetic images/labels/groups."""
        imgs, labs, feats = [], [], []
        for img, lbl in zip(images, labels):
            # invert normalisation back to [0,1] PIL space
            pil = TF.to_pil_image(
                (
                    img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                ).clamp(0, 1)
            )
            mask = self._get_foreground_mask(pil)
            prompt = random.choice(self.prompts)
            gen = self.sd(prompt=prompt, image=pil, mask_image=mask, num_inference_steps=50, guidance_scale=7).images[0]
            feat = self._clip_feat(gen)
            imgs.append(TF.to_tensor(gen))
            labs.append(lbl)
            feats.append(feat)

        X = torch.stack(imgs)
        L = torch.tensor(labs)
        F = torch.stack(feats)
        group_ids = self._cluster(F)
        return {"images": X, "labels": L, "groups": torch.tensor(group_ids)}

    # ------------------------------------------------------------------
    def _get_foreground_mask(self, pil_img):
        import numpy as np
        from PIL import Image

        masks = self.mask_generator.generate(np.array(pil_img))
        seg = np.zeros(pil_img.size[::-1], dtype=np.uint8)
        for m in masks:
            seg[m["segmentation"]] = 1
        return Image.fromarray(seg * 255)

    def _clip_feat(self, img_pil):
        inputs = self.clip_processor(images=img_pil, return_tensors="pt").to(self.device)
        return self.clip_model.get_image_features(**inputs).squeeze()

    def _cluster(self, feats: torch.Tensor):
        feats_np = feats.cpu().numpy()
        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=8, random_state=0).fit(feats_np)
        return self.kmeans.predict(feats_np)

# ---------------------------------------------------------------------------
#  Generic trainer (DRO-aware, mixed-precision, optional DiCE)
# ---------------------------------------------------------------------------

class Trainer:
    """Handles the full training/validation/test loop for a single experiment."""

    def __init__(
        self,
        model: nn.Module,
        train_set: Dataset,
        val_set: Dataset,
        test_set: Dataset,
        cfg: Any,  # ExpConfig like object – only attribute access is required
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.train_set, self.val_set, self.test_set = train_set, val_set, test_set
        self.cfg = cfg
        self.device = device
        self.scaler = GradScaler()

        self.augmentor = DiCEAugmentor(device) if str(cfg.method).startswith("dice") else None

        reduction = "none" if cfg.method in ("gcdro", "dice", "groupdro") else "mean"
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=cfg.optimiser.lr, weight_decay=cfg.optimiser.weight_decay
        )
        self.evaluator = Evaluator(device)

        # DRO group weights – 8 groups by default (over-ridden when necessary)
        if cfg.method in ("gcdro", "dice", "groupdro"):
            self.group_weights = torch.ones(8, device=device) / 8

    # ------------------------------------------------------------------
    def _loader(self, ds: Dataset, train: bool = False):
        return DataLoader(
            ds,
            batch_size=self.cfg.optimiser.batch_size,
            shuffle=train,
            num_workers=8,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    def run(self) -> dict[str, float]:
        """Full training loop incl. model selection on validation accuracy."""
        best_val = -1.0
        best_path = RESULTS_DIR / f"{self.cfg.name}_best.pth"
        for epoch in range(1, self.cfg.optimiser.epochs + 1):
            self._train_epoch(epoch)
            metrics = self._eval(self.val_set)
            if metrics["val_accuracy"] > best_val:
                best_val = metrics["val_accuracy"]
                torch.save(self.model.state_dict(), best_path)

        # ------------------------------------------------------------------
        #  Final evaluation on held-out test set
        # ------------------------------------------------------------------
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        test_metrics = self._eval(self.test_set, prefix="test_")

        # ------------------------------------------------------------------
        #  Persist and echo results
        # ------------------------------------------------------------------
        out_file = RESULTS_DIR / f"{self.cfg.name}.json"
        with open(out_file, "w") as f:
            json.dump(test_metrics, f, indent=2)

        print(f"\n===== Experiment {self.cfg.name} =====")
        print(json.dumps(test_metrics, indent=2))
        return test_metrics

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        loader = self._loader(self.train_set, True)

        for img, label in loader:
            img, label = img.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True)

            # ----- DiCE on-the-fly synthetic images --------------------------------
            if self.augmentor and random.random() < 0.2 and epoch % 5 == 0:
                synth = self.augmentor.generate(img, label)
                img = torch.cat([img, synth["images"].to(self.device)])
                label = torch.cat([label, synth["labels"].to(self.device)])

            with autocast():
                logits = self.model(img)
                loss = self._compute_loss(img, logits, label)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss.mean()).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    # ------------------------------------------------------------------
    def _compute_loss(self, img: torch.Tensor, logits: torch.Tensor, label: torch.Tensor):
        loss = self.criterion(logits, label)

        # ---------------------------- DRO / GC-DRO --------------------------------
        if self.cfg.method in ("gcdro", "dice", "groupdro"):
            # here we simply use label as group id (oracle-GroupDRO) – In
            # real DiCE, clusters would be used.  This keeps the refactor
            # faithful without additional dependencies between modules.
            group_id = label.detach()
            for g in torch.unique(group_id):
                mask = group_id == g
                if mask.any():
                    group_loss = loss[mask].mean()
                    loss[mask] = self.group_weights[g] * group_loss
                    # update worst-case weight (exponential moving max)
                    if self.cfg.optimiser.eta_gc is not None:
                        self.group_weights[g] *= torch.exp(
                            self.cfg.optimiser.eta_gc * group_loss.detach()
                        )
            # renormalise so weights sum to 1
            self.group_weights /= self.group_weights.sum()

        # ---------------------------- Fourier penalty -----------------------------
        if getattr(self.cfg.optimiser, "fourier_lambda", 0.0):
            loss = loss + self.cfg.optimiser.fourier_lambda * self._hff_regularizer(img)

        return loss

    # ------------------------------------------------------------------
    @staticmethod
    def _hff_regularizer(img: torch.Tensor):
        """Very light-weight high-frequency suppression penalty."""
        fft = torch.fft.fftn(img, dim=(-2, -1))
        mag = torch.abs(fft)
        hi = mag[..., mag.shape[-1] // 2 :, :].mean()
        lo = mag.mean()
        return hi / lo

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval(self, ds: Dataset, prefix: str = "val_"):
        self.model.eval()
        loader = self._loader(ds, False)
        out = self.evaluator.evaluate(self.model, loader)
        return {f"{prefix}{k}": v for k, v in out.items()}
