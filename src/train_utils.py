"""Shared transforms and metrics for detector training (reduce overfitting / domain gap)."""

import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from src.hybrid_common import IMAGE_SIZE, NORM_MEAN, NORM_STD


def build_train_transforms(strong: bool = True) -> transforms.Compose:
    if strong:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    IMAGE_SIZE, scale=(0.78, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(12),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03
                ),
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=3, sigma=(0.1, 1.2)
                        )
                    ],
                    p=0.2,
                ),
                transforms.ToTensor(),
                transforms.Normalize(NORM_MEAN, NORM_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def build_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def smooth_binary_labels(y: torch.Tensor, eps: float) -> torch.Tensor:
    """0 -> eps/2, 1 -> 1 - eps/2. Reduces overconfident logits."""
    y = y.float()
    return y * (1.0 - eps / 2.0) + (1.0 - y) * (eps / 2.0)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    all_labels = []
    all_scores = []
    use_cuda_amp = use_amp and device.type == "cuda"
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_f = labels.float().unsqueeze(1).to(device, non_blocking=True)
        if use_cuda_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(images)
        else:
            out = model(images)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out, labels_f, reduction="mean"
        )
        p = torch.sigmoid(out)
        pred = (p > 0.5).float()
        correct += (pred == labels_f).sum().item()
        total += labels_f.size(0)
        losses.append(loss.item())
        all_labels.append(labels_f.detach().float().cpu().numpy().ravel())
        all_scores.append(p.detach().float().cpu().numpy().ravel())

    y = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)
    s = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
    acc = 100.0 * correct / max(total, 1)
    if len(np.unique(y)) < 2:
        auc = float("nan")
    else:
        try:
            auc = float(roc_auc_score(y, s))
        except ValueError:
            auc = float("nan")
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": acc,
        "auc": auc,
    }


def find_eval_split_path(data_root: str) -> Tuple[str, str]:
    """Return (subfolder name, full path) for val data: prefer 'val' then 'test'."""
    for name in ("val", "test"):
        p = os.path.join(data_root, name)
        if not os.path.isdir(p):
            continue
        try:
            class_dirs = [
                d
                for d in os.listdir(p)
                if os.path.isdir(os.path.join(p, d)) and not d.startswith(".")
            ]
        except OSError:
            continue
        if class_dirs:
            return name, p
    raise FileNotFoundError(
        f"No 'val' or 'test' split with class subfolders under: {data_root}"
    )
