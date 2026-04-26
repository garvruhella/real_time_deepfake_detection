import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, models
from tqdm import tqdm

from src.hybrid_common import IMAGE_SIZE
from src.train_utils import (
    build_eval_transforms,
    build_train_transforms,
    evaluate_split,
    find_eval_split_path,
    smooth_binary_labels,
)


def set_backbone_requires_grad(model: nn.Module, requires: bool) -> None:
    for name, p in model.named_parameters():
        if name.startswith("fc"):
            continue
        p.requires_grad = requires


def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet18 deepfake detector (anti-overfit defaults)."
    )
    parser.add_argument(
        "--data_root",
        default="datasets/detector_data",
        help="Root with train/ and val/ or test/ subfolders (ImageFolder).",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--label_smoothing", type=float, default=0.08)
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=2,
        help="Epochs to train only the classification head (backbone frozen).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Stop if val AUC does not improve for this many epochs.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--strong_aug",
        action="store_true",
        default=True,
        help="Use RandomResizedCrop + blur (default: on).",
    )
    parser.add_argument(
        "--no_strong_aug",
        action="store_false",
        dest="strong_aug",
    )
    parser.add_argument(
        "--out",
        default="results_detector/best_detector.pth",
        help="Path to save best weights (val AUC).",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | IMAGE_SIZE: {IMAGE_SIZE}")
    eval_name, eval_path = find_eval_split_path(args.data_root)
    print(f"Using evaluation split: '{eval_name}' at {eval_path}")

    tr_tf = build_train_transforms(strong=args.strong_aug)
    ev_tf = build_eval_transforms()
    train_ds = datasets.ImageFolder(
        os.path.join(args.data_root, "train"), transform=tr_tf
    )
    val_ds = datasets.ImageFolder(os.path.join(args.data_root, eval_name), transform=ev_tf)
    print("Class -> index:", train_ds.class_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 1),
    )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_auc = -1.0
    patience = 0

    for epoch in range(args.epochs):
        if epoch < args.freeze_epochs:
            set_backbone_requires_grad(model, False)
            for p in model.fc.parameters():
                p.requires_grad = True
        else:
            set_backbone_requires_grad(model, True)

        model.train()
        train_loss = 0.0
        t_correct = 0
        t_total = 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=100)

        for images, labels in bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
            targets = smooth_binary_labels(
                labels, float(args.label_smoothing)
            ).to(device)

            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            with torch.no_grad():
                p = torch.sigmoid(outputs)
                pred = (p > 0.5).float()
                t_correct += (pred == labels).sum().item()
                t_total += labels.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")

        tr_acc = 100.0 * t_correct / max(t_total, 1)
        vmet = evaluate_split(
            model, val_loader, device, use_amp=(device.type == "cuda")
        )
        scheduler.step()

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss/len(train_loader):.4f} "
            f"train_acc={tr_acc:.2f}% | val_loss={vmet['loss']:.4f} "
            f"val_acc={vmet['acc']:.2f}% val_auc={vmet['auc']}"
        )

        vauc = vmet["auc"]
        if not (vauc == vauc) or vauc is None:  # nan
            vauc = -1.0

        if vauc > best_auc + 1e-6:
            best_auc = vauc
            torch.save(model.state_dict(), args.out)
            print(f"Saved best (val_auc={best_auc:.4f}) -> {args.out}")
            patience = 0
        else:
            patience += 1
            print(f"No val AUC improve ({patience}/{args.patience})")
            if patience >= args.patience:
                print("Early stop.")
                break

    if best_auc < 0 and os.path.isfile(args.out):
        print(f"Best weights at {args.out} (AUC was nan at times; re-check data).")
    else:
        print(f"Done. Best val_auc ~ {best_auc:.4f}  checkpoint: {args.out}")


if __name__ == "__main__":
    main()
