#
# Trains a ResNet18 on datasets/final_data (train/val) from prepare_training_data.py.
# Prefer src/train_detector.py for the same task; this file keeps compatibility with
# best_classifier.pth and api_server.py when you use --mlp_head.
#
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
from tqdm import tqdm

from src.train_utils import (
    build_eval_transforms,
    build_train_transforms,
    evaluate_split,
)


def make_head(
    in_features: int, mlp_head: bool, dropout: float
) -> nn.Module:
    if mlp_head:
        return nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))


def main():
    parser = argparse.ArgumentParser(
        description="ResNet18 classifier (final_data); AUC early stopping."
    )
    parser.add_argument("--data_dir", default="datasets/final_data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--label_smoothing", type=float, default=0.08)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--mlp_head", action="store_true", help="128-dim MLP (matches old api_server).")
    parser.add_argument("--head_dropout", type=float, default=0.4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out", default="best_classifier.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tr = build_train_transforms(strong=True)
    ev = build_eval_transforms()

    train_set = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"), transform=tr
    )
    val_set = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=ev)
    print("Classes / idx:", train_set.class_to_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = make_head(
        model.fc.in_features, mlp_head=args.mlp_head, dropout=args.head_dropout
    )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    from src.train_utils import smooth_binary_labels

    best_auc = -1.0
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        t_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device, non_blocking=True)
            labels = (
                labels.to(device, non_blocking=True).float().unsqueeze(1)
            )
            targs = smooth_binary_labels(labels, args.label_smoothing).to(device)
            optimizer.zero_grad()
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(images)
                    loss = criterion(out, targs)
            else:
                out = model(images)
                loss = criterion(out, targs)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        sched.step()
        vmet = evaluate_split(
            model, val_loader, device, use_amp=(device.type == "cuda")
        )
        print(
            f"Epoch {epoch+1} train_loss={t_loss/len(train_loader):.4f} "
            f"val_auc={vmet['auc']} val_acc={vmet['acc']:.2f}%"
        )
        vauc = vmet["auc"]
        if vauc == vauc and vauc is not None and vauc > best_auc + 1e-6:
            best_auc = vauc
            torch.save(model.state_dict(), args.out)
            print(f"Saved {args.out}  val_auc={vauc:.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stop.")
                break
    print("Training complete. Best val_auc:", best_auc, "->", args.out)
    if args.mlp_head:
        print("Head matches two-layer MLP in api_server.py (fc -> 128 -> 1).")
    else:
        print("For api_server.py, set model.fc to the same one-layer+dropout head or use this script with --mlp_head.")


if __name__ == "__main__":
    main()
