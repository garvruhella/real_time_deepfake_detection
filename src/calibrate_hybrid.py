import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets

from src.hybrid_common import (
    compute_hybrid_scores,
    get_eval_transform,
    load_hybrid_models,
    save_hybrid_config,
)


def gather_scores(
    data_root: str,
    split: str,
    detector_path: str,
    discriminator_path: str,
    batch_size: int,
    device: torch.device,
):
    dataset = datasets.ImageFolder(
        root=str(Path(data_root) / split),
        transform=get_eval_transform(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    detector, discriminator = load_hybrid_models(detector_path, discriminator_path, device)

    y_true = []
    detector_fake = []
    disc_fake = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = compute_hybrid_scores(images, detector, discriminator)
            y_true.extend(labels.tolist())
            detector_fake.extend(outputs["detector_fake_prob"].cpu().tolist())
            disc_fake.extend(outputs["disc_fake_prob"].cpu().tolist())

    return torch.tensor(y_true), torch.tensor(detector_fake), torch.tensor(disc_fake)


def evaluate_combination(y_true, detector_fake, disc_fake, alpha, threshold):
    beta = 1.0 - alpha
    hybrid = alpha * detector_fake + beta * disc_fake
    pred = (hybrid >= threshold).int()
    f1 = f1_score(y_true.numpy(), pred.numpy())
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true.numpy(), pred.numpy(), average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true.numpy(), hybrid.numpy())
    return {
        "alpha": alpha,
        "beta": beta,
        "threshold": threshold,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="datasets/detector_data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--detector", default="results_detector/best_detector.pth")
    parser.add_argument("--discriminator", default="results/D_epoch_50.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_metrics", default="results_hybrid/calibration_metrics.json")
    parser.add_argument("--output_config", default="results_hybrid/hybrid_config.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true, detector_fake, disc_fake = gather_scores(
        args.data_root,
        args.split,
        args.detector,
        args.discriminator,
        args.batch_size,
        device,
    )

    best = None
    for alpha_step in range(0, 11):
        alpha = alpha_step / 10.0
        for threshold_step in range(10, 91, 5):
            threshold = threshold_step / 100.0
            metrics = evaluate_combination(
                y_true, detector_fake, disc_fake, alpha=alpha, threshold=threshold
            )
            if best is None or metrics["f1"] > best["f1"]:
                best = metrics

    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metrics).write_text(json.dumps(best, indent=2), encoding="utf-8")
    save_hybrid_config(
        args.output_config,
        detector_weight=best["alpha"],
        discriminator_weight=best["beta"],
        threshold=best["threshold"],
    )
    print(json.dumps(best, indent=2))
    print(f"Saved metrics to {args.output_metrics}")
    print(f"Saved config to {args.output_config}")


if __name__ == "__main__":
    main()
