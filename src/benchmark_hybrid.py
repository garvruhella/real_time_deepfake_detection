import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from src.hybrid_common import compute_hybrid_scores, load_hybrid_models


def percentile(values, pct):
    idx = max(0, min(len(values) - 1, int(len(values) * pct)))
    return values[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="results_detector/best_detector.pth")
    parser.add_argument("--discriminator", default="results/D_epoch_50.pth")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", default="results_hybrid/latency_report.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, discriminator = load_hybrid_models(args.detector, args.discriminator, device)

    for _ in range(args.warmup):
        x = torch.randn(args.batch_size, 3, 128, 128, device=device)
        with torch.no_grad():
            _ = compute_hybrid_scores(x, detector, discriminator)

    times = []
    for _ in range(args.iterations):
        x = torch.randn(args.batch_size, 3, 128, 128, device=device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = compute_hybrid_scores(x, detector, discriminator)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    times.sort()
    report = {
        "device": str(device),
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "mean_ms": sum(times) / len(times),
        "p50_ms": percentile(times, 0.50),
        "p95_ms": percentile(times, 0.95),
        "p99_ms": percentile(times, 0.99),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
