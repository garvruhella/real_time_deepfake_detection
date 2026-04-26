import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import onnxruntime as ort
import torch

from src.hybrid_common import build_detector_model, build_discriminator_model


def run_onnx(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: x})[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector_pth", default="results_detector/best_detector.pth")
    parser.add_argument("--discriminator_pth", default="results/D_epoch_50.pth")
    parser.add_argument("--detector_onnx", default="extension/models/detector.onnx")
    parser.add_argument("--discriminator_onnx", default="extension/models/discriminator.onnx")
    parser.add_argument("--output", default="results_hybrid/parity_report.json")
    args = parser.parse_args()

    x = torch.randn(4, 3, 128, 128)
    x_np = x.numpy().astype(np.float32)

    detector = build_detector_model()
    detector.load_state_dict(torch.load(args.detector_pth, map_location="cpu"))
    detector.eval()
    with torch.no_grad():
        torch_detector = detector(x).view(-1).numpy()

    discriminator = build_discriminator_model()
    discriminator.load_state_dict(torch.load(args.discriminator_pth, map_location="cpu"))
    discriminator.eval()
    with torch.no_grad():
        torch_discriminator = discriminator(x).view(-1).numpy()

    det_sess = ort.InferenceSession(args.detector_onnx, providers=["CPUExecutionProvider"])
    disc_sess = ort.InferenceSession(args.discriminator_onnx, providers=["CPUExecutionProvider"])
    onnx_detector = run_onnx(det_sess, x_np).reshape(-1)
    onnx_discriminator = run_onnx(disc_sess, x_np).reshape(-1)

    report = {
        "detector_max_abs_diff": float(np.max(np.abs(torch_detector - onnx_detector))),
        "detector_mean_abs_diff": float(np.mean(np.abs(torch_detector - onnx_detector))),
        "discriminator_max_abs_diff": float(np.max(np.abs(torch_discriminator - onnx_discriminator))),
        "discriminator_mean_abs_diff": float(np.mean(np.abs(torch_discriminator - onnx_discriminator))),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
