import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from src.hybrid_common import build_detector_model, build_discriminator_model


def export_onnx(model, input_tensor, output_path: str, dynamic_batch: bool = True):
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    torch.onnx.export(
        model,
        input_tensor,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="results_detector/best_detector.pth")
    parser.add_argument("--discriminator", default="results/D_epoch_50.pth")
    parser.add_argument("--out_dir", default="extension/models")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector_model()
    detector.load_state_dict(torch.load(args.detector, map_location="cpu"))
    detector.eval()

    discriminator = build_discriminator_model()
    discriminator.load_state_dict(torch.load(args.discriminator, map_location="cpu"))
    discriminator.eval()

    dummy = torch.randn(1, 3, 128, 128)
    export_onnx(detector, dummy, str(out_dir / "detector.onnx"))
    export_onnx(discriminator, dummy, str(out_dir / "discriminator.onnx"))

    print(f"Exported ONNX models to {out_dir}")


if __name__ == "__main__":
    main()
