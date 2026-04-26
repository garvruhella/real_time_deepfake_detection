# On-Device Hybrid Deepfake Detector (CED-DCGAN + FF++)

This repository now includes a full hybrid pipeline:

- Calibration/fusion: `src/calibrate_hybrid.py`
- Browser export: `src/export_browser_models.py`
- Export parity check: `src/verify_export_parity.py`
- Latency benchmark: `src/benchmark_hybrid.py`
- MV3 extension runtime: `extension/` (content script + service worker + overlays)

## 1) Train models

- Train detector: `python src/train_detector.py`
- Train CED-DCGAN: `python src/train_ced_dcgan.py`

Expected checkpoints:

- `results_detector/best_detector.pth`
- `results/D_epoch_50.pth` (or another chosen epoch)

## 2) Calibrate hybrid score

`python src/calibrate_hybrid.py --data_root datasets/detector_data --split test`

Outputs:

- `results_hybrid/calibration_metrics.json`
- `results_hybrid/hybrid_config.json`

Copy config to extension:

- `copy results_hybrid\hybrid_config.json extension\models\hybrid_config.json`

## 3) Export ONNX models

`python src/export_browser_models.py --detector results_detector/best_detector.pth --discriminator results/D_epoch_50.pth`

Outputs:

- `extension/models/detector.onnx`
- `extension/models/discriminator.onnx`

## 3.1) Bundle ONNX Runtime Web assets

From project root:

- `npm init -y`
- `npm install onnxruntime-web`

Then copy browser runtime files from `node_modules/onnxruntime-web/dist/` to:

- `extension/vendor/onnxruntime-web/`

Required at minimum:

- `ort.min.mjs`
- `ort-wasm-simd.wasm`
- `ort-wasm.wasm`

## 4) Verify ONNX parity

`python src/verify_export_parity.py`

Output:

- `results_hybrid/parity_report.json`

## 5) Run latency benchmarks

Offline PyTorch benchmark:

- `python src/benchmark_hybrid.py --batch_size 1 --iterations 200`

In-browser benchmark:

- Open a page after loading the extension.
- In DevTools console (tab context), run:
  - `window.__deepfakeDetectorBenchmark?.()`

## 6) Load extension

1. Open `chrome://extensions`
2. Enable Developer mode
3. Load unpacked -> select `extension/`
4. Open popup:
   - Enable realtime scanning
   - Set performance mode (`fast`, `balanced`, `quality`)
5. Use "Rescan tab now" for immediate pass

## Notes

- The runtime now attempts ONNX Runtime Web first (`detector.onnx` + `discriminator.onnx`) and automatically falls back to a deterministic local proxy scorer if ONNX assets are missing.
