Place onnxruntime-web browser distribution files in this folder.

Minimum required files:
- ort.min.mjs
- ort-wasm-simd.wasm
- ort-wasm.wasm

How to get them (from repository root):
1) npm init -y
2) npm install onnxruntime-web
3) Copy files from node_modules/onnxruntime-web/dist/ into this folder.

The extension loads `vendor/onnxruntime-web/ort.min.mjs` at runtime.
If the files are missing, it falls back to the lightweight proxy scorer.
