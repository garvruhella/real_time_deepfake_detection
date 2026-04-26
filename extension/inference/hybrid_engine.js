import {
  cropCenterToTensor,
  fakeProbabilityToLabel,
  imageElementToCanvas,
  videoElementToCanvas,
} from "./media_preprocess.js";

import { runLatencyBenchmark } from "./benchmark.js";

const DEFAULT_FUSION = {
  detectorWeight: 0.7,
  discriminatorWeight: 0.3,
  threshold: 0.5,
};

export class HybridEngine {
  constructor() {
    this.ready = false;
    this.degradedMode = false;
    this.fusion = { ...DEFAULT_FUSION };
    this.ort = null;
    this.detectorSession = null;
    this.discriminatorSession = null;
    this.modelReady = false;
  }

  async init() {
    try {
      const cfgUrl = chrome.runtime.getURL("models/hybrid_config.json");
      const cfg = await fetch(cfgUrl);
      if (cfg.ok) {
        const payload = await cfg.json();
        if (payload?.fusion) {
          this.fusion = {
            detectorWeight: payload.fusion.detector_weight ?? DEFAULT_FUSION.detectorWeight,
            discriminatorWeight:
              payload.fusion.discriminator_weight ?? DEFAULT_FUSION.discriminatorWeight,
            threshold: payload.fusion.threshold ?? DEFAULT_FUSION.threshold,
          };
        }
      }
    } catch (err) {
      console.warn("hybrid config not available, using defaults", err);
    }

    await this.initOnnxSessions();
    this.ready = true;
  }

  async scoreImageElement(imgEl) {
    const canvas = imageElementToCanvas(imgEl);
    return this.scoreCanvas(canvas, "image");
  }

  async scoreVideoElement(videoEl) {
    const canvas = videoElementToCanvas(videoEl);
    return this.scoreCanvas(canvas, "video");
  }

  async scoreCanvas(canvas, mediaKind = "unknown") {
    if (!this.ready) {
      await this.init();
    }
    const tensor = cropCenterToTensor(canvas, 128);
    const nchw = rgbInterleavedToNCHW(tensor, 128, 128);

    let detectorFake;
    let discriminatorFake;
    let degradedMode = false;

    if (this.modelReady) {
      try {
        ({ detectorFake, discriminatorFake } = await this.runOnnxInference(nchw));
      } catch (err) {
        console.warn("onnx inference failed, falling back to proxy scorer", err);
        degradedMode = true;
        detectorFake = simpleDetectorProxy(tensor);
        discriminatorFake = simpleDiscriminatorProxy(tensor);
      }
    } else {
      degradedMode = true;
      detectorFake = simpleDetectorProxy(tensor);
      discriminatorFake = simpleDiscriminatorProxy(tensor);
    }

    const hybridFake =
      this.fusion.detectorWeight * detectorFake +
      this.fusion.discriminatorWeight * discriminatorFake;

    const outcome = fakeProbabilityToLabel(hybridFake, this.fusion.threshold);
    return {
      mediaKind,
      detectorFake,
      discriminatorFake,
      hybridFake,
      ...outcome,
      degradedMode,
    };
  }

  async benchmark(iterations = 30) {
    return runLatencyBenchmark(this, iterations);
  }

  async initOnnxSessions() {
    try {
      this.ort = await loadOrtModule();
      if (!this.ort) {
        this.modelReady = false;
        this.degradedMode = true;
        return;
      }

      const detectorUrl = chrome.runtime.getURL("models/detector.onnx");
      const discriminatorUrl = chrome.runtime.getURL("models/discriminator.onnx");
      const options = {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      };
      this.detectorSession = await this.ort.InferenceSession.create(detectorUrl, options);
      this.discriminatorSession = await this.ort.InferenceSession.create(discriminatorUrl, options);
      this.modelReady = true;
      this.degradedMode = false;
    } catch (err) {
      console.warn("onnx runtime assets not available; using fallback scorer", err);
      this.modelReady = false;
      this.degradedMode = true;
    }
  }

  async runOnnxInference(nchw) {
    const input = new this.ort.Tensor("float32", nchw, [1, 3, 128, 128]);

    const detectorInputName = this.detectorSession.inputNames[0];
    const detectorOutputName = this.detectorSession.outputNames[0];
    const detectorOutput = await this.detectorSession.run({ [detectorInputName]: input });
    const detectorLogit = detectorOutput[detectorOutputName].data[0];
    const detectorFake = sigmoid(detectorLogit);

    const discInputName = this.discriminatorSession.inputNames[0];
    const discOutputName = this.discriminatorSession.outputNames[0];
    const discOutput = await this.discriminatorSession.run({ [discInputName]: input });
    const discReal = clamp01(discOutput[discOutputName].data[0]);
    const discriminatorFake = 1 - discReal;

    return { detectorFake, discriminatorFake };
  }
}

function simpleDetectorProxy(tensor) {
  // Mean absolute adjacent pixel difference as proxy for texture artifacts.
  let sum = 0;
  let count = 0;
  for (let i = 0; i < tensor.length - 3; i += 3) {
    const d =
      Math.abs(tensor[i] - tensor[i + 3]) +
      Math.abs(tensor[i + 1] - tensor[i + 4]) +
      Math.abs(tensor[i + 2] - tensor[i + 5]);
    sum += d;
    count += 1;
  }
  return clamp01((sum / Math.max(count, 1)) * 1.4);
}

function simpleDiscriminatorProxy(tensor) {
  // Channel discrepancy proxy as a lightweight anomaly indicator.
  let rg = 0;
  let gb = 0;
  let count = 0;
  for (let i = 0; i < tensor.length; i += 3) {
    rg += Math.abs(tensor[i] - tensor[i + 1]);
    gb += Math.abs(tensor[i + 1] - tensor[i + 2]);
    count += 1;
  }
  return clamp01(((rg + gb) / Math.max(count, 1)) * 0.9);
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function rgbInterleavedToNCHW(rgbTensor, width, height) {
  const area = width * height;
  const out = new Float32Array(3 * area);
  for (let i = 0; i < area; i += 1) {
    const src = i * 3;
    out[i] = rgbTensor[src];
    out[area + i] = rgbTensor[src + 1];
    out[2 * area + i] = rgbTensor[src + 2];
  }
  return out;
}

async function loadOrtModule() {
  try {
    const mod = await import(chrome.runtime.getURL("vendor/onnxruntime-web/ort.min.mjs"));
    return mod.default || mod;
  } catch (err) {
    console.warn("failed to load ort.min.mjs", err);
    return null;
  }
}
