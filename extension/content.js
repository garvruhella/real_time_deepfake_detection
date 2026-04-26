const OVERLAY_CLASS = "deepfake-detector-overlay-badge";
const PROCESSED_ATTR = "data-deepfake-processed";

let runtimeConfig = {
  settings: { enabled: true, perfMode: "balanced" },
  perf: { videoIntervalMs: 800, imageBatchSize: 4 },
};

let engine = null;
const observedVideos = new WeakMap();

init().catch((err) => console.error("Deepfake detector init failed", err));

async function init() {
  const mod = await import(chrome.runtime.getURL("inference/hybrid_engine.js"));
  engine = new mod.HybridEngine();
  runtimeConfig = await sendRuntimeConfigRequest();
  await engine.init();
  injectStyles();
  if (runtimeConfig.settings.enabled) {
    scanImages();
    scanVideos();
  }
  setInterval(() => {
    if (runtimeConfig.settings.enabled) {
      scanImages();
    }
  }, 1800);

  window.__deepfakeDetectorBenchmark = async (iterations = 30) => {
    const result = await engine.benchmark(iterations);
    console.log("Deepfake detector benchmark", result);
    return result;
  };
}

chrome.runtime.onMessage.addListener((message) => {
  if (message?.type === "SETTINGS_BROADCAST") {
    runtimeConfig.settings = message.payload;
    if (runtimeConfig.settings.enabled) {
      scanImages();
      scanVideos();
    }
  }
  if (message?.type === "RESCAN_MEDIA") {
    scanImages(true);
    scanVideos(true);
  }
});

async function scanImages(force = false) {
  const images = Array.from(document.querySelectorAll("img"));
  let budget = runtimeConfig.perf.imageBatchSize || 4;

  for (const img of images) {
    if (budget <= 0) break;
    if (!isElementVisible(img) || !img.complete) continue;
    if (!force && img.getAttribute(PROCESSED_ATTR) === "1") continue;

    budget -= 1;
    img.setAttribute(PROCESSED_ATTR, "1");
    try {
      const result = await engine.scoreImageElement(img);
      renderOverlay(img, result);
      chrome.runtime.sendMessage({
        type: "REPORT_DETECTION",
        payload: {
          mediaKind: "image",
          score: result.hybridFake,
          label: result.label,
          confidence: result.confidence,
        },
      });
    } catch (err) {
      console.warn("image scan failed", err);
    }
  }
}

function scanVideos(force = false) {
  const videos = Array.from(document.querySelectorAll("video"));
  for (const video of videos) {
    if (!isElementVisible(video)) continue;
    if (!force && observedVideos.has(video)) continue;
    startVideoLoop(video);
  }
}

function startVideoLoop(video) {
  stopVideoLoop(video);
  const interval = window.setInterval(async () => {
    if (!runtimeConfig.settings.enabled || video.paused || video.ended) return;
    if (!isElementVisible(video)) return;
    try {
      const result = await engine.scoreVideoElement(video);
      renderOverlay(video, result);
    } catch (err) {
      console.warn("video scan failed", err);
    }
  }, runtimeConfig.perf.videoIntervalMs || 800);
  observedVideos.set(video, interval);
}

function stopVideoLoop(video) {
  const id = observedVideos.get(video);
  if (id) {
    window.clearInterval(id);
    observedVideos.delete(video);
  }
}

function renderOverlay(mediaEl, result) {
  const parent = mediaEl.parentElement || mediaEl;
  if (getComputedStyle(parent).position === "static") {
    parent.style.position = "relative";
  }

  let badge = parent.querySelector(`.${OVERLAY_CLASS}`);
  if (!badge) {
    badge = document.createElement("div");
    badge.className = OVERLAY_CLASS;
    parent.appendChild(badge);
  }

  const confidencePct = Math.round((result.confidence || 0) * 100);
  const scorePct = Math.round((result.hybridFake || 0) * 100);
  badge.textContent = `${result.label} | conf ${confidencePct}% | fakeScore ${scorePct}%`;
  badge.style.background = result.label === "FAKE" ? "rgba(170,0,0,0.85)" : "rgba(0,120,0,0.85)";
}

function isElementVisible(el) {
  const rect = el.getBoundingClientRect();
  return rect.width >= 64 && rect.height >= 64 && rect.bottom > 0 && rect.right > 0;
}

function injectStyles() {
  if (document.getElementById("deepfake-detector-style")) return;
  const style = document.createElement("style");
  style.id = "deepfake-detector-style";
  style.textContent = `
    .${OVERLAY_CLASS} {
      position: absolute;
      top: 6px;
      left: 6px;
      z-index: 2147483647;
      color: #fff;
      font-size: 12px;
      line-height: 1.2;
      padding: 4px 6px;
      border-radius: 4px;
      pointer-events: none;
      font-family: Arial, sans-serif;
      box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    }
  `;
  document.documentElement.appendChild(style);
}

function sendRuntimeConfigRequest() {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: "GET_RUNTIME_CONFIG" }, (response) => {
      resolve(
        response || {
          settings: { enabled: true, perfMode: "balanced" },
          perf: { videoIntervalMs: 800, imageBatchSize: 4 },
        }
      );
    });
  });
}
