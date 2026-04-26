function createCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

export function imageElementToCanvas(imgEl) {
  const w = Math.max(1, imgEl.naturalWidth || imgEl.width || 1);
  const h = Math.max(1, imgEl.naturalHeight || imgEl.height || 1);
  const canvas = createCanvas(w, h);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(imgEl, 0, 0, w, h);
  return canvas;
}

export function videoElementToCanvas(videoEl) {
  const w = Math.max(1, videoEl.videoWidth || videoEl.clientWidth || 1);
  const h = Math.max(1, videoEl.videoHeight || videoEl.clientHeight || 1);
  const canvas = createCanvas(w, h);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(videoEl, 0, 0, w, h);
  return canvas;
}

export function cropCenterToTensor(canvas, size = 128) {
  const { width, height } = canvas;
  const side = Math.min(width, height);
  const x = Math.floor((width - side) / 2);
  const y = Math.floor((height - side) / 2);

  const tmp = createCanvas(size, size);
  const ctx = tmp.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(canvas, x, y, side, side, 0, 0, size, size);

  const pixels = ctx.getImageData(0, 0, size, size).data;
  const out = new Float32Array(size * size * 3);
  let j = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    out[j++] = (pixels[i] / 255) * 2 - 1;
    out[j++] = (pixels[i + 1] / 255) * 2 - 1;
    out[j++] = (pixels[i + 2] / 255) * 2 - 1;
  }
  return out;
}

export function fakeProbabilityToLabel(fakeProb, threshold) {
  const isFake = fakeProb >= threshold;
  return {
    label: isFake ? "FAKE" : "REAL",
    confidence: isFake ? fakeProb : 1 - fakeProb,
  };
}
