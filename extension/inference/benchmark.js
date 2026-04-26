export async function runLatencyBenchmark(engine, iterations = 30) {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");

  const samples = [];
  for (let i = 0; i < iterations; i += 1) {
    fillRandom(ctx, i);
    const t0 = performance.now();
    await engine.scoreCanvas(canvas, "benchmark");
    const t1 = performance.now();
    samples.push(t1 - t0);
  }

  samples.sort((a, b) => a - b);
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const p95 = samples[Math.floor(samples.length * 0.95) - 1] ?? samples[samples.length - 1];
  return { iterations, meanMs: round2(mean), p95Ms: round2(p95) };
}

function fillRandom(ctx, seed) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  const imageData = ctx.createImageData(w, h);
  const data = imageData.data;
  let value = (seed + 1) * 1664525;
  for (let i = 0; i < data.length; i += 4) {
    value = (value * 1664525 + 1013904223) >>> 0;
    data[i] = value & 255;
    data[i + 1] = (value >> 8) & 255;
    data[i + 2] = (value >> 16) & 255;
    data[i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

function round2(v) {
  return Math.round(v * 100) / 100;
}
