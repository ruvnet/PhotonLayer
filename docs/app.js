// PhotonLayer optical-compression playground.
//
// Loads the real photonlayer-wasm module and runs the deterministic optical
// pipeline entirely in the browser. The "verify" button re-runs the identical
// inputs and confirms the BLAKE3 frame hash matches byte-for-byte.

import init, { compress, photonlayer_version } from "./pkg/photonlayer.js";

const N = 64; // grid side (power of two, required by OpticalConfig::validate)

// ── Preset inputs: N×N grayscale Uint8Array, generated deterministically ──────

function makeDigit() {
  // A blocky "7": top bar + diagonal stroke.
  const a = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let on = false;
      if (y >= 12 && y < 20 && x >= 14 && x < 50) on = true; // top bar
      const diag = 50 - Math.floor((y - 18) * 0.55);
      if (y >= 18 && y < 54 && x >= diag - 4 && x < diag + 4) on = true; // stroke
      a[y * N + x] = on ? 255 : 20;
    }
  }
  return a;
}

function makeShape() {
  // A centered diamond.
  const a = new Uint8Array(N * N);
  const c = N / 2;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const d = Math.abs(x - c) + Math.abs(y - c);
      a[y * N + x] = d < 22 ? 235 : 18;
    }
  }
  return a;
}

function makeChecker() {
  const a = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      a[y * N + x] = (((x >> 3) + (y >> 3)) & 1) === 0 ? 255 : 0;
    }
  }
  return a;
}

const PRESETS = { digit: makeDigit(), shape: makeShape(), checker: makeChecker() };
const POOL_FACTORS = [1, 2, 4, 8]; // slider index -> pool factor (each divides 64)

// ── DOM ───────────────────────────────────────────────────────────────────────

const el = (id) => document.getElementById(id);
const presetSel = el("preset");
const maskSel = el("mask");
const seedInput = el("seed");
const distInput = el("dist");
const poolInput = el("pool");
const seedOut = el("seedOut");
const distOut = el("distOut");
const poolOut = el("poolOut");
const origCanvas = el("origCanvas");
const measCanvas = el("measCanvas");
const origBytes = el("origBytes");
const measBytes = el("measBytes");
const reductionEl = el("reduction");
const hashEl = el("hash");
const verifyBtn = el("verifyBtn");
const verifyResult = el("verifyResult");
const statusEl = el("status");

// ── Rendering helpers ─────────────────────────────────────────────────────────

// Draw a w×h grayscale Uint8Array onto a canvas (sized to fit).
function drawGray(canvas, buf, w, h) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = buf[i];
    img.data[i * 4 + 0] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function currentParams() {
  return {
    image: PRESETS[presetSel.value],
    mask: maskSel.value,
    seed: parseInt(seedInput.value, 10),
    dist: parseFloat(distInput.value),
    pool: POOL_FACTORS[parseInt(poolInput.value, 10)],
  };
}

// Run one compression and return the CompressResult.
function runCompress(p) {
  // seed must be a BigInt: the wasm export takes a u64.
  return compress(p.image, N, N, p.mask, BigInt(p.seed), 1.0, p.dist, p.pool);
}

function syncLabels(p) {
  seedOut.textContent = String(p.seed);
  distOut.textContent = `${p.dist} mm`;
  const ratio = p.pool * p.pool;
  poolOut.textContent = `${p.pool}× (${ratio}:1)`;
}

let lastResult = null;

function render() {
  const p = currentParams();
  syncLabels(p);
  let r;
  try {
    r = runCompress(p);
  } catch (e) {
    statusEl.textContent = `Error: ${e}`;
    return;
  }
  lastResult = r;

  drawGray(origCanvas, r.orig_buf, r.orig_width, r.orig_height);
  drawGray(measCanvas, r.measurement_buf, r.meas_width, r.meas_height);

  origBytes.textContent = `${r.input_bytes} bytes (${r.orig_width}×${r.orig_height})`;
  measBytes.textContent = `${r.measurement_bytes} bytes (${r.meas_width}×${r.meas_height})`;

  const factor = r.input_bytes / r.measurement_bytes;
  reductionEl.textContent =
    `${r.input_bytes} B → ${r.measurement_bytes} B  ·  ${factor.toFixed(0)}× smaller`;

  hashEl.textContent = r.frame_hash;
  verifyResult.textContent = " ";
  verifyResult.className = "verify-result";
  statusEl.textContent = "Running live in WebAssembly.";
}

function verify() {
  if (!lastResult) return;
  const p = currentParams();
  verifyResult.textContent = "Re-running…";
  verifyResult.className = "verify-result";
  // Defer so the "Re-running…" text paints before the synchronous re-run.
  setTimeout(() => {
    let r2;
    try {
      r2 = runCompress(p);
    } catch (e) {
      verifyResult.textContent = `Error: ${e}`;
      verifyResult.className = "verify-result fail";
      return;
    }
    const match = r2.frame_hash === lastResult.frame_hash;
    if (match) {
      verifyResult.textContent = `✓ Verified — hash matches (${r2.frame_hash.slice(0, 12)}…)`;
      verifyResult.className = "verify-result ok";
    } else {
      verifyResult.textContent = "✗ Mismatch — non-deterministic!";
      verifyResult.className = "verify-result fail";
    }
  }, 30);
}

// ── Wire up ─────────────────────────────────────────────────────────────────

async function main() {
  try {
    await init(); // fetch + instantiate ./pkg/photonlayer_bg.wasm
  } catch (e) {
    statusEl.textContent =
      `Failed to load WebAssembly module: ${e}. Serve this page over HTTP (see README).`;
    return;
  }

  el("ver").textContent = `photonlayer-wasm v${photonlayer_version()}`;

  for (const c of [presetSel, maskSel]) c.addEventListener("change", render);
  for (const s of [seedInput, distInput, poolInput]) s.addEventListener("input", render);
  verifyBtn.addEventListener("click", verify);

  render();
}

main();
