# PhotonLayer — Optical Compression Playground (demo)

A self-contained, **real** in-browser demo of the PhotonLayer optical pipeline.
Everything runs client-side via WebAssembly compiled from `photonlayer-wasm` —
there is no server and no mocked output.

## What it shows

1. **Original input** — one of a few bundled grayscale presets (a digit-like
   "7", a diamond, a checkerboard).
2. **Sensor measurement** — the input is transformed by a preset/learned phase
   mask, propagated to a sensor, and average-pooled into a smaller, noise-like
   measurement. A `N bytes → M bytes` readout shows the data reduction.
3. **Receipt** — a BLAKE3 hash binding the run inputs + measurement. The
   **Verify** button re-runs the identical inputs and confirms the hash matches
   byte-for-byte (a determinism / anti-swap integrity check).

Sliders for mask seed, propagation distance, and sensor pooling make the
measurement change live.

> PhotonLayer stores a learned measurement, not the image. No privacy or
> security guarantee is claimed.

## Local preview

GitHub Pages serves this `docs/` directory directly. To preview locally, serve
over HTTP (ES-module + WASM `fetch` won't work from a `file://` URL):

```bash
# from the repository root
python -m http.server 8000 --directory docs
# then open http://localhost:8000/
```

## Rebuilding the WASM module

The compiled artifacts live in `docs/pkg/` (committed so Pages can serve them).
To rebuild after changing `crates/photonlayer-wasm`:

```bash
rustup target add wasm32-unknown-unknown   # one-time
wasm-pack build crates/photonlayer-wasm \
  --target web --release \
  --out-dir ../../docs/pkg --out-name photonlayer
```

Files: `index.html`, `app.css`, `app.js`, and `pkg/` (the generated
`photonlayer.js` ES module + `photonlayer_bg.wasm`).
