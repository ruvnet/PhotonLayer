<div align="center">

# PhotonLayer

**Task-trained optics in pure Rust: a learned phase mask compresses light into a tiny sensor measurement, and a small digital decoder reads the answer — with a signed, reproducible receipt of exactly what was measured.**

*Capture less · decide faster · leak less · prove what happened.*

[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Deterministic](https://img.shields.io/badge/output-bit--reproducible-success.svg)](#determinism--receipts)
[![WASM](https://img.shields.io/badge/wasm-ready-purple.svg)](#crates)

</div>

---

## The wedge

PhotonLayer is **not** "an optical neural network." It is **auditable optical compression for task-useful sensing**: a learned phase mask performs a trained analog transform on the incoming light, a small sensor reads a compressed measurement, and a tiny digital decoder produces the decision. The measurement need not look like the scene.

```
scene → trained optical transform → tiny sensor measurement → small decoder → decision (+ receipt)
```

The system sees enough to decide, but captures far less than a full image — and emits a deterministic, hash-bound receipt of the transform and the decision.

## Measured results (real data, deterministic)

Real MNIST (public IDX, 4000 train / 2000 **blind** test, balanced 10-class, seed `0x6e157`). Two masks, two objectives — the product claim and the mechanism — both always reported, no cherry-picking.

**Config A — the product claim** (decoder objective):

| | sensor px | decoder params | blind-test acc |
|---|---:|---:|---:|
| full-image baseline (same tiny centroid decoder) | 1024 | 10 240 | **75.40 %** |
| **optical compressed** (learned mask + pooled read) | **64** | **640** | **73.05 %** |
| Δ vs baseline | — | — | **−2.35 pp** |

**→ 16× fewer sensor pixels and 16× fewer digital MACs**, for a 2.35 pp accuracy cost. A learned mask beats a random mask by **+8.1 pp** decoded — learning the optics genuinely helps.

**Config B — the mechanism** (argmax-differential objective, no decoder): isolates the Li/Ozcan differential-detection lever — plain argmax `I⁺` 18.40 % vs differential `I⁺−I⁻` 34.90 % = **+16.5 pp**. (Absolute accuracy is modest by construction; the delta isolates the lever, it is not a headline accuracy.)

### Breaking the ceiling — gradient training (the headline)

The hill-climb result above is an **optimizer** ceiling, not an optics limit. Training the **same single phase mask** by **analytic gradient descent** — through a proven adjoint of the diffraction operator (`Propagator::backward_into`, validated by an exact-adjoint identity *and* a finite-difference grad-check) — clears it decisively:

| accuracy @ 16× compression (same NCC eval decoder) | |
|---|---:|
| random-init mask | 65.35 % |
| hill-climb mask (single-mask optimum) | 73.05 % |
| full-image baseline (matched tiny decoder, 1024 px) | 75.40 % |
| **gradient-trained optical (64 px)** | **83.30 %** |

**+10.25 pp over hill-climb**, and it **exceeds the matched full-image baseline by +7.9 pp** — at 16× fewer sensor pixels. The learned diffractive transform is acting as a *useful analog feature extractor*, not merely a compressor: the 64-pixel optical features are more linearly separable for a tiny decoder than the raw 1024 pixels. Deterministic (no FMA), reproduced from clean in ~24 s.

```sh
cargo test -p photonlayer-bench --release --test mnist_gradient_bench \
    mnist_gradient_full -- --ignored --nocapture
```

> Honest framing: the baseline is a *matched tiny (nearest-centroid) decoder*, not a CNN — so "beats the full-image baseline" means beats that matched baseline, not all digital methods. It is a real, reproduced ceiling-break for single-layer task-trained optics, not an absolute MNIST SOTA.

Reproduce the single-mask numbers:
```sh
# fetch MNIST IDX once into crates/photonlayer-bench/data/mnist/ (see test header), then:
cargo test -p photonlayer-bench --release --test mnist_differential_bench \
    mnist_differential_full -- --ignored --nocapture
```

### Honest scope — what this is and is not

- This is a **single** task-trained optical layer plus a tiny decoder = **competitive single-layer optical compression**. It is **not** a new accuracy state-of-the-art. Multi-layer ~97–99 % diffractive/optoelectronic networks are explicitly out of scope.
- Hill-climbing converges to an **optimizer ceiling** (~73 %); **analytic gradient descent breaks it to 83.30 %** (see above), reproduced and deterministic. Further headroom (multi-plane cascade) is roadmap, not yet measured.
- **No privacy or security guarantee is claimed.** PhotonLayer stores a *learned measurement, not the raw image* — a description, not a theorem. Reconstruction-resistance is an empirical property of one trained model. Never read this as "cannot be reconstructed," "privacy-preserving," or "zero-knowledge."

## Determinism & receipts

The moat is not the optics — it is **reproducibility**. PhotonLayer uses an in-house scalar `Complex` type and a hand-rolled FFT (no FMA, fixed reduction order), so the same input + mask + config + seed produces **bit-identical** output across Linux/macOS/WASM. Every experiment emits a BLAKE3-bound receipt (model hash, config, what was measured, the decision), so an optical result is a *reproducible experiment*, not a claim.

## Quickstart

```sh
cargo add photonlayer-core
```

```rust
use photonlayer_core::prelude::*;

let n = 32;
let pixels: Vec<f32> = (0..n * n).map(|i| (i % n) as f32 / n as f32).collect();
let img  = InputImage::from_norm_f32(n, n, pixels).unwrap();
let mask = PhaseMask::random(n, n, 42);
let cfg  = OpticalConfig::demo(n, n);

let frame = ScalarSimulator.simulate(&img, &mask, &cfg).unwrap();
// Re-running is bit-identical:
assert_eq!(frame.frame_hash, ScalarSimulator.simulate(&img, &mask, &cfg).unwrap().frame_hash);
```

## Crates

| Crate | What it is |
|---|---|
| **`photonlayer-core`** | the optical simulator: scalar diffraction (Fresnel / Fraunhofer / angular-spectrum), deterministic FFT, phase mask, sensor, metrics, receipts |
| **`photonlayer-bench`** | benchmarks: synthetic compression sweep + the real-MNIST optical-compression benchmark with the differential-detection ablation |
| **`photonlayer-wasm`** | WebAssembly bindings for the browser playground |
| **`photonlayer-cli`** | command-line driver |

The hot path is hyper-optimized and proven: a cached, in-place `Propagator` plus a checkerboard-fftshift fold and precomputed FFT twiddle tables give a **measured ~2.0×** speedup, fully deterministic.

## Roadmap

1. **Gradient descent (keystone)** — analytic adjoint through the diffraction FFT (`Propagator::backward_into` with `conj(H)`), a differentiable training head (NCC kept for eval), Adam. Expected ~85–89 % on the same single plane (+12–15 pp), then 2–3 cascaded planes.
2. **FiberGate** — a multimode-fiber transmission-matrix substrate (ADR-263) with drift-aware training and receipt-verified (not zero-knowledge) verification; simulate the deterministic non-square replay first.
3. **Hardware bridge** — a DiffuserCam-style lensless camera as the first physical demo, after an in-sim phase-quantization sweep to size the fabrication.

## Citations

- Wirth-Singh et al., *Compressed Meta-Optical Encoder for Image Classification*, arXiv:2406.06534 (2024) — closest architectural twin.
- Bezzam, Vetterli, Simeoni, arXiv:2206.01429 (2022) — few-pixel learned-mask anchor.
- Lin et al., *All-optical machine learning using diffractive deep neural networks*, *Science* 361:1004 (2018) — the multi-layer D2NN regime.
- Li, Ozcan et al., arXiv:1906.03417 — differential detection (`I⁺−I⁻`).

## License

[MIT](LICENSE) © Ruvector
