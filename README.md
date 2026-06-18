<div align="center">

# PhotonLayer

**Task-trained optics in pure Rust: a learned phase mask compresses light into a tiny sensor measurement, and a small digital decoder reads the answer — with a signed, reproducible receipt of exactly what was measured.**

*Capture less · decide faster · leak less · prove what happened.*

[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Deterministic](https://img.shields.io/badge/output-bit--reproducible-success.svg)](#determinism--receipts)
[![crates.io](https://img.shields.io/badge/crates.io-photonlayer--core-blue.svg)](https://crates.io/crates/photonlayer-core)

### ▶ [Live demo — optical-compression playground](https://ruvnet.github.io/PhotonLayer/)

*Runs entirely in your browser via WASM: shape light through a learned mask, watch it compress to a tiny measurement, and verify the deterministic receipt.*

</div>

---

## In plain language

**What if a camera could skip taking the picture and just capture the answer?**

A normal camera records every pixel of a scene, and then a computer reads all of them to decide what it's looking at. PhotonLayer flips that around. It places a specially-shaped piece of "smart glass" — a *phase mask* — in front of a small sensor. The mask bends the incoming light so that, by the time it reaches the sensor, the useful information has been squeezed into just a handful of numbers. A tiny program reads those numbers and gives the answer.

Think of it like a translator who listens to a whole speech and hands you a one-line summary — you never needed the full transcript to act on it. The "lens" is *trained* (by trial and improvement) to do that summarizing **in the light itself**, before anything is digitized.

Why that's interesting:

- **Capture less** — a few sensor pixels instead of a full image, so there is far less data to move and process.
- **Decide faster and cheaper** — the heavy lifting happens in the optics; the digital part stays tiny.
- **Leak less** — the sensor records a compressed *measurement*, not a viewable photo of the scene. *(This is something we measure, not a privacy guarantee — see [Honest scope](#honest-scope--what-this-is-and-is-not).)*
- **Prove what happened** — every run emits a tamper-evident *receipt* (a cryptographic fingerprint), so a result is a reproducible experiment, not just a claim.

Today this is a **software simulator written in Rust** — the physics, the training, and the receipts are all real and reproducible; building the actual glass is on the roadmap. You can try it right now in your browser (no install) via the **[live demo](https://ruvnet.github.io/PhotonLayer/)**, or run a 30-line tour locally:

```sh
cargo run --release --example hello_optics -p photonlayer-core
```

The rest of this README gets progressively more technical — keep reading for the sharp framing, the measured results, and the honest caveats.

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

**+10.25 pp over hill-climb**, at 16× fewer sensor pixels — a real, reproduced ceiling-break for single-layer task-trained optics. Reproduced from clean in ~24 s.

**Going deeper — a multi-plane diffractive cascade** (several phase planes with free-space propagation between them, trained end-to-end by the *composed* adjoint) climbs further, still at 16× sensor compression and the same NCC eval:

| trained planes | blind-test acc | Δ vs single-plane |
|---|---:|---:|
| 1 (single) | 83.30 % | — |
| **2** | **88.80 %** | **+5.50 pp** |
| 3 | 89.80 % | +6.50 pp |

Each added plane sees a genuinely different diffracted field (verified decorrelated to ~0.04 correlation, not redundant). The **2-plane 88.80 %** is the robust headline; the 3-plane 89.80 % is real but more init-sensitive, so it is reported, not over-asserted. Reproduced from clean in ~62 s, deterministic.

```sh
cargo test -p photonlayer-bench --release --test mnist_gradient_bench \
    mnist_gradient_full -- --ignored --nocapture     # single-plane 83.30%
cargo test -p photonlayer-bench --release --test mnist_cascade_bench \
    mnist_cascade_full -- --ignored --nocapture       # 2-plane 88.80%, 3-plane 89.80%
```

> **Read the +7.9 pp vs the full-image baseline carefully — it is NOT a superiority claim.** Holding the decoder fixed at a *tiny nearest-centroid head*, the 64 learned optical features are more linearly separable than 1024 raw pixels (83.30 % vs 75.40 %). That is a statement about **feature separability under a fixed weak decoder**, not evidence that optics beat digital methods: nearest-centroid on raw pixels is a deliberately weak baseline, and a small CNN on the *same* 1024 pixels reaches ~99 % and beats both. We report this so the obvious objection is already answered. It is a single-layer ceiling-break, not an absolute MNIST SOTA.

Reproduce the single-mask numbers:
```sh
# fetch MNIST IDX once into crates/photonlayer-bench/data/mnist/ (see test header), then:
cargo test -p photonlayer-bench --release --test mnist_differential_bench \
    mnist_differential_full -- --ignored --nocapture
```

### Honest scope — what this is and is not

- This is a **single** task-trained optical layer plus a tiny decoder = **competitive single-layer optical compression**. It is **not** a new accuracy state-of-the-art. Multi-layer ~97–99 % diffractive/optoelectronic networks are explicitly out of scope.
- Hill-climbing converges to an **optimizer ceiling** (~73 %); **analytic gradient descent breaks it to 83.30 %** single-plane and a **multi-plane cascade reaches 88.80 % (2-plane) / 89.80 % (3-plane)** — all reproduced and deterministic, all at 16× sensor compression with the same matched decoder.
- **No privacy or security guarantee is claimed.** PhotonLayer stores a *learned measurement, not the raw image* — a description, not a theorem. Reconstruction-resistance is an empirical property of one trained model; the bundled probe measures **linear** invertibility only, and nonlinear (CNN/U-Net) reconstruction is expected to succeed. Never read this as "cannot be reconstructed," "privacy-preserving," or "zero-knowledge."
- **The "16× MAC reduction" counts the *digital decoder* only** (640 vs 10 240). The optical front end performs an FFT-scale transform that is *passive in real hardware* (free-space diffraction) but is **not free in this simulator** — it is not counted in that figure. The honest claim is 16× fewer **sensor pixels** and 16× fewer **digital-decoder MACs**.
- **All accuracy figures are noise-free scalar-diffraction simulation with continuous phase.** Robustness to phase quantization, sensor noise, and fabrication error is not yet characterized; expect degradation on real hardware. (A quantization/SNR ablation is roadmap.)

## Determinism & receipts

The intended moat is **reproducibility**. PhotonLayer uses an in-house scalar `Complex` type and a hand-rolled FFT with a fixed reduction order, designed so the same input + mask + config + seed yields **bit-identical** output. This is **verified within runs/builds on x86-64**; full cross-platform (Linux/macOS/WASM) bit-identity is a **design goal, not yet proven** — it depends on disabling FP contraction (FMA) and on platform-independent transcendentals (`sin`/`cos`/`sqrt`), which a CI matrix + checked-in golden hash will enforce (roadmap). Every experiment emits a BLAKE3-bound receipt (model hash, config, measurement, decision), so a result is a *reproducible experiment* on a fixed target.

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

## Examples

A practical-to-exotic ladder of runnable examples — full catalog (what each shows
+ run command) in **[`examples/README.md`](examples/README.md)**. Every one
compiles clean and runs offline on the built-in synthetic dataset; all numbers are
computed, never fabricated.

```sh
# Practical
cargo run --release --example hello_optics           -p photonlayer-core   # minimal pipeline + deterministic hash
cargo run --release --example compression            -p photonlayer-core   # 64x fewer sensor pixels, rendered as ASCII
cargo run --release --example receipt                -p photonlayer-core   # build, verify, tamper -> fails
# Intermediate
cargo run --release --example propagation_modes      -p photonlayer-core   # Fresnel / Fraunhofer / AngularSpectrum
cargo run --release --example learn_mask             -p photonlayer-bench  # hill-climb: learned beats random
cargo run --release --example differential_detection -p photonlayer-bench  # the Li/Ozcan I+ - I- lever
# Advanced
cargo run --release --example gradient_training      -p photonlayer-bench  # train through the proven adjoint
cargo run --release --example multiplane_cascade     -p photonlayer-bench  # 2-plane composed adjoint
# Exotic
cargo run --release --example optical_feature_extractor -p photonlayer-bench  # optics as analog feature extractor
cargo run --release --example wavefront_focus        -p photonlayer-core   # a learned lens concentrates light
cargo run --release --example privacy_probe          -p photonlayer-bench  # linear reconstruction attack (lower bound)
# Real data (skips cleanly if the MNIST cache is absent)
cargo run --release --example mnist_compression      -p photonlayer-bench
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
