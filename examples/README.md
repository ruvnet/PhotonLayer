# PhotonLayer Examples

A practical-to-exotic ladder of **runnable** examples. Every one compiles
(`cargo build --examples --workspace`, 0 errors / 0 warnings) and runs **offline**
on the built-in deterministic synthetic dataset — no network, no external data.
Each example computes real results from the crates; none print fabricated numbers.

The examples live next to the crate whose API they exercise (Cargo convention):

- core-only demos in [`crates/photonlayer-core/examples/`](../crates/photonlayer-core/examples)
- learning / data demos in [`crates/photonlayer-bench/examples/`](../crates/photonlayer-bench/examples)

Run any example with the command in its row. `--release` is recommended for the
learning examples (they still finish in seconds); the core demos are instant.

## Practical

| Example | What it shows | Run |
|---|---|---|
| **hello_optics** | The minimal pipeline: image → phase mask → propagate → sensor frame. Prints the deterministic frame hash and shows a re-run is bit-identical. | `cargo run --release --example hello_optics -p photonlayer-core` |
| **compression** | A 64×64 image vs a tiny 8×8 sensor measurement: ~64× pixel/byte reduction, with the measurement rendered as ASCII so you can SEE it is not the scene (near-zero correlation). | `cargo run --release --example compression -p photonlayer-core` |
| **receipt** | Builds an RVF receipt over a run, verifies it PASSES, then tampers one field and shows verification FAILS. | `cargo run --release --example receipt -p photonlayer-core` |

## Intermediate

| Example | What it shows | Run |
|---|---|---|
| **propagation_modes** | A single bright pixel propagated under Fresnel vs Fraunhofer vs AngularSpectrum — real diffraction physics (spread grows with distance; Fraunhofer of a point is a flat spectrum). | `cargo run --release --example propagation_modes -p photonlayer-core` |
| **learn_mask** | Trains a phase mask by hill-climbing on the synthetic shapes at a tiny 2×2 sensor; learned mask beats the random mask (≈75% → 100%). | `cargo run --release --example learn_mask -p photonlayer-bench` |
| **differential_detection** | The Li/Ozcan I⁺−I⁻ readout lever: plain argmax vs differential argmax on the same trained mask (model selection by train accuracy), differential wins by ~+0.22. | `cargo run --release --example differential_detection -p photonlayer-bench` |

## Advanced

| Example | What it shows | Run |
|---|---|---|
| **gradient_training** | Trains a mask by analytic gradient descent through the PROVEN diffraction adjoint (`Propagator::backward_into` + `phase_gradient`). Prints the CE loss curve falling and gradient beating the random init. | `cargo run --release --example gradient_training -p photonlayer-bench` |
| **multiplane_cascade** | Trains a 2-plane diffractive cascade with the COMPOSED adjoint. Shows the 2-plane reaching lower loss and the second plane seeing a decorrelated (non-redundant) field. (Accuracy saturates on the easy synthetic task; the per-plane accuracy gain is a MNIST result, see README.) | `cargo run --release --example multiplane_cascade -p photonlayer-bench` |

## Exotic

| Example | What it shows | Run |
|---|---|---|
| **optical_feature_extractor** | Optics as an analog feature extractor: under a fixed weak nearest-centroid decoder at a 4-number sensor, the learned optical features are more linearly separable than the raw pixels (≈70% → 100%) — with the honest weak-decoder caveat. | `cargo run --release --example optical_feature_extractor -p photonlayer-bench` |
| **wavefront_focus** | A designed quadratic-lens phase concentrates a plane wave to a focal spot; prints the peak/mean intensity concentration ratio (flat ≈1× vs lens ≈25×) and the focal-strength sweet spot. | `cargo run --release --example wavefront_focus -p photonlayer-core` |
| **privacy_probe** | A linear reconstruction attack on the optical measurement: identity mask leaks (high PSNR), optical masks leak less — with the explicit caveat that this is a LINEAR lower bound, not a privacy guarantee. | `cargo run --release --example privacy_probe -p photonlayer-bench` |

## Real data (optional)

| Example | What it shows | Run |
|---|---|---|
| **mnist_compression** | The real-MNIST optical-compression + differential-detection benchmark. Loads IDX files from a local cache; if absent it prints the fetch instructions and exits cleanly (never panics), so it always runs offline. | `cargo run --release --example mnist_compression -p photonlayer-bench` |

The MNIST IDX files are not downloaded by the example or committed to the repo.
Place the four standard files in `crates/photonlayer-bench/data/mnist/` to enable
the real-data run (the example prints the exact filenames when the cache is
missing).

## Honesty notes

These examples match the README's hardened framing:

- The decoder is a **deliberately weak** nearest-centroid head — results are about
  feature *separability* under a fixed weak decoder, not a claim that optics beat
  digital methods (a small CNN on raw pixels beats both).
- The privacy probe is a **linear lower bound**; low linear-attack PSNR is not a
  privacy guarantee.
- Everything is **simulation-only** and **deterministic** (same input + mask +
  config ⇒ same output hash).
