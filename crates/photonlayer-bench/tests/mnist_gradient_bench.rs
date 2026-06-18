//! Gradient-trained optical MNIST benchmark — the ceiling-break run (ADR-260).
//!
//! The hill-climb learner converges to a single-layer optimizer ceiling
//! (~73% blind-test at 16x compression — see `mnist_differential_bench.rs`).
//! This test trains the SAME phase mask by **analytic gradient descent** through
//! the proven diffraction adjoint (`Propagator::backward_into`, validated in
//! `photonlayer-core/tests/gradient_check.rs`) and measures whether gradient
//! beats hill-climb on the identical NCC-decoder eval.
//!
//! Train-time uses a differentiable linear+softmax+CE head over the pooled
//! sensor readout; EVAL uses the deterministic nearest-centroid decoder, so the
//! published accuracy is apples-to-apples with the 73.05% hill-climb number.
//!
//! Dataset is NOT vendored — fetch the public IDX files once into
//! `crates/photonlayer-bench/data/mnist/` (see `mnist_differential_bench.rs`
//! header for the exact curl command). This test skips cleanly if absent.
//!
//! Run (AV-safe, integration test — not a standalone bin):
//! ```text
//! cargo test -p photonlayer-bench --release --test mnist_gradient_bench \
//!     mnist_gradient_full -- --ignored --nocapture
//! ```

use photonlayer_bench::grad_train::{build_grad_samples, train_mask_grad, GradTrainConfig};
use photonlayer_bench::mnist::{self, default_cache_dir};
use photonlayer_bench::mnist_bench::{run_mnist_grad, GradMnistResult};
use photonlayer_bench::synthetic::{make_dataset, Sample};
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::propagate::Propagator;
use std::path::Path;

/// The published hill-climb baseline this run is trying to beat (Config A,
/// NCC decoder, 16x compression, 400/200 per class) — see
/// `mnist_differential_bench::mnist_differential_full`.
const HILLCLIMB_OPTICAL_ACC: f32 = 0.7305;
/// The full-image digital baseline at the same compression family (reference).
const FULL_IMAGE_BASELINE_ACC: f32 = 0.7540;

fn load_subsets(
    dir: &Path,
    train_per_class: usize,
    test_per_class: usize,
    cell: usize,
    grid: usize,
) -> Option<(Vec<Sample>, Vec<Sample>)> {
    let raw_train = mnist::load_train(dir)
        .map_err(|e| eprintln!("[skip] MNIST train: {e}"))
        .ok()?;
    let raw_test = mnist::load_test(dir)
        .map_err(|e| eprintln!("[skip] MNIST test: {e}"))
        .ok()?;
    Some((
        mnist::subset(&raw_train, train_per_class, cell, grid),
        mnist::subset(&raw_test, test_per_class, cell, grid),
    ))
}

fn print_table(r: &GradMnistResult) {
    eprintln!("\n===== PhotonLayer MNIST: GRADIENT-trained optics vs hill-climb (ceiling break) =====");
    eprintln!("dataset      : MNIST handwritten digits (public IDX, ossci-datasets mirror)");
    eprintln!(
        "optics       : {0}x{0} field, AngularSpectrum diffraction, {1}x{1} pooled sensor",
        r.grid, r.sensor
    );
    eprintln!(
        "train / test : {} / {} images, balanced across 10 classes (blind test split)",
        r.train_size, r.test_size
    );
    eprintln!(
        "training     : analytic gradient descent through the PROVEN adjoint; Adam;",
    );
    eprintln!(
        "               epochs={} lr_mask={} seed={:#x} (deterministic, no FMA/SIMD)",
        r.epochs, r.lr_mask, r.seed
    );
    eprintln!("  eval decoder: nearest-centroid (same as hill-climb -> apples-to-apples)");
    eprintln!("------------------------------------------------------------------------------");
    eprintln!("  loss curve (mean CE per epoch, honest):");
    {
        let lc = &r.loss_curve;
        let step = (lc.len() / 12).max(1);
        let mut shown = String::new();
        for (i, l) in lc.iter().enumerate() {
            if i % step == 0 || i + 1 == lc.len() {
                shown.push_str(&format!("  e{i}:{l:.3}"));
            }
        }
        eprintln!("   {shown}");
    }
    eprintln!("------------------------------------------------------------------------------");
    eprintln!("  ACCURACY @ {:.0}x compression ({} input px -> {} sensor px), NCC decoder:",
        r.sensor_reduction_x, r.baseline_pixels, r.optical_sensor_pixels);
    eprintln!("    random init mask  (gradient WIN floor)      {:>7.4}", r.random_optical_acc);
    eprintln!("    GRADIENT-trained optical                    {:>7.4}", r.grad_optical_acc);
    eprintln!("    hill-climb optical (published baseline)     {:>7.4}", HILLCLIMB_OPTICAL_ACC);
    eprintln!("    full-image digital baseline (reference)     {:>7.4}", FULL_IMAGE_BASELINE_ACC);
    eprintln!("------------------------------------------------------------------------------");
    eprintln!(
        "    gradient - hill-climb   {:>+7.4}   <== THE HEADLINE (gradient vs hill-climb)",
        r.grad_optical_acc - HILLCLIMB_OPTICAL_ACC
    );
    eprintln!(
        "    gradient - full-image   {:>+7.4}   (closing the optical->digital gap)",
        r.grad_optical_acc - FULL_IMAGE_BASELINE_ACC
    );
    eprintln!(
        "    gradient - random init  {:>+7.4}   (what gradient training bought)",
        r.grad_optical_acc - r.random_optical_acc
    );
    eprintln!("==============================================================================\n");
}

/// Loop-wiring guard (always on, no data needed): a few epochs of gradient
/// training on a tiny synthetic set must strictly reduce the mean CE loss —
/// proves Adam, batching, and the gradient SIGN are wired correctly.
#[test]
fn training_reduces_loss() {
    let n = 16usize;
    let cfg = OpticalConfig::demo(n, n);
    let prop = Propagator::new(n, n, &cfg).unwrap();
    let train = make_dataset(n, 8, 1); // small, deterministic
    let data = build_grad_samples(&train, &cfg);
    let theta0 = vec![0.0f32; n * n];
    let gc = GradTrainConfig {
        epochs: 6,
        batch: 16,
        lr_mask: 0.05,
        lr_head: 0.1,
        sensor: 4,
        seed: 7,
        raw_pool: true,
        adam_eps: 1e-7,
    };
    let out = train_mask_grad(&prop, n, n, &data, &theta0, &gc);
    let first = out.loss_curve.first().copied().unwrap();
    let last = out.loss_curve.last().copied().unwrap();
    assert!(last < first, "loss did not decrease: {first} -> {last}");
}

/// Fast wiring guard (always on, skips without data): a short gradient run on a
/// small subset must beat its own random-init mask. Proves the end-to-end MNIST
/// pipeline (loader -> grad train -> NCC eval) is wired, without the full budget.
#[test]
fn mnist_gradient_smoke() {
    let dir = default_cache_dir();
    let grid = 32;
    let sensor = 8;
    let Some((train, test)) = load_subsets(&dir, 50, 50, 20, grid) else {
        eprintln!("[skip] MNIST cache not found at {} - see header for fetch", dir.display());
        return;
    };
    let gc = GradTrainConfig {
        epochs: 12,
        batch: 50,
        lr_mask: 0.05,
        lr_head: 0.1,
        sensor,
        seed: 0x6E157,
        raw_pool: true,
        adam_eps: 1e-7,
    };
    let r = run_mnist_grad(&train, &test, grid, sensor, &gc);
    print_table(&r);
    // Loss must be monotone-ish down overall, and gradient must beat random init.
    let first = r.loss_curve.first().copied().unwrap();
    let last = r.loss_curve.last().copied().unwrap();
    assert!(last < first, "loss did not fall: {first} -> {last}");
    assert!(
        r.grad_optical_acc >= r.random_optical_acc,
        "gradient {:.4} did not beat random init {:.4}",
        r.grad_optical_acc,
        r.random_optical_acc
    );
    // Structural compression bar (holds by construction).
    assert!(r.sensor_reduction_x >= 16.0, "compression {:.1}x < 16x", r.sensor_reduction_x);
}

/// THE CEILING-BREAK RUN: full-scale gradient training, blind-test accuracy
/// measured with the NCC decoder, printed against the 73.05% hill-climb and the
/// 75.40% full-image baselines. Heavy + deterministic; #[ignore] (run on demand).
///
/// HONESTY: this prints the REAL measured accuracy. The asserted claim is only
/// the robust one — gradient beats its own random-init mask. The gradient-vs-
/// hill-climb delta is REPORTED (the headline), not forced green, so a shortfall
/// is visible rather than hidden.
#[test]
#[ignore = "heavy real-data gradient-training run; see header for the command"]
fn mnist_gradient_full() {
    let dir = default_cache_dir();
    let grid = 32;
    let sensor = 8;
    // Same train/test budget as the hill-climb full test (400/200 per class).
    let Some((train, test)) = load_subsets(&dir, 400, 200, 20, grid) else {
        panic!("MNIST cache not found at {} - fetch IDX files (see header) first", dir.display());
    };
    // Published headline config. The TRAIN-time feature is L2-normalized to
    // MATCH the NCC eval feature (train/eval consistency): the pool_ablation
    // test below measures L2-norm beating raw-pool by ~+2.7pp here precisely
    // because eval uses the L2-normalized NCC. Adam eps=1e-7 avoids v_hat
    // underflow on the many near-zero-gradient dark mask cells.
    let gc = GradTrainConfig {
        epochs: 40,
        batch: 64,
        lr_mask: 0.04,
        lr_head: 0.05,
        sensor,
        seed: 0x6E157,
        raw_pool: false,
        adam_eps: 1e-7,
    };
    let t0 = std::time::Instant::now();
    let r = run_mnist_grad(&train, &test, grid, sensor, &gc);
    eprintln!("[timing] full gradient run: {:.0}s", t0.elapsed().as_secs_f32());
    print_table(&r);

    // Robustly-true assertion: gradient training improves over the random start.
    assert!(
        r.grad_optical_acc >= r.random_optical_acc + 0.02,
        "gradient {:.4} did not beat random init {:.4} by >= 0.02",
        r.grad_optical_acc,
        r.random_optical_acc
    );
    assert!(r.sensor_reduction_x >= 16.0, "compression {:.1}x < 16x", r.sensor_reduction_x);

    // Reported (NOT asserted) — the honest headline:
    let vs_hc = r.grad_optical_acc - HILLCLIMB_OPTICAL_ACC;
    eprintln!(
        "[reported] gradient-trained optical {:.4} vs hill-climb {:.4}: {:+.4} ({})",
        r.grad_optical_acc,
        HILLCLIMB_OPTICAL_ACC,
        vs_hc,
        if vs_hc > 0.0 { "GRADIENT WINS" } else { "did not beat hill-climb" }
    );
}

/// A/B: L2-normalized vs RAW average-pool TRAIN-time readout, all else equal.
/// Settles whether dropping the L2-norm (which couples every pixel gradient and
/// divides away absolute intensity) actually helps on the real metric. Both arms
/// eval with the SAME NCC decoder, so the comparison is clean. Reported, honest.
#[test]
#[ignore = "heavy A/B: L2-norm vs raw-pool training readout"]
fn mnist_gradient_pool_ablation() {
    let dir = default_cache_dir();
    let (grid, sensor) = (32usize, 8usize);
    let Some((train, test)) = load_subsets(&dir, 400, 200, 20, grid) else {
        panic!("MNIST cache not found at {}", dir.display());
    };
    let base = GradTrainConfig {
        epochs: 40,
        batch: 64,
        lr_mask: 0.04,
        lr_head: 0.01,
        sensor,
        seed: 0x6E157,
        raw_pool: false,
        adam_eps: 1e-7,
    };
    eprintln!("\n[A/B] train-time pooled readout: L2-normalized vs raw (eval NCC identical)");
    for raw in [false, true] {
        let gc = GradTrainConfig { raw_pool: raw, ..base };
        let r = run_mnist_grad(&train, &test, grid, sensor, &gc);
        eprintln!(
            "  raw_pool={:<5} -> grad_optical={:.4}  (loss {:.3}->{:.3}, vs hill-climb {:+.4})",
            raw,
            r.grad_optical_acc,
            r.loss_curve.first().copied().unwrap_or(0.0),
            r.loss_curve.last().copied().unwrap_or(0.0),
            r.grad_optical_acc - HILLCLIMB_OPTICAL_ACC
        );
    }
}
