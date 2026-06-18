//! Multi-plane diffractive cascade MNIST benchmark — does stacking phase planes
//! beat the single-plane gradient result? (ADR-260 roadmap, "the cascade".)
//!
//! The single-plane gradient trainer reaches 83.30% MNIST blind-test at 16x
//! compression (see `mnist_gradient_bench::mnist_gradient_full`). This test
//! trains a `K`-plane cascade (a small D2NN) end-to-end through the SAME proven
//! adjoint — composed across planes — and measures whether K=2 / K=3 beat that
//! 83.30% single-plane number on the IDENTICAL NCC-decoder eval.
//!
//! Train-time uses the differentiable linear+softmax+CE head over the pooled
//! sensor readout; EVAL uses the deterministic nearest-centroid decoder on the
//! cascade's `|u_K|^2` readout — the exact same feature path (`pool_features`)
//! the single-plane eval uses, so the comparison is apples-to-apples and the
//! only thing that changed is the NUMBER OF PHASE PLANES.
//!
//! Dataset is NOT vendored — fetch the public IDX files once into
//! `crates/photonlayer-bench/data/mnist/` (see `mnist_differential_bench.rs`
//! header for the exact curl command). This test skips/panics cleanly if absent.
//!
//! Run (AV-safe integration test):
//! ```text
//! cargo test -p photonlayer-bench --release --test mnist_cascade_bench \
//!     mnist_cascade_full -- --ignored --nocapture
//! ```

use photonlayer_bench::decoder::{pool_features, NearestCentroid};
use photonlayer_bench::grad_cascade::{train_cascade_grad, Cascade, CascadeSample, CascadeTrainConfig};
use photonlayer_bench::mnist::{self, default_cache_dir, MNIST_CLASSES};
use photonlayer_bench::synthetic::Sample;
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::field::OpticalField;
use photonlayer_core::mask::PhaseMask;
use photonlayer_core::propagate::Propagator;
use std::path::Path;

/// The PROVEN single-plane gradient result this cascade is trying to beat
/// (NCC decoder, 16x compression, 400/200 per class) — reproduced from
/// `mnist_gradient_bench::mnist_gradient_full` (83.30%).
const SINGLE_PLANE_GRAD_ACC: f32 = 0.8330;

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

/// Build incident fields once (image -> centered amplitude), reused every epoch.
fn build_cascade_samples(samples: &[Sample], grid: usize) -> Vec<CascadeSample> {
    samples
        .iter()
        .map(|s| CascadeSample {
            u0: OpticalField::from_image(&s.image, grid, grid)
                .expect("image fits grid")
                .data,
            label: s.label,
        })
        .collect()
}

/// NCC blind-test accuracy for a fixed set of `K` phase planes. Runs the cascade
/// forward to `|u_K|^2`, then the SAME `pool_features` (avg-pool + L2-norm) the
/// single-plane eval uses, then nearest-centroid fit-on-train / score-on-test.
/// This is the cascade analogue of `mnist_bench::decode_optical_acc`.
fn cascade_ncc_acc(
    prop: &Propagator,
    planes: &[Vec<f32>],
    grid: usize,
    sensor: usize,
    train: &[CascadeSample],
    test: &[CascadeSample],
) -> f32 {
    let n = grid * grid;
    let cascade = Cascade::new(prop, planes.to_vec(), n);
    let feats = |data: &[CascadeSample]| -> (Vec<Vec<f32>>, Vec<usize>) {
        let f = data
            .iter()
            .map(|s| {
                let intensity = cascade.intensity(&s.u0);
                pool_features(&intensity, grid, grid, sensor)
            })
            .collect();
        let l = data.iter().map(|s| s.label).collect();
        (f, l)
    };
    let (tr_f, tr_l) = feats(train);
    let (te_f, te_l) = feats(test);
    NearestCentroid::fit(&tr_f, &tr_l, MNIST_CLASSES).accuracy(&te_f, &te_l)
}

/// Deterministic per-plane init: each plane gets a distinct random phase (so the
/// cascade does not start with K identical planes), derived from the master seed.
fn init_planes(grid: usize, planes: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..planes)
        .map(|p| {
            // Distinct, deterministic seed per plane; plane 0 reuses `seed` so the
            // first plane's init matches the single-plane random-init family.
            let s = seed ^ (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            PhaseMask::random(grid, grid, s).phase_radians
        })
        .collect()
}

/// Run one cascade configuration end-to-end and return
/// `(grad_acc, random_init_acc, first_loss, last_loss)`.
fn run_cascade_config(
    prop: &Propagator,
    grid: usize,
    sensor: usize,
    train: &[CascadeSample],
    test: &[CascadeSample],
    cc: &CascadeTrainConfig,
) -> (f32, f32, f32, f32) {
    let theta0 = init_planes(grid, cc.planes, cc.seed);
    // WIN floor: NCC on the random-init planes (what training must beat).
    let random_acc = cascade_ncc_acc(prop, &theta0, grid, sensor, train, test);
    // Gradient training through the composed proven adjoint.
    let out = train_cascade_grad(prop, grid, grid, train, &theta0, cc);
    let grad_acc = cascade_ncc_acc(prop, &out.theta, grid, sensor, train, test);
    let first = out.loss_curve.first().copied().unwrap_or(0.0);
    let last = out.loss_curve.last().copied().unwrap_or(0.0);
    (grad_acc, random_acc, first, last)
}

/// THE CASCADE RUN: full-scale K=2 and K=3 gradient training, blind-test accuracy
/// measured with the NCC decoder, printed against the 83.30% single-plane result.
/// Heavy + deterministic; `#[ignore]` (run on demand).
///
/// HONESTY: this prints the REAL measured accuracy. The asserted claim is only
/// the robust one — each cascade beats its OWN random-init planes. The cascade-
/// vs-single-plane delta is REPORTED (the headline), not forced green, so a
/// shortfall is visible rather than hidden.
#[test]
#[ignore = "heavy real-data cascade gradient-training run; see header for the command"]
fn mnist_cascade_full() {
    let dir = default_cache_dir();
    let grid = 32;
    let sensor = 8;
    let cell = 20;
    // Same train/test budget + seed family as the single-plane full test.
    let Some((train_s, test_s)) = load_subsets(&dir, 400, 200, cell, grid) else {
        panic!("MNIST cache not found at {} - fetch IDX files (see header) first", dir.display());
    };
    let train = build_cascade_samples(&train_s, grid);
    let test = build_cascade_samples(&test_s, grid);

    // Shared optical operator: AngularSpectrum at the demo inter-plane distance
    // (10mm), the SAME P the single plane uses — one inter-plane gap per stage.
    let cfg = OpticalConfig::demo(grid, grid);
    let prop = Propagator::new(grid, grid, &cfg).expect("propagator");

    // Headline cascade config (mirrors the single-plane full config: L2-norm
    // train feature, Adam eps=1e-7, seed 0x6E157, 16x compression).
    let base = CascadeTrainConfig {
        planes: 2,
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
    let (k2_acc, k2_rand, k2_l0, k2_l1) =
        run_cascade_config(&prop, grid, sensor, &train, &test, &CascadeTrainConfig { planes: 2, ..base });
    let (k3_acc, k3_rand, k3_l0, k3_l1) =
        run_cascade_config(&prop, grid, sensor, &train, &test, &CascadeTrainConfig { planes: 3, ..base });
    eprintln!("[timing] full cascade run (K=2 + K=3): {:.0}s", t0.elapsed().as_secs_f32());

    let baseline_pixels = grid * grid;
    let sensor_px = sensor * sensor;
    let compression = baseline_pixels as f32 / sensor_px as f32;

    eprintln!("\n===== PhotonLayer MNIST: MULTI-PLANE CASCADE vs single-plane gradient =====");
    eprintln!("dataset      : MNIST handwritten digits (public IDX, ossci-datasets mirror)");
    eprintln!(
        "optics       : {grid}x{grid} field, AngularSpectrum, {sensor}x{sensor} pooled sensor, {compression:.0}x compression"
    );
    eprintln!(
        "train / test : {} / {} images, balanced across 10 classes (blind test split)",
        train.len(),
        test.len()
    );
    eprintln!(
        "training     : end-to-end gradient descent through the PROVEN composed adjoint; Adam;"
    );
    eprintln!(
        "               epochs={} lr_mask={} seed={:#x} (deterministic, no FMA/SIMD)",
        base.epochs, base.lr_mask, base.seed
    );
    eprintln!("  eval decoder: nearest-centroid (same feature path as single-plane -> apples-to-apples)");
    eprintln!("------------------------------------------------------------------------------");
    eprintln!("  CONFIG        random-init   GRADIENT-trained   loss (first->last)");
    eprintln!("  single-plane  {:>10}   {SINGLE_PLANE_GRAD_ACC:>16.4}   (reproduced 83.30%)", "(ref)");
    eprintln!("  2-plane       {k2_rand:>10.4}   {k2_acc:>16.4}   {k2_l0:.3}->{k2_l1:.3}");
    eprintln!("  3-plane       {k3_rand:>10.4}   {k3_acc:>16.4}   {k3_l0:.3}->{k3_l1:.3}");
    eprintln!("------------------------------------------------------------------------------");
    eprintln!("  THE HEADLINE (cascade gradient-trained vs single-plane 83.30%):");
    eprintln!(
        "    2-plane - single-plane   {:>+7.4}   ({})",
        k2_acc - SINGLE_PLANE_GRAD_ACC,
        if k2_acc > SINGLE_PLANE_GRAD_ACC { "2-PLANE WINS" } else { "did not beat single-plane" }
    );
    eprintln!(
        "    3-plane - single-plane   {:>+7.4}   ({})",
        k3_acc - SINGLE_PLANE_GRAD_ACC,
        if k3_acc > SINGLE_PLANE_GRAD_ACC { "3-PLANE WINS" } else { "did not beat single-plane" }
    );
    eprintln!(
        "    2-plane - random init    {:>+7.4}   (what gradient training bought, K=2)",
        k2_acc - k2_rand
    );
    eprintln!(
        "    3-plane - random init    {:>+7.4}   (what gradient training bought, K=3)",
        k3_acc - k3_rand
    );
    eprintln!("==============================================================================\n");

    // Robustly-true assertions: each cascade improves over its own random start.
    assert!(
        k2_acc >= k2_rand + 0.02,
        "K=2 gradient {k2_acc:.4} did not beat its random init {k2_rand:.4} by >= 0.02"
    );
    assert!(
        k3_acc >= k3_rand + 0.02,
        "K=3 gradient {k3_acc:.4} did not beat its random init {k3_rand:.4} by >= 0.02"
    );
    assert!(compression >= 16.0, "compression {compression:.1}x < 16x");
}
