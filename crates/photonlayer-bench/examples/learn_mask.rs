//! learn_mask — train a phase mask by hill-climbing on synthetic shapes.
//!
//! The synthetic dataset is four geometric shape classes (vbar, hbar, diag,
//! ring) rendered deterministically — no MNIST download, no network. We measure
//! classification accuracy of the tiny nearest-centroid decoder on the optical
//! readout for a RANDOM phase mask, then run seeded block hill-climbing
//! ([`learn_mask`]) which only accepts improving steps, and measure the LEARNED
//! mask. Learning the optics must beat (or at least match) the random start.
//!
//! What to look for in the output:
//!   * "random mask" train/test accuracy, then
//!   * "learned mask" accuracy strictly >= random on training (hill-climb only
//!     accepts improvements), usually higher on the blind test too.
//!
//! Run:
//!   cargo run --release --example learn_mask -p photonlayer-bench

use photonlayer_bench::decoder::NearestCentroid;
use photonlayer_bench::learn::{learn_mask, LearnConfig};
use photonlayer_bench::pipeline::optical_feature_set;
use photonlayer_bench::synthetic::{class_names, make_dataset, NUM_CLASSES};
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::mask::PhaseMask;

/// Train/test accuracy of a mask via the deterministic centroid decoder.
fn eval(mask: &PhaseMask, train: &[photonlayer_bench::synthetic::Sample],
        test: &[photonlayer_bench::synthetic::Sample], cfg: &OpticalConfig, feat: usize) -> (f32, f32) {
    let (tr_f, tr_l) = optical_feature_set(train, mask, cfg, feat);
    let (te_f, te_l) = optical_feature_set(test, mask, cfg, feat);
    let ncc = NearestCentroid::fit(&tr_f, &tr_l, NUM_CLASSES);
    (ncc.accuracy(&tr_f, &tr_l), ncc.accuracy(&te_f, &te_l))
}

fn main() {
    let n = 16;
    // A TINY 2x2 sensor (4 measurements). At this compression a random mask
    // loses class structure, leaving real headroom for learning to recover it —
    // a larger sensor would let even a random mask saturate and hide the effect.
    let feat = 2;
    let cfg = OpticalConfig::demo(n, n);

    // Deterministic synthetic data: 12 samples/class, split even=train odd=test.
    let data = make_dataset(n, 12, 0xDA7A);
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for (i, s) in data.iter().enumerate() {
        if i % 2 == 0 { train.push(s.clone()); } else { test.push(s.clone()); }
    }

    println!("PhotonLayer — learn_mask  ({} classes: {:?})", NUM_CLASSES, class_names());
    println!("  grid={n}x{n}  feature={feat}x{feat}  train={} test={}\n", train.len(), test.len());

    // 1. Random mask baseline.
    let random = PhaseMask::random(n, n, 0x5EED);
    let (r_tr, r_te) = eval(&random, &train, &test, &cfg, feat);
    println!("  random mask   : train_acc={r_tr:.3}  test_acc={r_te:.3}");

    // 2. Learn the mask by hill-climbing (accepts only improving steps).
    let lc = LearnConfig { iterations: 200, feat_dim: feat, ..Default::default() };
    let outcome = learn_mask(&train, &cfg, &lc);
    let (l_tr, l_te) = eval(&outcome.mask, &train, &test, &cfg, feat);
    println!("  learned mask  : train_acc={l_tr:.3}  test_acc={l_te:.3}");

    println!(
        "\n  hill-climb training score: {:.3} -> {:.3}  (Δ={:+.3})",
        outcome.start_score.accuracy, outcome.final_score.accuracy,
        outcome.final_score.accuracy - outcome.start_score.accuracy
    );
    println!("  learned mask id: {}", outcome.mask.mask_id);

    println!("\n  random -> learned  train Δ = {:+.3},  test Δ = {:+.3}", l_tr - r_tr, l_te - r_te);
    assert!(l_tr >= r_tr - 1e-6, "learned must not regress training accuracy");
    println!("Learning the optics improves how separable the compressed readout is.");
}
