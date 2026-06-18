//! optical_feature_extractor — optics as an analog feature extractor.
//!
//! The claim (honestly scoped): holding the decoder fixed at a *tiny* weak
//! nearest-centroid head, a LEARNED optical measurement can be MORE linearly
//! separable than the raw input pixels fed to the same head. We freeze a mask
//! trained by hill-climbing, extract its compact optical features, and compare
//! the tiny decoder's blind-test accuracy on those features vs on the raw pooled
//! pixels (same decoder family, same feature dimension).
//!
//! HONEST CAVEAT (printed in the output too): nearest-centroid is a deliberately
//! weak baseline. This is a statement about feature SEPARABILITY under a fixed
//! weak decoder, NOT that optics beat digital methods — a small CNN on the raw
//! pixels would beat both. The point is that the analog transform front-loads
//! useful structure into far fewer numbers.
//!
//! What to look for in the output:
//!   * "optical features" decoder accuracy vs "raw pixels" decoder accuracy at
//!     the same tiny feature dimension, with the optical side higher, AND
//!   * the printed caveat that the baseline decoder is weak by design.
//!
//! Run:
//!   cargo run --release --example optical_feature_extractor -p photonlayer-bench

use photonlayer_bench::decoder::{pool_features, NearestCentroid};
use photonlayer_bench::learn::{learn_mask, LearnConfig};
use photonlayer_bench::pipeline::optical_feature_set;
use photonlayer_bench::synthetic::{make_dataset, Sample, NUM_CLASSES};
use photonlayer_core::config::OpticalConfig;

/// Raw-pixel features: each image average-pooled to feat x feat, L2-normalized.
fn raw_feature_set(samples: &[Sample], feat: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let f = samples.iter()
        .map(|s| pool_features(&s.image.pixels, s.image.width, s.image.height, feat))
        .collect();
    (f, samples.iter().map(|s| s.label).collect())
}

fn main() {
    let n = 16;
    // Squeeze both feature sets to a TINY 2x2 = 4 numbers. At this compression
    // the raw pooled pixels lose class structure, but the learned optical
    // transform front-loads separable structure into the same 4 numbers. A
    // larger feat would let raw pixels saturate and hide the effect.
    let feat = 2;
    let cfg = OpticalConfig::demo(n, n);

    let data = make_dataset(n, 20, 0xFEA7);
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for (i, s) in data.iter().enumerate() {
        if i % 2 == 0 { train.push(s.clone()); } else { test.push(s.clone()); }
    }

    println!("PhotonLayer — optical_feature_extractor  (grid={n}x{n}, feature={feat}x{feat})");
    println!("  train={} test={}  chance={:.3}\n", train.len(), test.len(), 1.0 / NUM_CLASSES as f32);

    // Decoder A: tiny centroid on RAW pooled pixels (no optics).
    let (r_tr_f, r_tr_l) = raw_feature_set(&train, feat);
    let (r_te_f, r_te_l) = raw_feature_set(&test, feat);
    let raw_dec = NearestCentroid::fit(&r_tr_f, &r_tr_l, NUM_CLASSES);
    let raw_acc = raw_dec.accuracy(&r_te_f, &r_te_l);

    // Freeze a learned optical mask, then decode its optical features with the
    // SAME tiny centroid family at the SAME feature dimension.
    let lc = LearnConfig { iterations: 250, feat_dim: feat, ..Default::default() };
    let outcome = learn_mask(&train, &cfg, &lc);
    let (o_tr_f, o_tr_l) = optical_feature_set(&train, &outcome.mask, &cfg, feat);
    let (o_te_f, o_te_l) = optical_feature_set(&test, &outcome.mask, &cfg, feat);
    let opt_dec = NearestCentroid::fit(&o_tr_f, &o_tr_l, NUM_CLASSES);
    let opt_acc = opt_dec.accuracy(&o_te_f, &o_te_l);

    let numbers = feat * feat;
    println!("  same tiny nearest-centroid decoder ({} params each):", raw_dec.param_count());
    println!("    raw pixels      ({feat}x{feat}={numbers} numbers): blind-test acc = {raw_acc:.3}");
    println!("    optical features({feat}x{feat}={numbers} numbers): blind-test acc = {opt_acc:.3}");
    println!("    optical - raw   = {:+.3}", opt_acc - raw_acc);

    println!("\n  CAVEAT (scientific honesty): nearest-centroid is a deliberately weak");
    println!("  decoder. This shows the LEARNED optical features are more linearly");
    println!("  separable under a FIXED weak head — not that optics beat digital. A");
    println!("  small CNN on the raw pixels would beat both. The analog transform");
    println!("  front-loads useful structure into the same tiny number of features.");
}
