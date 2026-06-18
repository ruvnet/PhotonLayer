//! differential_detection — the Li/Ozcan I⁺−I⁻ readout lever, in isolation.
//!
//! A plain optical classifier reads ONE intensity region per class and takes the
//! argmax. Differential detection reads TWO regions per class and scores
//! `class k = I⁺_k − I⁻_k` — the trick (Li/Ozcan, arXiv:1906.03417) that lifts
//! diffractive-net MNIST accuracy substantially for only +K detector regions and
//! a subtraction. Here we train a phase mask DIRECTLY against the differential
//! argmax objective on the synthetic shapes (no decoder at all), then compare
//! plain argmax vs differential argmax on the SAME trained mask — isolating the
//! readout lever exactly.
//!
//! What to look for in the output:
//!   * "plain argmax" vs "differential argmax" on the identical trained mask,
//!     with differential >= plain (the lever's contribution), and
//!   * both readouts above chance (1/{NUM_CLASSES}) — the optics route class energy.
//!
//! Run:
//!   cargo run --release --example differential_detection -p photonlayer-bench

use photonlayer_bench::diffdetect::DiffDetector;
use photonlayer_bench::synthetic::{class_names, make_dataset, Sample, NUM_CLASSES};
use core::f32::consts::PI;
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::mask::PhaseMask;
use photonlayer_core::rng::DeterministicRng;
use photonlayer_core::simulator::{OpticalSimulator, ScalarSimulator};

/// Fraction of `samples` whose differential argmax equals the label.
fn diff_acc(samples: &[Sample], mask: &PhaseMask, cfg: &OpticalConfig, det: &DiffDetector) -> f32 {
    let correct = samples
        .iter()
        .filter(|s| {
            let frame = ScalarSimulator.simulate(&s.image, mask, cfg).expect("sim");
            det.predict_differential(&frame) == s.label
        })
        .count();
    correct as f32 / samples.len().max(1) as f32
}

/// Fraction of `samples` whose plain (positive-region) argmax equals the label.
fn plain_acc(samples: &[Sample], mask: &PhaseMask, cfg: &OpticalConfig, det: &DiffDetector) -> f32 {
    let correct = samples
        .iter()
        .filter(|s| {
            let frame = ScalarSimulator.simulate(&s.image, mask, cfg).expect("sim");
            det.predict_plain(&frame) == s.label
        })
        .count();
    correct as f32 / samples.len().max(1) as f32
}

fn main() {
    let n = 32; // need room for 2*NUM_CLASSES detector tiles
    let cfg = OpticalConfig::demo(n, n);
    let det = DiffDetector::new(NUM_CLASSES, n, n);

    let data = make_dataset(n, 32, 0xD1FF);
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for (i, s) in data.iter().enumerate() {
        if i % 2 == 0 { train.push(s.clone()); } else { test.push(s.clone()); }
    }

    println!("PhotonLayer — differential_detection  ({} classes: {:?})", NUM_CLASSES, class_names());
    println!("  grid={n}x{n}  detector regions read: {} (= 2 x {})", det.readout_regions, NUM_CLASSES);
    println!("  chance accuracy = {:.3}\n", 1.0 / NUM_CLASSES as f32);

    // Train a mask directly so argmax_k (I⁺_k - I⁻_k) hits the label — seeded
    // block hill-climb on the TRAIN differential accuracy (no decoder). The
    // differential objective is init-sensitive (the README flags this), so we
    // train from several seeds and SELECT by TRAIN accuracy — legitimate model
    // selection, never peeking at the test set.
    let (block, sigma, iters) = (7usize, 1.0f32, 3000usize);
    let train_diff_mask = |seed: u64| -> (PhaseMask, f32) {
        let mut rng = DeterministicRng::new(seed);
        let mut mask = PhaseMask::random(n, n, seed);
        let mut score = diff_acc(&train, &mask, &cfg, &det);
        for _ in 0..iters {
            let mut cand = mask.clone();
            let bx = (rng.next_f32() * (n.saturating_sub(block) + 1) as f32) as usize;
            let by = (rng.next_f32() * (n.saturating_sub(block) + 1) as f32) as usize;
            for dy in 0..block { for dx in 0..block {
                let idx = (by + dy).min(n - 1) * n + (bx + dx).min(n - 1);
                cand.phase_radians[idx] = (cand.phase_radians[idx] + rng.next_gaussian() * sigma).rem_euclid(2.0 * PI);
            }}
            let c = diff_acc(&train, &cand, &cfg, &det);
            if c > score { mask = cand; score = c; }
        }
        (mask, score)
    };

    let mut best: Option<(PhaseMask, u64, f32)> = None;
    println!("  training (differential objective) across seeds, selecting by TRAIN acc:");
    for seed in [0x6E157u64, 0xABCD, 0x1234, 0xBEEF, 0xF00D] {
        let (mask, train_score) = train_diff_mask(seed);
        println!("    seed {seed:#x}: train_diff_acc={train_score:.3}");
        if best.as_ref().map(|b| train_score > b.2).unwrap_or(true) {
            best = Some((mask, seed, train_score));
        }
    }
    let (mask, seed, train_score) = best.expect("at least one seed");

    // Same trained mask, two readouts — isolates the lever.
    let plain = plain_acc(&test, &mask, &cfg, &det);
    let differential = diff_acc(&test, &mask, &cfg, &det);

    println!("\n  SELECTED mask (seed {seed:#x}, train_diff_acc={train_score:.3}):");
    println!("    plain argmax        (reads {} regions): test_acc={plain:.3}", NUM_CLASSES);
    println!("    differential argmax (reads {} regions): test_acc={differential:.3}", det.readout_regions);
    println!("    lever Δ (differential - plain) = {:+.3}", differential - plain);

    println!(
        "\nReading I⁺−I⁻ instead of I⁺ alone changes accuracy by {:+.3} on the same mask.",
        differential - plain
    );
}
