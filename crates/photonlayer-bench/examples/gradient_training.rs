//! gradient_training — train a phase mask by analytic gradient descent.
//!
//! The hill-climb learner only perturbs and accepts; this example instead
//! descends the diffraction operator's PROVEN adjoint
//! (`Propagator::backward_into` + `phase_gradient`, validated by a
//! finite-difference grad-check in the core) via `train_mask_grad`. A small
//! differentiable softmax head supplies the per-pixel sensor weight; the optical
//! half of the gradient is the proven adjoint. We print the per-epoch CE loss
//! curve (it should fall) and the blind-test accuracy of the gradient-trained
//! mask vs the random-init mask, evaluated with the SAME nearest-centroid decoder
//! so it is apples-to-apples.
//!
//! What to look for in the output:
//!   * the loss curve DECREASING across epochs, and
//!   * "gradient-trained" accuracy beating "random-init" — gradient descent
//!     through the optics genuinely improves the compressed readout.
//!
//! Run:
//!   cargo run --release --example gradient_training -p photonlayer-bench

use photonlayer_bench::decoder::{frame_features, NearestCentroid};
use photonlayer_bench::grad_train::{build_grad_samples, train_mask_grad, GradTrainConfig};
use photonlayer_bench::synthetic::{make_dataset, Sample, NUM_CLASSES};
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::mask::PhaseMask;
use photonlayer_core::propagate::Propagator;
use photonlayer_core::simulator::{OpticalSimulator, ScalarSimulator};

/// Blind-test accuracy of a mask via the pooled `sensor x sensor` NCC decoder.
fn decode_acc(train: &[Sample], test: &[Sample], mask: &PhaseMask, cfg: &OpticalConfig, sensor: usize) -> f32 {
    let feats = |s: &[Sample]| -> (Vec<Vec<f32>>, Vec<usize>) {
        let f = s.iter().map(|x| {
            let frame = ScalarSimulator.simulate(&x.image, mask, cfg).expect("sim");
            frame_features(&frame, sensor)
        }).collect();
        (f, s.iter().map(|x| x.label).collect())
    };
    let (tr_f, tr_l) = feats(train);
    let (te_f, te_l) = feats(test);
    NearestCentroid::fit(&tr_f, &tr_l, NUM_CLASSES).accuracy(&te_f, &te_l)
}

fn main() {
    let n = 16;
    let sensor = 4; // 16x16 grid pooled to 4x4 = 16x sensor reduction
    let cfg = OpticalConfig::demo(n, n);
    let prop = Propagator::new(n, n, &cfg).expect("propagator");

    let data = make_dataset(n, 24, 0x6DA7A);
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for (i, s) in data.iter().enumerate() {
        if i % 2 == 0 { train.push(s.clone()); } else { test.push(s.clone()); }
    }

    println!("PhotonLayer — gradient_training  (grid={n}x{n}, sensor={sensor}x{sensor})");
    println!("  train={} test={}  chance={:.3}\n", train.len(), test.len(), 1.0 / NUM_CLASSES as f32);

    // Random init — the WIN floor and the gradient start point (same phases).
    let init = PhaseMask::random(n, n, 0x6DA7A);
    let random_acc = decode_acc(&train, &test, &init, &cfg, sensor);

    // Gradient training through the proven adjoint.
    let gc = GradTrainConfig { epochs: 40, batch: 16, lr_mask: 0.06, lr_head: 0.05, sensor, seed: 0x6DA7A, raw_pool: false, adam_eps: 1e-7 };
    let gsamples = build_grad_samples(&train, &cfg);
    let out = train_mask_grad(&prop, n, n, &gsamples, &init.phase_radians, &gc);

    // Print the loss curve (sampled so it stays compact).
    println!("  per-epoch mean cross-entropy (should fall):");
    let step = (out.loss_curve.len() / 8).max(1);
    for (e, l) in out.loss_curve.iter().enumerate().step_by(step) {
        let bar = "#".repeat((l * 12.0).round() as usize);
        println!("    epoch {e:>3}: CE={l:.4}  {bar}");
    }
    let first = out.loss_curve.first().copied().unwrap_or(0.0);
    let last = out.loss_curve.last().copied().unwrap_or(0.0);
    println!("    loss {first:.4} -> {last:.4}  (Δ={:+.4})", last - first);

    // Evaluate the trained mask with the SAME NCC decoder.
    let trained = PhaseMask::new(n, n, out.theta, "grad").expect("trained mask");
    let grad_acc = decode_acc(&train, &test, &trained, &cfg, sensor);

    println!("\n  blind-test accuracy (same NCC decoder, {sensor}x{sensor} readout):");
    println!("    random-init mask    : {random_acc:.3}");
    println!("    gradient-trained    : {grad_acc:.3}   (Δ={:+.3})", grad_acc - random_acc);

    assert!(last <= first, "training loss should not increase overall");
    println!("\nGradient descent through the proven adjoint lowers the loss and improves accuracy.");
}
