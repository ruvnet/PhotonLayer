//! privacy_probe — a linear reconstruction attack on the optical measurement.
//!
//! When an optical mask diffuses and encodes the input before any pixel hits the
//! sensor, the raw detector pattern is not a human-readable image. To quantify
//! this we run a *reconstruction attack*: fit a ridge-regularized LINEAR inverse
//! map from the compact optical feature vector back to the (downsampled) input,
//! then measure how faithfully it reconstructs held-out images (PSNR, dB). An
//! identity mask (no optics) leaks heavily; a random/learned optical mask
//! scrambles the measurement, so the linear attack reconstructs poorly.
//!
//! HONEST CAVEAT (printed in the output too): this is a LINEAR lower bound on
//! leakage. A nonlinear attacker (e.g. a trained network) is expected to recover
//! more. Low linear-attack PSNR is evidence the measurement is not trivially the
//! image — it is NOT a privacy guarantee. Never imply privacy.
//!
//! What to look for in the output:
//!   * the identity (no-optics) mask leaking with HIGH reconstruction PSNR, and
//!   * the optical masks (random, learned) leaking LESS (lower PSNR / leakage),
//!     plus the explicit "linear lower bound, not a guarantee" caveat.
//!
//! Run:
//!   cargo run --release --example privacy_probe -p photonlayer-bench

use photonlayer_bench::learn::{learn_mask, LearnConfig};
use photonlayer_bench::privacy::privacy_leakage;
use photonlayer_bench::synthetic::make_dataset;
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::mask::PhaseMask;

fn main() {
    let n = 16;
    let feat = 4; // keep the Gram matrix tiny (16x16 solve)
    let cfg = OpticalConfig::demo(n, n);

    let samples = make_dataset(n, 12, 0xCAFE);

    println!("PhotonLayer — privacy_probe  (linear reconstruction attack, grid={n}x{n})");
    println!("  attack: ridge-linear inverse  features -> {feat}x{feat} image; higher PSNR = more leakage\n");

    // 1. Identity mask = no optical processing -> detector ≈ input -> leaks.
    let identity = PhaseMask::identity(n, n);
    let id_rep = privacy_leakage(&samples, &identity, &cfg, feat);
    println!("  identity mask (NO optics):");
    println!("    reconstruction PSNR = {:>6.2} dB   leakage = {:.3}   |corr(frame,input)| = {:.3}",
        id_rep.reconstruction_psnr, id_rep.leakage_score, id_rep.frame_input_similarity);

    // 2. Random optical mask -> scrambled measurement.
    let random = PhaseMask::random(n, n, 0xDEAD);
    let rnd_rep = privacy_leakage(&samples, &random, &cfg, feat);
    println!("  random optical mask:");
    println!("    reconstruction PSNR = {:>6.2} dB   leakage = {:.3}   |corr(frame,input)| = {:.3}",
        rnd_rep.reconstruction_psnr, rnd_rep.leakage_score, rnd_rep.frame_input_similarity);

    // 3. Learned optical mask (frozen after hill-climb training).
    let lc = LearnConfig { iterations: 150, feat_dim: feat, ..Default::default() };
    let learned = learn_mask(&samples, &cfg, &lc).mask;
    let lrn_rep = privacy_leakage(&samples, &learned, &cfg, feat);
    println!("  learned optical mask:");
    println!("    reconstruction PSNR = {:>6.2} dB   leakage = {:.3}   |corr(frame,input)| = {:.3}",
        lrn_rep.reconstruction_psnr, lrn_rep.leakage_score, lrn_rep.frame_input_similarity);

    println!("\n  optical masks leak less under the LINEAR attack than the identity mask:");
    println!("    identity {:.2} dB -> random {:.2} dB ({:+.2}), learned {:.2} dB ({:+.2})",
        id_rep.reconstruction_psnr,
        rnd_rep.reconstruction_psnr, rnd_rep.reconstruction_psnr - id_rep.reconstruction_psnr,
        lrn_rep.reconstruction_psnr, lrn_rep.reconstruction_psnr - id_rep.reconstruction_psnr);

    println!("\n  CAVEAT: this is a LINEAR lower bound on leakage. A nonlinear attacker");
    println!("  (a trained network) is expected to recover more. Low linear-attack PSNR");
    println!("  shows the measurement is not trivially the image — it is NOT a privacy");
    println!("  guarantee. This is sim-only and makes no privacy claim.");
}
