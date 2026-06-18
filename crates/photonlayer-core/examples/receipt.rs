//! receipt — prove what was measured, and catch tampering.
//!
//! Every PhotonLayer run can emit an RVF-style experiment receipt (ADR-260 §15):
//! a set of content hashes binding the input, mask, config, output frame, and
//! metrics, plus build provenance, all folded into one anti-swap digest. This
//! example builds a receipt over a real run, verifies it passes, then tampers a
//! single field and shows verification FAILS — the receipt detects the change.
//!
//! What to look for in the output:
//!   * "verify (clean)    : PASS", then
//!   * "verify (tampered) : FAIL" after flipping one stored hash.
//!
//! Run:
//!   cargo run --release --example receipt -p photonlayer-core

use photonlayer_core::prelude::*;

fn main() {
    let n = 16;

    // A small structured input.
    let pixels: Vec<f32> = (0..n * n).map(|i| (i % 3) as f32 / 2.0).collect();
    let img = InputImage::from_norm_f32(n, n, pixels).expect("image");
    let mask = PhaseMask::random(n, n, 7);
    let cfg = OpticalConfig::demo(n, n);

    // Run the pipeline and capture the sensor frame.
    let frame = ScalarSimulator.simulate(&img, &mask, &cfg).expect("simulate");

    // A minimal metric report (real metrics would be filled by a benchmark).
    let metrics = MetricReport {
        compression_ratio: compression_ratio(&img, &frame),
        ..Default::default()
    };

    // Provenance the receipt binds in (left empty here; CI fills it).
    let prov = Provenance {
        git_commit: "example".into(),
        rustc_version: "example".into(),
        feature_flags: vec![],
    };

    // Build the receipt: hashes of every output-determining input.
    let receipt = build_receipt("hello-receipt", &img, &mask, &cfg, &frame, &metrics, &prov);

    println!("PhotonLayer — receipt");
    println!("  experiment_id      : {}", receipt.experiment_id);
    println!("  input_hash         : {}", receipt.input_hash);
    println!("  mask_hash          : {}", receipt.mask_hash);
    println!("  config_hash        : {}", receipt.config_hash);
    println!("  output_hash        : {}", receipt.output_hash);
    println!("  metrics_hash       : {}", receipt.metrics_hash);
    println!("  rvf_receipt_hash   : {}", receipt.rvf_receipt_hash);

    // 1. Verify the untouched receipt.
    let clean_ok = verify_receipt(&receipt);
    println!("\n  verify (clean)     : {}", if clean_ok { "PASS" } else { "FAIL" });
    assert!(clean_ok, "a freshly built receipt must verify");

    // 2. Tamper exactly one field — pretend an attacker swapped the output frame
    //    but kept the original binding digest.
    let mut tampered = receipt.clone();
    tampered.output_hash.push('x');
    let tampered_ok = verify_receipt(&tampered);
    println!("  verify (tampered)  : {}", if tampered_ok { "PASS (!!)" } else { "FAIL" });
    assert!(!tampered_ok, "tampering must break verification");

    println!("\nThe receipt's single digest binds every input — changing one byte is detected.");
}
