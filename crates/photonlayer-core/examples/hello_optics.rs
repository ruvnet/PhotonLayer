//! hello_optics — the minimal PhotonLayer pipeline, end to end.
//!
//! Demonstrates the whole optical core in one screen: an image becomes a scalar
//! optical field, a phase mask shapes it, free-space propagation diffracts it,
//! and a sensor records an intensity frame. The frame carries a BLAKE3 hash that
//! binds its dimensions + values (the determinism invariant, ADR-260 §21).
//!
//! What to look for in the output:
//!   * a non-trivial `frame_hash` printed for the captured sensor frame, and
//!   * a SECOND run producing the *identical* hash — the pipeline is
//!     bit-reproducible for the same input + mask + config.
//!
//! Run:
//!   cargo run --release --example hello_optics -p photonlayer-core

use photonlayer_core::prelude::*;

fn main() {
    let n = 32;

    // 1. Input image: a simple horizontal intensity ramp in [0, 1].
    let pixels: Vec<f32> = (0..n * n).map(|i| (i % n) as f32 / n as f32).collect();
    let img = InputImage::from_norm_f32(n, n, pixels).expect("well-formed image");

    // 2. A deterministic random phase mask (seeded -> reproducible).
    let mask = PhaseMask::random(n, n, 42);

    // 3. A small green-light far-field config (532 nm, angular-spectrum).
    let cfg = OpticalConfig::demo(n, n);

    // 4. Run the full pipeline: image -> field -> mask -> propagate -> sensor.
    let frame = ScalarSimulator
        .simulate(&img, &mask, &cfg)
        .expect("simulation succeeds");

    // The trace exposes every intermediate stage (used by the studio UI).
    let trace = ScalarSimulator
        .trace(&img, &mask, &cfg)
        .expect("trace succeeds");

    println!("PhotonLayer — hello_optics");
    println!("  grid              : {}x{}", n, n);
    println!("  propagation       : {:?}", cfg.propagation);
    println!("  wavelength        : {} nm", cfg.wavelength_nm);
    println!("  input power        = {:.4}", trace.incoming.power());
    println!("  power after mask   = {:.4}  (phase-only: conserved)", trace.masked.power());
    println!("  power at sensor    = {:.4}", trace.propagated.power());
    println!("  sensor frame       : {}x{}", frame.width, frame.height);
    println!("  frame_hash         : {}", frame.frame_hash);

    // 5. Determinism: re-run and confirm the hash is bit-identical.
    let frame2 = ScalarSimulator
        .simulate(&img, &mask, &cfg)
        .expect("re-simulation succeeds");
    let identical = frame.frame_hash == frame2.frame_hash;
    println!("  re-run frame_hash  : {}", frame2.frame_hash);
    println!(
        "  deterministic?     : {}",
        if identical { "YES — hashes match" } else { "NO — MISMATCH" }
    );

    assert!(identical, "determinism invariant violated");
    println!("\nThe same input + mask + config always produces the same sensor frame.");
}
