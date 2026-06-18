//! wavefront_focus — a designed phase lens concentrates a plane wave to a spot.
//!
//! A flat (uniform) wavefront carries its energy spread evenly across the grid.
//! Stamping a quadratic *lens* phase onto it (the hand-designed `PhaseMask::lens`
//! baseline) curves the wavefront so free-space propagation brings it to a focus
//! — a "learned lens" expressed purely as a phase-only optical element. We
//! measure the intensity CONCENTRATION RATIO at the detector: peak focal
//! intensity vs the mean, for a flat mask vs the lens. A real lens concentrates;
//! a flat mask does not.
//!
//! What to look for in the output:
//!   * the lens mask producing a much higher peak/mean concentration ratio than
//!     the flat mask, and a small high-intensity "focal" region.
//!   * a sweep over focal strength showing the concentration peak at a sweet spot
//!     (too weak: no focus; too strong: the phase wraps and the spot smears).
//!
//! Run:
//!   cargo run --release --example wavefront_focus -p photonlayer-core

use photonlayer_core::prelude::*;

/// A uniform plane wave entering the grid (every cell unit amplitude).
fn plane_wave(n: usize) -> InputImage {
    InputImage::from_norm_f32(n, n, vec![1.0; n * n]).expect("plane wave")
}

/// Concentration stats of a sensor frame: peak/mean ratio and the fraction of
/// pixels holding 90% of the energy (smaller = tighter focus).
struct Focus {
    peak_over_mean: f32,
    hot_pixels_for_90pct: usize,
    total_pixels: usize,
}

fn concentration(frame: &OpticalFrame) -> Focus {
    let n = frame.width * frame.height;
    let total: f32 = frame.intensity.iter().sum::<f32>().max(1e-12);
    let mean = total / n as f32;
    let peak = frame.intensity.iter().cloned().fold(0.0f32, f32::max);

    // Sort intensities descending; count how many top pixels reach 90% energy.
    let mut sorted = frame.intensity.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0f32;
    let mut hot = 0usize;
    for &v in &sorted {
        cum += v;
        hot += 1;
        if cum >= 0.9 * total {
            break;
        }
    }

    Focus {
        peak_over_mean: peak / mean.max(1e-12),
        hot_pixels_for_90pct: hot,
        total_pixels: n,
    }
}

fn focus_with(mask: &PhaseMask, cfg: &OpticalConfig, img: &InputImage, label: &str) -> Focus {
    let frame = ScalarSimulator.simulate(img, mask, cfg).expect("simulate");
    let f = concentration(&frame);
    println!(
        "  {:<22} peak/mean={:>8.2}   90%-energy in {:>5}/{} px",
        label, f.peak_over_mean, f.hot_pixels_for_90pct, f.total_pixels
    );
    f
}

fn main() {
    let n = 64;
    let img = plane_wave(n);
    // Fresnel near-field is where a lens brings a beam to focus.
    let mut cfg = OpticalConfig::demo(n, n);
    cfg.propagation = PropagationMode::Fresnel;
    cfg.propagation_mm = 6.0;

    println!("PhotonLayer — wavefront_focus  (plane wave on {n}x{n}, Fresnel z=6mm)\n");

    // Flat reference: no lensing, energy stays spread.
    let flat = focus_with(&PhaseMask::identity(n, n), &cfg, &img, "flat (no lens)");

    // A designed quadratic lens phase — the strength controls the curvature.
    println!("\n  Designed lens phase, sweeping focal strength:");
    let mut best_ratio = flat.peak_over_mean;
    let mut best_strength = 0.0f32;
    for &strength in &[0.005f32, 0.01, 0.02, 0.04, 0.08, 0.16] {
        let lens = PhaseMask::lens(n, n, strength);
        let f = focus_with(&lens, &cfg, &img, &format!("lens(strength={strength})"));
        if f.peak_over_mean > best_ratio {
            best_ratio = f.peak_over_mean;
            best_strength = strength;
        }
    }

    println!(
        "\n  Best focus: strength={best_strength} gives peak/mean={best_ratio:.2} \
         vs {:.2} flat ({:.1}x tighter).",
        flat.peak_over_mean,
        best_ratio / flat.peak_over_mean.max(1e-12)
    );
    println!("A phase-only lens curves the wavefront so propagation concentrates the light.");

    assert!(
        best_ratio > flat.peak_over_mean,
        "a lens must concentrate more than a flat mask"
    );
}
