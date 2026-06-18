//! propagation_modes — real diffraction physics under three models.
//!
//! A single bright pixel (a point source) is propagated under each of the three
//! scalar-diffraction models the core supports: Fresnel (near-field), Fraunhofer
//! (far-field), and AngularSpectrum (full-regime). We measure how the point's
//! energy spreads: how many sensor pixels light up, how concentrated the peak
//! stays, and the radius that contains half the energy. These are genuine
//! diffraction observables computed from the field — not hard-coded numbers.
//!
//! What to look for in the output:
//!   * Fraunhofer of a point => essentially UNIFORM magnitude (a flat spectrum:
//!     the Fourier transform of a delta), so it "spreads" to the whole grid.
//!   * Fresnel / AngularSpectrum => the point diffracts into a localized but
//!     spreading pattern; the spread grows with propagation distance.
//!
//! Run:
//!   cargo run --release --example propagation_modes -p photonlayer-core

use photonlayer_core::prelude::*;

/// A centered single-pixel point source on an n x n grid.
fn point_field(n: usize) -> OpticalField {
    let mut px = vec![0.0f32; n * n];
    px[(n / 2) * n + n / 2] = 1.0;
    let img = InputImage::from_norm_f32(n, n, px).expect("point image");
    OpticalField::from_image(&img, n, n).expect("point field")
}

/// Diffraction observables of a propagated field's intensity.
struct Spread {
    lit_pixels: usize, // pixels above a small fraction of the peak
    peak_fraction: f32, // peak intensity / total intensity (concentration)
    energy_radius: f32, // radius (px) containing 50% of total energy
}

fn measure(field: &OpticalField) -> Spread {
    let n = field.width;
    let intensity: Vec<f32> = field.data.iter().map(|c| c.norm_sqr()).collect();
    let total: f32 = intensity.iter().sum::<f32>().max(1e-12);
    let peak = intensity.iter().cloned().fold(0.0f32, f32::max);
    let thresh = peak * 0.05;
    let lit = intensity.iter().filter(|&&v| v > thresh).count();

    // Radius containing 50% of energy, measured from the grid center.
    let (cx, cy) = (n as f32 / 2.0, field.height as f32 / 2.0);
    let max_r = (cx * cx + cy * cy).sqrt().ceil() as usize + 1;
    let mut ring = vec![0.0f32; max_r + 1];
    for y in 0..field.height {
        for x in 0..n {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let r = (dx * dx + dy * dy).sqrt().round() as usize;
            ring[r.min(max_r)] += intensity[y * n + x];
        }
    }
    let mut cum = 0.0f32;
    let mut energy_radius = max_r as f32;
    for (r, &e) in ring.iter().enumerate() {
        cum += e;
        if cum >= 0.5 * total {
            energy_radius = r as f32;
            break;
        }
    }

    Spread {
        lit_pixels: lit,
        peak_fraction: peak / total,
        energy_radius,
    }
}

fn run(mode: PropagationMode, distance_mm: f32, n: usize) {
    let f = point_field(n);
    let mut cfg = OpticalConfig::demo(n, n);
    cfg.propagation = mode;
    cfg.propagation_mm = distance_mm;
    let out = propagate(&f, &cfg).expect("propagate");
    let s = measure(&out);
    println!(
        "  {:<16} z={:>5.1}mm  lit_px={:>5}/{:<5}  peak/total={:>7.4}  E50_radius={:>5.1}px  power={:.3}",
        format!("{:?}", mode),
        distance_mm,
        s.lit_pixels,
        n * n,
        s.peak_fraction,
        s.energy_radius,
        out.power()
    );
}

fn main() {
    let n = 64;
    println!("PhotonLayer — propagation_modes  (point source on {n}x{n} grid)");
    println!("  A single bright pixel diffracts differently under each model.\n");

    println!("  Far-field (Fraunhofer) — FT of a delta is a flat spectrum:");
    run(PropagationMode::Fraunhofer, 10.0, n);

    println!("\n  Near-field (Fresnel) — spread grows with distance:");
    run(PropagationMode::Fresnel, 2.0, n);
    run(PropagationMode::Fresnel, 8.0, n);
    run(PropagationMode::Fresnel, 20.0, n);

    println!("\n  Full-regime (AngularSpectrum) — power-conserving transfer fn:");
    run(PropagationMode::AngularSpectrum, 2.0, n);
    run(PropagationMode::AngularSpectrum, 8.0, n);
    run(PropagationMode::AngularSpectrum, 20.0, n);

    println!("\nFraunhofer lights the whole grid (uniform); near-field spreads with z.");
}
