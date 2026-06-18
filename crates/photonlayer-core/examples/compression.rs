//! compression — the wedge: capture far fewer numbers than the input image.
//!
//! The optical core's whole point is that the sensor reads a *compressed*
//! measurement, not a picture. Here we feed a 64x64 image (4096 pixels) through
//! a phase mask and propagation, then bin the sensor down to a tiny 8x8 grid
//! (64 measurements). We print the pixel/byte reduction and render BOTH the
//! input and the sensor measurement as ASCII so you can SEE that the
//! measurement is not the image — it does not look like the scene.
//!
//! What to look for in the output:
//!   * "reduction" of ~64x in pixels and bytes (4096 -> 64), and
//!   * the ASCII sensor block looking nothing like the ASCII input block.
//!
//! Run:
//!   cargo run --release --example compression -p photonlayer-core

use photonlayer_core::prelude::*;

/// Render a row-major intensity grid as ASCII (10 brightness levels).
fn ascii(values: &[f32], w: usize, h: usize) {
    let ramp = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
    let max = values.iter().cloned().fold(0.0f32, f32::max).max(1e-9);
    for y in 0..h {
        print!("    ");
        for x in 0..w {
            let v = (values[y * w + x] / max).clamp(0.0, 1.0);
            let idx = ((v * (ramp.len() - 1) as f32).round() as usize).min(ramp.len() - 1);
            // Double each char so the block is roughly square in a terminal.
            print!("{}{}", ramp[idx], ramp[idx]);
        }
        println!();
    }
}

fn main() {
    let n = 64;
    let sensor_bin = 8; // 64x64 sensor binned 8x -> 8x8 measurement

    // A centered bright square as the "scene".
    let pixels: Vec<f32> = (0..n * n)
        .map(|i| {
            let (x, y) = (i % n, i / n);
            let inside = x >= n / 4 && x < 3 * n / 4 && y >= n / 4 && y < 3 * n / 4;
            if inside { 1.0 } else { 0.0 }
        })
        .collect();
    let img = InputImage::from_norm_f32(n, n, pixels).expect("image");

    // Use a learned-style random mask + Fraunhofer so the measurement scrambles.
    let mask = PhaseMask::random(n, n, 0xC0DE);
    let mut cfg = OpticalConfig::demo(n, n);
    cfg.propagation = PropagationMode::Fraunhofer;
    // Bin the sensor down to a tiny measurement grid.
    cfg.detector.binning = sensor_bin;

    let frame = ScalarSimulator.simulate(&img, &mask, &cfg).expect("simulate");

    let input_px = img.width * img.height;
    let sensor_px = frame.width * frame.height;
    let ratio = compression_ratio(&img, &frame);
    // Each value is an f32 = 4 bytes.
    let input_bytes = input_px * 4;
    let sensor_bytes = sensor_px * 4;
    // Cross-correlation: how much the sensor pattern looks like the input.
    let similarity = input_frame_similarity(&img, &frame);

    println!("PhotonLayer — compression");
    println!("  input image        : {}x{} = {} pixels  ({} bytes f32)", img.width, img.height, input_px, input_bytes);
    println!("  sensor measurement : {}x{} = {} pixels  ({} bytes f32)", frame.width, frame.height, sensor_px, sensor_bytes);
    println!("  compression ratio  : {:.1}x fewer sensor pixels", ratio);
    println!("  byte reduction     : {:.1}x  ({} -> {} bytes)", input_bytes as f32 / sensor_bytes as f32, input_bytes, sensor_bytes);
    println!("  |corr(input,sensor)|: {:.3}  (low => measurement is NOT the scene)", similarity.abs());

    println!("\n  INPUT scene (64x64, downsampled to 16x16 for display):");
    // Downsample input for a compact display only.
    let disp = 16;
    let mut small = vec![0.0f32; disp * disp];
    for oy in 0..disp {
        for ox in 0..disp {
            let mut acc = 0.0;
            let mut cnt = 0.0;
            for dy in 0..(n / disp) {
                for dx in 0..(n / disp) {
                    acc += img.pixels[(oy * (n / disp) + dy) * n + (ox * (n / disp) + dx)];
                    cnt += 1.0;
                }
            }
            small[oy * disp + ox] = acc / cnt;
        }
    }
    ascii(&small, disp, disp);

    println!("\n  SENSOR measurement ({}x{}) — what the system actually captures:", frame.width, frame.height);
    ascii(&frame.intensity, frame.width, frame.height);

    println!("\nThe sensor stores {} numbers, not {} — and they do not depict the scene.", sensor_px, input_px);
}
