//! mnist_compression — the real-data optical-compression run (skips if no data).
//!
//! This is the honest real-data counterpart to the synthetic demos: it loads
//! standard MNIST from a local cache dir and runs the full
//! optical-compression + differential-detection benchmark on it. The IDX files
//! are NOT downloaded by this example — if the cache is absent it prints how to
//! fetch them and EXITS CLEANLY (never panics), so the example always builds and
//! runs offline. When the data IS present, every printed number is measured from
//! the crates, not fabricated.
//!
//! What to look for in the output:
//!   * WITHOUT data: a clear "MNIST cache not found — skipping" message + the
//!     fetch command, exit 0.
//!   * WITH data: full-image baseline vs optical-compressed accuracy at 16x
//!     sensor reduction, plus the plain-vs-differential argmax lever.
//!
//! Run:
//!   cargo run --release --example mnist_compression -p photonlayer-bench
//!
//! Fetch the data first (one time) to see the full run; see the printed command.

use photonlayer_bench::mnist::{default_cache_dir, load_test, load_train, subset, MnistError};
use photonlayer_bench::mnist_bench::{run_mnist_differential, MnistBenchConfig};

fn main() {
    let dir = default_cache_dir();
    let bcfg = MnistBenchConfig {
        // Small per-class counts keep the example fast; the README headline run
        // uses 400/200 per class via the ignored integration test.
        iterations: 300,
        ..Default::default()
    };

    // Try to load; on a missing cache, skip gracefully (no panic, exit 0).
    let train_raw = match load_train(&dir) {
        Ok(r) => r,
        Err(MnistError::Missing(p)) => return skip(&p),
        Err(e) => return skip_err(e),
    };
    let test_raw = match load_test(&dir) {
        Ok(r) => r,
        Err(MnistError::Missing(p)) => return skip(&p),
        Err(e) => return skip_err(e),
    };

    // 60/class train, 30/class test: a small, fast, real-data subset.
    let train = subset(&train_raw, 60, bcfg.cell, bcfg.grid);
    let test = subset(&test_raw, 30, bcfg.cell, bcfg.grid);

    println!("PhotonLayer — mnist_compression  (REAL MNIST, grid={}x{})", bcfg.grid, bcfg.grid);
    println!("  train={} test={}  sensor={}x{} (16x reduction)\n", train.len(), test.len(), bcfg.sensor, bcfg.sensor);

    let r = run_mnist_differential(&train, &test, &bcfg);

    println!("  Config A — optical compression vs full-image baseline (same tiny decoder):");
    println!("    full-image baseline : acc={:.3}  ({} px, {} params)", r.baseline_acc, r.baseline_pixels, r.baseline_decoder_params);
    println!("    optical compressed  : acc={:.3}  ({} px, {} params)", r.optical_acc, r.optical_sensor_pixels, r.decoder_params);
    println!("    sensor reduction    : {:.1}x     MAC reduction: {:.1}x", r.sensor_reduction_x, r.mac_reduction_x);
    println!("    Δ optical-baseline  : {:+.3} pp", r.optical_acc - r.baseline_acc);

    println!("\n  Config B — differential-detection lever (no decoder, same trained mask):");
    println!("    plain argmax I+      : acc={:.3}", r.config_b_plain);
    println!("    differential I+ - I- : acc={:.3}", r.config_b_differential);
    println!("    lever Δ              : {:+.3}", r.config_b_differential - r.config_b_plain);

    println!("\n  learned-vs-random differential (optics-only floor): {:.3} vs {:.3}",
        r.optics_only_differential, r.random_optics_only_differential);
    println!("\nNote: this small/fast subset under-performs the README's 400/200-per-class run.");
}

fn skip(missing: &std::path::Path) {
    println!("PhotonLayer — mnist_compression");
    println!("  MNIST cache not found: {}", missing.display());
    println!("  Skipping the real-data run (this is expected offline).\n");
    println!("  To enable it, place the four standard IDX files in:");
    println!("    {}", default_cache_dir().display());
    println!("  Files: train-images-idx3-ubyte, train-labels-idx1-ubyte,");
    println!("         t10k-images-idx3-ubyte,  t10k-labels-idx1-ubyte");
    println!("  (download + gunzip the MNIST IDX archives into that dir).");
}

fn skip_err(e: MnistError) {
    println!("PhotonLayer — mnist_compression");
    println!("  MNIST data present but unreadable: {e}");
    println!("  Skipping the real-data run.");
}
