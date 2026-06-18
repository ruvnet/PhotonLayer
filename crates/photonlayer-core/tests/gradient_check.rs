//! Gradient-based optical training: adjoint + finite-difference gradient check.
//!
//! These tests are the keystone proof for differentiable phase-mask training
//! (ADR-260): they validate that [`Propagator::backward_into`] is the exact
//! adjoint of [`Propagator::propagate_into`], and that the analytic phase
//! gradient ([`phase_gradient`]) matches a central finite difference of the
//! intensity loss. They live as an integration test (only public API) to keep
//! `propagate.rs` under the 500-line limit while still running under
//! `cargo test -p photonlayer-core`.

use core::f32::consts::PI;
use photonlayer_core::prelude::*;

/// A small deterministic LCG, used only here to seed a smooth random
/// field/mask/weights without pulling the crate RNG into the gradient code.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Top 24 bits -> [0, 1).
    ((*state >> 40) as f32) / (1u32 << 24) as f32
}

/// Build a deterministic Transfer-arm propagator plus a seeded
/// (u0, theta, w) sample on an `n x n` grid.
fn grad_sample(
    n: usize,
    mode: PropagationMode,
) -> (Propagator, Vec<Complex>, Vec<f32>, Vec<f32>) {
    let mut cfg = OpticalConfig::demo(n, n);
    cfg.propagation = mode;
    cfg.propagation_mm = 3.0;
    let prop = Propagator::new(n, n, &cfg).unwrap();

    let mut s = 0x1234_5678_9abc_def0u64;
    // Smooth-ish complex incident field (amplitude in [0,1], small phase).
    let u0: Vec<Complex> = (0..n * n)
        .map(|_| {
            let a = lcg(&mut s);
            let ph = (lcg(&mut s) - 0.5) * 0.6;
            Complex::from_phase(ph).scale(a)
        })
        .collect();
    // Trainable phase, spread across [0, 2π).
    let theta: Vec<f32> = (0..n * n).map(|_| lcg(&mut s) * 2.0 * PI).collect();
    // Fixed real sensor weights (a target-like vector); some negative so the
    // loss is a genuine signed linear-in-intensity objective, not just power.
    let w: Vec<f32> = (0..n * n).map(|_| lcg(&mut s) - 0.5).collect();
    (prop, u0, theta, w)
}

/// Adjoint identity `⟨P(a), b⟩ == ⟨a, P^H(b)⟩` (Hermitian inner product),
/// the precision-robust proof that `backward_into` is the true adjoint of
/// `propagate_into` independent of any finite-difference noise. Both sides
/// are single complex dot products, so f32 cancellation is minimal.
#[test]
fn backward_is_exact_adjoint_of_forward() {
    fn hdot(x: &[Complex], y: &[Complex]) -> Complex {
        // ⟨x, y⟩ = Σ conj(x)·y.
        x.iter()
            .zip(y.iter())
            .fold(Complex::ZERO, |acc, (&xi, &yi)| acc + xi.conj() * yi)
    }
    for &mode in &[PropagationMode::Fresnel, PropagationMode::AngularSpectrum] {
        let n = 16;
        let (prop, a, theta, _w) = grad_sample(n, mode);
        // Use theta to manufacture a second independent vector b.
        let b: Vec<Complex> = theta.iter().map(|&t| Complex::from_phase(t)).collect();

        let mut pa = a.clone();
        prop.propagate_into(&mut pa).unwrap();
        let mut phb = b.clone();
        prop.backward_into(&mut phb).unwrap();

        let lhs = hdot(&pa, &b); // ⟨P a, b⟩
        let rhs = hdot(&a, &phb); // ⟨a, P^H b⟩
        let err = (lhs - rhs).abs() / lhs.abs().max(1.0);
        assert!(
            err < 1e-4,
            "{mode:?} adjoint identity broken: <Pa,b>={lhs:?} <a,P^Hb>={rhs:?} rel={err:e}"
        );
    }
}

/// THE PROOF: the analytic phase gradient must match a central finite
/// difference of the loss to tight tolerance. This validates both the
/// adjoint `backward_into` and the exact constant/sign of the
/// `2·Im(conj(u1)·gback)` rule. Fast, always-on, deterministic.
#[test]
fn phase_gradient_matches_finite_difference() {
    for &mode in &[PropagationMode::Fresnel, PropagationMode::AngularSpectrum] {
        let n = 16;
        let (prop, u0, theta, w) = grad_sample(n, mode);

        let analytic = phase_gradient(&prop, &u0, &theta, &w).unwrap();

        // f64-accumulated loss: the per-pixel intensity term stays f32 (same
        // function the analytic gradient differentiates), but the reduction is
        // done in f64 so the central difference of two near-equal losses is not
        // swamped by f32 summation noise. This makes the FD a *fair* reference
        // for a tight tolerance rather than artificially loosening the bound.
        let loss_f64 = |th: &[f32]| -> f64 {
            let mut y: Vec<Complex> = u0
                .iter()
                .zip(th.iter())
                .map(|(&c, &t)| c * Complex::from_phase(t))
                .collect();
            prop.propagate_into(&mut y).unwrap();
            y.iter()
                .zip(w.iter())
                .map(|(&yk, &wk)| (wk * yk.norm_sqr()) as f64)
                .sum()
        };

        // The simulator runs in f32, so finite-differencing it against itself
        // has a real noise floor. `eps = 1e-2` balances the two FD error
        // sources for an f32 pipeline: a smaller eps amplifies f32 *cancellation*
        // noise (subtracting two near-equal losses), a larger eps grows the
        // O(eps²)·curvature *truncation* error. Even at the optimum, a single-
        // cell central difference of the *smallest* gradient entries can be
        // 10-30% off purely from the f32 floor (those entries contribute
        // negligibly to the L2 norm). We therefore prove correctness two ways:
        //   (a) aggregate relative-L2 over the whole gradient vector — the
        //       meaningful metric, since zero-mean per-entry noise averages out
        //       (empirically ~1.3e-4 here); and
        //   (b) per-entry, but only on entries large enough to sit above the
        //       f32 FD floor (magnitude >= 5% of the max |grad|).
        let eps = 1e-2f64;
        let n2 = n * n;
        let mut num = 0.0f64; // Σ (a - fd)²
        let mut den = 0.0f64; // Σ a²
        let gmax = analytic.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let entry_floor = 0.05 * gmax; // ignore tiny entries dominated by FD noise
        for k in 0..n2 {
            let mut tp = theta.clone();
            tp[k] += eps as f32;
            let mut tm = theta.clone();
            tm[k] -= eps as f32;
            let fd = ((loss_f64(&tp) - loss_f64(&tm)) / (2.0 * eps)) as f32;
            let a = analytic[k];

            num += (a - fd) as f64 * (a - fd) as f64;
            den += a as f64 * a as f64;

            if a.abs() >= entry_floor {
                let rel = (a - fd).abs() / a.abs().max(fd.abs());
                assert!(
                    rel <= 1e-2,
                    "{mode:?} grad mismatch at {k}: analytic={a:e} fd={fd:e} rel={rel:e}"
                );
            }
        }
        // Aggregate relative-L2 error: the primary proof of the constant+sign.
        let rel_l2 = (num / den.max(1e-30)).sqrt();
        assert!(
            rel_l2 <= 1e-2,
            "{mode:?} analytic gradient disagrees with FD: rel_L2={rel_l2:e}"
        );
        // Sanity: the gradient is not trivially zero everywhere.
        assert!(gmax > 1e-4, "{mode:?} gradient is degenerate (all ~0)");
    }
}

/// Smoke test: a single gradient-descent step on θ strictly decreases L,
/// proving the gradient *sign* is usable for training (not just numerically
/// correct in magnitude). Deterministic.
#[test]
fn one_descent_step_decreases_loss() {
    let n = 16;
    let (prop, u0, theta, w) = grad_sample(n, PropagationMode::Fresnel);
    let l0 = intensity_loss(&prop, &u0, &theta, &w).unwrap();
    let grad = phase_gradient(&prop, &u0, &theta, &w).unwrap();

    // Step down the gradient. lr chosen small enough that the linearization
    // holds for this smooth loss.
    let gnorm2: f32 = grad.iter().map(|g| g * g).sum();
    assert!(gnorm2 > 0.0, "gradient vanished");
    let lr = 1e-2f32;
    let theta1: Vec<f32> = theta
        .iter()
        .zip(grad.iter())
        .map(|(&t, &g)| t - lr * g)
        .collect();
    let l1 = intensity_loss(&prop, &u0, &theta1, &w).unwrap();
    assert!(l1 < l0, "descent did not reduce loss: {l0} -> {l1}");
}

/// The Fraunhofer adjoint is documented but not implemented; it must fail
/// loudly rather than silently returning the forward (or wrong) result.
#[test]
fn backward_into_fraunhofer_is_explicit_error() {
    let mut cfg = OpticalConfig::demo(8, 8);
    cfg.propagation = PropagationMode::Fraunhofer;
    let prop = Propagator::new(8, 8, &cfg).unwrap();
    let mut data = vec![Complex::ONE; 64];
    assert!(prop.backward_into(&mut data).is_err());
}
