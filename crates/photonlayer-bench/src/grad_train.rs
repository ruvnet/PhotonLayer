//! Gradient-based phase-mask training (the ceiling-break path, ADR-260 roadmap).
//!
//! The hill-climb learner ([`crate::learn`]) converges to a single-layer
//! optimizer ceiling (~73% MNIST blind-test at 16x compression). This module
//! replaces the block hill-climb with **analytic gradient descent through the
//! diffraction operator**, using the PROVEN adjoint in `photonlayer-core`
//! ([`Propagator::backward_into`] + [`phase_gradient`], validated by
//! `tests/gradient_check.rs`).
//!
//! Eval keeps the deterministic nearest-centroid (NCC) decoder so the published
//! metric stays apples-to-apples with the hill-climb baseline. Training uses a
//! small differentiable head over the SAME pooled sensor readout the NCC sees:
//! `image -> u0 -> mask e^{iθ} -> P(.) -> |.|^2 -> avg-pool -> L2-norm = f ->
//! logits W·f+b -> softmax -> CE`.
//!
//! The whole optical half of the chain (`|.|^2`, propagation adjoint, the
//! `2·Im(conj(u1)·gback)` phase rule) is exactly what [`phase_gradient`]
//! computes for the linear-in-intensity loss `L = Σ_j w[j]·I[j]`. We only have
//! to produce the per-pixel intensity weight `w[j] = ∂L/∂I[j]` by backpropagating
//! CE → softmax → linear head → (L2-norm) → average-pool, then one call to
//! `phase_gradient(prop, u0, θ, w)` yields `∂L/∂θ` through the validated adjoint.
//! Nothing in the optical core is reimplemented or re-trusted here.
//!
//! Determinism: fixed minibatch order, fixed reduction order, no FMA/SIMD, a
//! seeded plain-Rust [`Adam`](crate::grad_adam::Adam). The end-to-end FD
//! gradient check (below) is the de-risk gate — it validates both the
//! L2-normalized and raw-pool feature paths.

use crate::grad_adam::{l2_normalize, softmax, Adam, Pooling};
use crate::synthetic::Sample;
use photonlayer_core::complex::Complex;
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::field::OpticalField;
use photonlayer_core::propagate::{phase_gradient, Propagator};

/// Number of MNIST digit classes (kept local to avoid a cross-module dep).
const CLASSES: usize = 10;

/// Linear softmax head over the pooled sensor readout. Plain f32, fixed index
/// order; the only trainable digital params (eval discards it for the NCC).
pub struct LinearHead {
    /// Row-major `CLASSES × dim`.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub dim: usize,
}

impl LinearHead {
    pub fn zeros(dim: usize) -> Self {
        Self {
            w: vec![0.0; CLASSES * dim],
            b: vec![0.0; CLASSES],
            dim,
        }
    }

    /// Logits `z = W·f + b` (length CLASSES), fixed accumulation order.
    pub fn logits(&self, f: &[f32]) -> Vec<f32> {
        let mut z = self.b.clone();
        for c in 0..CLASSES {
            let row = &self.w[c * self.dim..(c + 1) * self.dim];
            let mut acc = 0.0f32;
            for (wv, fv) in row.iter().zip(f) {
                acc += *wv * *fv;
            }
            z[c] += acc;
        }
        z
    }
}

/// One sample's precomputed incident field `u0` (image -> centered amplitude),
/// reused every epoch (it never changes — only θ trains).
pub struct GradSample {
    pub u0: Vec<Complex>,
    pub label: usize,
}

/// Build the per-sample incident fields once. `cfg.width/height` is the grid.
pub fn build_grad_samples(samples: &[Sample], cfg: &OpticalConfig) -> Vec<GradSample> {
    samples
        .iter()
        .map(|s| {
            let field = OpticalField::from_image(&s.image, cfg.width, cfg.height)
                .expect("image fits grid");
            GradSample {
                u0: field.data,
                label: s.label,
            }
        })
        .collect()
}

/// Forward the differentiable head for one sample and return `(loss, dL/dI per
/// pixel)` — the per-pixel weight `w[j]` that [`phase_gradient`] consumes — also
/// accumulating the head's own `∂L/∂W,∂L/∂b`. Backprops the DIGITAL tail
/// (CE→softmax→head→(L2-norm)→avg-pool→∂L/∂I); the OPTICAL half (`∂I/∂θ`) is the
/// proven `phase_gradient`'s job. `raw_pool` skips the L2-norm Jacobian.
#[allow(clippy::too_many_arguments)]
fn sample_forward_backward(
    prop: &Propagator,
    pool: &Pooling,
    head: &LinearHead,
    u0: &[Complex],
    theta: &[f32],
    label: usize,
    raw_pool: bool,
    grad_w: &mut [f32],
    grad_b: &mut [f32],
) -> (f32, Vec<f32>) {
    let n = u0.len();
    // --- Optical forward: I = |P(u0 ⊙ e^{iθ})|^2 (same path NCC eval uses). ---
    let mut y: Vec<Complex> = u0
        .iter()
        .zip(theta.iter())
        .map(|(&c, &t)| c * Complex::from_phase(t))
        .collect();
    prop.propagate_into(&mut y).expect("forward propagate");
    let intensity: Vec<f32> = y.iter().map(|c| c.norm_sqr()).collect();

    // --- Pool -> features f. raw_pool=false L2-normalizes (matches the NCC eval
    // feature exactly); raw_pool=true feeds raw pooled intensities. ---
    let f_raw = pool.forward(&intensity);
    let norm = if raw_pool {
        1.0 // no normalization; f == f_raw, Jacobian is identity
    } else {
        f_raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9)
    };
    let mut f = f_raw.clone();
    if !raw_pool {
        l2_normalize(&mut f);
    }

    // --- Head forward: logits -> softmax -> CE. ---
    let z = head.logits(&f);
    let p = softmax(&z);
    let loss = -(p[label].max(1e-30)).ln();

    // --- Head backward: dL/dz = p - onehot. ---
    let mut dz = p.clone();
    dz[label] -= 1.0;
    // dL/dW, dL/db (accumulate), and dL/df = W^T dz.
    let mut df = vec![0.0f32; pool.dim()];
    for c in 0..CLASSES {
        let g = dz[c];
        grad_b[c] += g;
        let row = c * pool.dim();
        for d in 0..pool.dim() {
            grad_w[row + d] += g * f[d];
            df[d] += head.w[row + d] * g;
        }
    }

    // --- Pool-feature backward -> dL/df_raw.
    // raw_pool: identity (df_raw = df). Else L2-norm Jacobian:
    // dL/df_raw = (df - (df·f) f) / norm, with f the normalized vector. ---
    let df_raw: Vec<f32> = if raw_pool {
        df
    } else {
        let dot: f32 = df.iter().zip(f.iter()).map(|(a, b)| a * b).sum();
        df.iter()
            .zip(f.iter())
            .map(|(&g, &fk)| (g - dot * fk) / norm)
            .collect()
    };

    // --- Average-pool backward: scatter dL/df_raw[p]*inv onto member pixels.
    // This vector is exactly w[j] = ∂L/∂I[j]. ---
    let mut w_pix = vec![0.0f32; n];
    for (box_i, (idx, inv)) in pool.boxes.iter().enumerate() {
        let g = df_raw[box_i] * *inv;
        for &j in idx {
            w_pix[j] += g;
        }
    }
    (loss, w_pix)
}

/// Hyperparameters for one gradient-training run.
#[derive(Clone, Copy, Debug)]
pub struct GradTrainConfig {
    pub epochs: usize,
    pub batch: usize,
    /// Adam LR for the phase mask.
    pub lr_mask: f32,
    /// Adam LR for the linear head.
    pub lr_head: f32,
    pub sensor: usize,
    pub seed: u64,
    /// Train the head on the RAW average-pooled readout (no L2-normalize).
    /// L2-norm injects a `1/‖f_raw‖` coupling that ties every pixel gradient
    /// together and discards the absolute-intensity signal the optics encodes;
    /// `true` trains on raw pooled intensities. EVAL always L2-normalizes via
    /// the NCC decoder, so the published metric stays apples-to-apples either
    /// way — only the TRAIN-time feature differs.
    pub raw_pool: bool,
    /// Adam epsilon. `1e-7` avoids `v_hat` underflow that can freeze the many
    /// near-zero-gradient dark mask cells; `1e-8` is the textbook default.
    pub adam_eps: f32,
}

impl Default for GradTrainConfig {
    fn default() -> Self {
        Self {
            epochs: 30,
            batch: 64,
            lr_mask: 0.05,
            lr_head: 0.05,
            sensor: 8,
            seed: 0x6D_ADA_11,
            raw_pool: false,
            adam_eps: 1e-8,
        }
    }
}

/// Outcome of gradient training: the trained mask phases + per-epoch mean CE.
pub struct GradTrainOutcome {
    pub theta: Vec<f32>,
    pub loss_curve: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

/// Mean cross-entropy of the head over a sample set (no grad) — used to log a
/// clean full-set loss per epoch independent of minibatch noise.
fn mean_ce(
    prop: &Propagator,
    pool: &Pooling,
    head: &LinearHead,
    theta: &[f32],
    data: &[GradSample],
    raw_pool: bool,
) -> f32 {
    // Throwaway grad buffers — we only read the returned loss here.
    let mut dummy_w = vec![0.0f32; head.w.len()];
    let mut dummy_b = vec![0.0f32; head.b.len()];
    let mut total = 0.0f32;
    for s in data {
        let (loss, _) = sample_forward_backward(
            prop, pool, head, &s.u0, theta, s.label, raw_pool, &mut dummy_w, &mut dummy_b,
        );
        total += loss;
    }
    total / data.len().max(1) as f32
}

/// Train a phase mask by gradient descent on the differentiable CE head.
///
/// `theta0` is the (deterministic) initial phase, length `w*h`. The minibatch
/// order is a fixed seeded permutation regenerated each epoch; gradients are
/// summed in fixed sample order; Adam updates are fixed-order — so the whole run
/// is bit-reproducible for a given seed + data.
pub fn train_mask_grad(
    prop: &Propagator,
    width: usize,
    height: usize,
    data: &[GradSample],
    theta0: &[f32],
    cfg: &GradTrainConfig,
) -> GradTrainOutcome {
    let pool = Pooling::new(width, height, cfg.sensor);
    let dim = pool.dim();
    let mut theta = theta0.to_vec();
    let mut head = LinearHead::zeros(dim);

    let mut adam_theta = Adam::with_eps(theta.len(), cfg.lr_mask, cfg.adam_eps);
    let mut adam_w = Adam::with_eps(head.w.len(), cfg.lr_head, cfg.adam_eps);
    let mut adam_b = Adam::with_eps(head.b.len(), cfg.lr_head, cfg.adam_eps);

    let mut loss_curve = Vec::with_capacity(cfg.epochs);
    let n = data.len();

    for epoch in 0..cfg.epochs {
        // Deterministic per-epoch shuffle (seeded LCG Fisher–Yates).
        let order = shuffled_indices(n, cfg.seed ^ (epoch as u64).wrapping_mul(0x9E37_79B9));

        let mut start = 0;
        while start < n {
            let end = (start + cfg.batch).min(n);
            let bsz = (end - start) as f32;

            let mut g_theta = vec![0.0f32; theta.len()];
            let mut g_w = vec![0.0f32; head.w.len()];
            let mut g_b = vec![0.0f32; head.b.len()];

            // Fixed reduction order over the (shuffled-but-fixed) batch indices.
            for k in start..end {
                let s = &data[order[k]];
                let (_loss, w_pix) = sample_forward_backward(
                    prop, &pool, &head, &s.u0, &theta, s.label, cfg.raw_pool, &mut g_w, &mut g_b,
                );
                // Optical half via the PROVEN adjoint: dL/dθ for this sample.
                let g = phase_gradient(prop, &s.u0, &theta, &w_pix)
                    .expect("phase gradient");
                for (acc, gi) in g_theta.iter_mut().zip(g.iter()) {
                    *acc += *gi;
                }
            }

            // Mean over the batch (fixed scalar divide, no FMA).
            let inv = 1.0 / bsz;
            let scale = |g: &mut [f32]| g.iter_mut().for_each(|v| *v *= inv);
            scale(&mut g_theta);
            scale(&mut g_w);
            scale(&mut g_b);

            adam_theta.step(&mut theta, &g_theta);
            adam_w.step(&mut head.w, &g_w);
            adam_b.step(&mut head.b, &g_b);

            start = end;
        }

        loss_curve.push(mean_ce(prop, &pool, &head, &theta, data, cfg.raw_pool));
    }

    GradTrainOutcome { theta, loss_curve, width, height }
}

/// Deterministic Fisher–Yates shuffle of `0..n` from a seed (LCG stream).
fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = seed | 1; // avoid the all-zero fixed point
    for i in (1..n).rev() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = (s >> 33) as usize % (i + 1);
        idx.swap(i, r);
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use photonlayer_core::config::PropagationMode;

    /// END-TO-END finite-difference gradient check on the FULL CE loss: the
    /// de-risk gate. Perturb a handful of mask phases, central-difference the
    /// real CE loss of a fixed (random) head + one labeled sample, and require
    /// the analytic `∂L/∂θ` (head backward + proven phase_gradient) to match.
    ///
    /// The head is FIXED here (we are checking the θ-gradient path the optical
    /// adjoint feeds, not the head's own W/b grads — those are textbook softmax).
    #[test]
    fn end_to_end_ce_gradient_matches_finite_difference() {
        let n = 16usize;
        let mut cfg = OpticalConfig::demo(n, n);
        cfg.propagation = PropagationMode::AngularSpectrum;
        let prop = Propagator::new(n, n, &cfg).unwrap();
        let pool = Pooling::new(n, n, 4); // 4x4 sensor on a 16x16 grid
        let dim = pool.dim();

        // Deterministic incident field, phase, head (seeded LCG).
        let mut st = 0xC0FFEE_1234_5678u64;
        let rnd = |s: &mut u64| -> f32 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 40) as f32) / (1u32 << 24) as f32
        };
        let u0: Vec<Complex> = (0..n * n)
            .map(|_| {
                let a = rnd(&mut st);
                let ph = (rnd(&mut st) - 0.5) * 0.5;
                Complex::from_phase(ph).scale(a)
            })
            .collect();
        let theta: Vec<f32> = (0..n * n).map(|_| rnd(&mut st) * 6.283).collect();
        let mut head = LinearHead::zeros(dim);
        head.w.iter_mut().for_each(|w| *w = (rnd(&mut st) - 0.5) * 0.8);
        head.b.iter_mut().for_each(|b| *b = (rnd(&mut st) - 0.5) * 0.2);
        let label = 1usize;

        // Validate BOTH feature paths: L2-normalized (eval-matched) and raw pool.
        for raw_pool in [false, true] {
            let mut gw = vec![0.0f32; head.w.len()];
            let mut gb = vec![0.0f32; head.b.len()];
            let (_l0, w_pix) = sample_forward_backward(
                &prop, &pool, &head, &u0, &theta, label, raw_pool, &mut gw, &mut gb,
            );
            let analytic = phase_gradient(&prop, &u0, &theta, &w_pix).unwrap();

            // Pure CE loss as a function of θ (f64 reduction for a fair FD floor).
            let ce_loss = |th: &[f32]| -> f64 {
                let mut y: Vec<Complex> = u0
                    .iter()
                    .zip(th.iter())
                    .map(|(&c, &t)| c * Complex::from_phase(t))
                    .collect();
                prop.propagate_into(&mut y).unwrap();
                let intensity: Vec<f32> = y.iter().map(|c| c.norm_sqr()).collect();
                let mut f = pool.forward(&intensity);
                if !raw_pool {
                    l2_normalize(&mut f);
                }
                let z = head.logits(&f);
                let p = softmax(&z);
                -(p[label].max(1e-30) as f64).ln()
            };

            // Central difference; skip tiny entries dominated by the f32 FD
            // floor, exactly as the core's proven gradient check does.
            let eps = 1e-2f64;
            let gmax = analytic.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(gmax > 1e-5, "raw_pool={raw_pool}: gradient degenerate (all ~0)");
            let floor = 0.05 * gmax;

            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for k in 0..n * n {
                let a = analytic[k];
                let mut tp = theta.clone();
                tp[k] += eps as f32;
                let mut tm = theta.clone();
                tm[k] -= eps as f32;
                let fd = ((ce_loss(&tp) - ce_loss(&tm)) / (2.0 * eps)) as f32;
                num += (a - fd) as f64 * (a - fd) as f64;
                den += a as f64 * a as f64;
                if a.abs() >= floor {
                    let rel = (a - fd).abs() / a.abs().max(fd.abs());
                    assert!(
                        rel <= 5e-2,
                        "raw_pool={raw_pool} CE grad mismatch at {k}: analytic={a:e} fd={fd:e} rel={rel:e}"
                    );
                }
            }
            let rel_l2 = (num / den.max(1e-30)).sqrt();
            assert!(
                rel_l2 <= 1e-2,
                "raw_pool={raw_pool}: end-to-end CE gradient disagrees with FD: rel_L2={rel_l2:e}"
            );
        }
    }
}
