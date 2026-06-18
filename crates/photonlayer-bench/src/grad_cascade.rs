//! Multi-plane diffractive cascade trained end-to-end by gradient descent
//! (a small D2NN), built on the PROVEN single-plane pieces (ADR-260 roadmap).
//!
//! The single-plane gradient trainer ([`crate::grad_train`]) reaches ~83% MNIST
//! blind-test at 16x compression by descending through ONE phase plane with the
//! validated adjoint. This module stacks `K` phase planes separated by free-space
//! propagation and trains all of them jointly — the standard diffractive deep
//! neural network (Lin/Ozcan) backprop, composed entirely from this repo's
//! already-proven operators. Nothing in the optical core is reimplemented or
//! re-trusted.
//!
//! Forward (`K` planes, image `x` as amplitude):
//! ```text
//!   u_0 = x
//!   for i in 1..=K:  v_i = u_{i-1} ⊙ e^{iθ_i};  u_i = P(v_i)
//!   I = |u_K|^2  ->  avg-pool  ->  (L2-norm)  ->  W·f+b  ->  softmax  ->  CE
//! ```
//! where `P` is the SAME [`Propagator`] the single plane uses (one shared
//! inter-plane distance; AngularSpectrum). Per-plane `θ_i` (full grid), one
//! shared linear head.
//!
//! Backward (the adjoint COMPOSES — this is exactly D2NN backprop):
//! the digital tail (CE→softmax→head→(L2-norm)→pool→`∂L/∂I`) yields a per-pixel
//! weight `w_pix = ∂L/∂I`, identical to [`crate::grad_train::sample_forward_backward`].
//! The output-plane cotangent is `ḡ_K = w_pix ⊙ u_K` (from `L = Σ w·|u_K|^2`,
//! the same `ḡ_y = w⊙y` the single plane uses). Then, for `i = K down to 1`:
//! ```text
//!   ē_i      = P^H(ḡ_i)                          (proven backward_into)
//!   dL/dθ_i  = 2·Im( conj(v_i) ⊙ ē_i )          (proven phase rule, per plane)
//!   ḡ_{i-1}  = ē_i ⊙ conj(e^{iθ_i}) = ē_i ⊙ e^{-iθ_i}   (adjoint of the mask multiply)
//! ```
//! For `K = 1` this reduces EXACTLY to [`photonlayer_core::propagate::phase_gradient`].
//!
//! Determinism: full-grid per-plane θ, fixed minibatch order (seeded LCG shared
//! with the single-plane trainer), fixed reduction order, no FMA/SIMD, the same
//! plain-Rust [`Adam`]. The end-to-end finite-difference grad-check across BOTH
//! planes (in this module's tests) is the de-risk gate.

use crate::grad_adam::{l2_normalize, softmax, Adam, Pooling};
use crate::grad_train::LinearHead;
use photonlayer_core::complex::Complex;
use photonlayer_core::propagate::Propagator;

/// Number of MNIST digit classes (kept local, matches [`crate::grad_train`]).
const CLASSES: usize = 10;

/// Per-plane phase planes of a cascade, plus the shared optical operator.
/// `theta[i]` is the full-grid (`w*h`) phase of plane `i+1` in the forward order.
pub struct Cascade<'a> {
    pub prop: &'a Propagator,
    /// `K` phase planes, each length `n = w*h`.
    pub theta: Vec<Vec<f32>>,
    pub n: usize,
}

impl<'a> Cascade<'a> {
    pub fn new(prop: &'a Propagator, theta: Vec<Vec<f32>>, n: usize) -> Self {
        debug_assert!(theta.iter().all(|t| t.len() == n), "plane length must equal n");
        Self { prop, theta, n }
    }

    /// Number of cascaded phase planes.
    pub fn planes(&self) -> usize {
        self.theta.len()
    }

    /// Forward the cascade, returning the per-plane masked fields `v_i`
    /// (`v[i] = u_{i-1} ⊙ e^{iθ_i}`, kept for the backward phase rule) and the
    /// final field `u_K`. `u_0` is the incident field (image amplitude).
    ///
    /// Storing every `v_i` is what lets the backward pass apply the proven
    /// `2·Im(conj(v_i)·ē_i)` rule per plane without recomputing the forward.
    pub fn forward_fields(&self, u0: &[Complex]) -> (Vec<Vec<Complex>>, Vec<Complex>) {
        let mut v_planes: Vec<Vec<Complex>> = Vec::with_capacity(self.planes());
        let mut u = u0.to_vec();
        for theta_i in &self.theta {
            // v_i = u_{i-1} ⊙ e^{iθ_i}
            let v_i: Vec<Complex> = u
                .iter()
                .zip(theta_i.iter())
                .map(|(&c, &t)| c * Complex::from_phase(t))
                .collect();
            // u_i = P(v_i)
            u = v_i.clone();
            self.prop.propagate_into(&mut u).expect("cascade forward propagate");
            v_planes.push(v_i);
        }
        (v_planes, u)
    }

    /// Sensor intensity `I = |u_K|^2` for a sample (the readout the head/NCC see).
    pub fn intensity(&self, u0: &[Complex]) -> Vec<f32> {
        let (_v, u_k) = self.forward_fields(u0);
        u_k.iter().map(|c| c.norm_sqr()).collect()
    }

    /// Backprop a sensor-plane per-pixel weight `w_pix = ∂L/∂I` to each plane's
    /// phase gradient, COMPOSING the proven adjoint. Returns `K` gradient vectors
    /// (`grad[i] = ∂L/∂θ_i`, same forward order as `self.theta`).
    ///
    /// `v_planes` and `u_k` come from [`Self::forward_fields`] on the SAME `u0`.
    pub fn phase_grads(&self, v_planes: &[Vec<Complex>], u_k: &[Complex], w_pix: &[f32]) -> Vec<Vec<f32>> {
        let k = self.planes();
        let mut grads: Vec<Vec<f32>> = vec![Vec::new(); k];

        // Output-plane cotangent: ḡ_K = ∂L/∂conj(u_K) = w_pix ⊙ u_K (w real).
        // Identical to the single plane's ḡ_y = w ⊙ y.
        let mut g: Vec<Complex> = u_k
            .iter()
            .zip(w_pix.iter())
            .map(|(&uk, &wk)| uk.scale(wk))
            .collect();

        // Walk planes K..1. At entry `g` is the cotangent ḡ_i at plane u_i.
        for i in (0..k).rev() {
            // ē_i = P^H(ḡ_i) — pull the cotangent back across one propagation
            // stage with the PROVEN adjoint (this is backward_into).
            let mut e = g.clone();
            self.prop.backward_into(&mut e).expect("cascade adjoint backward");

            // dL/dθ_i[j] = 2·Im( conj(v_i[j]) · ē_i[j] ) — the proven phase rule,
            // with v_i playing the role of single-plane u1.
            let v_i = &v_planes[i];
            let grad_i: Vec<f32> = v_i
                .iter()
                .zip(e.iter())
                .map(|(&v, &ev)| {
                    let p = v.conj() * ev;
                    2.0 * p.im
                })
                .collect();
            grads[i] = grad_i;

            // Propagate the cotangent to the previous plane:
            // ḡ_{i-1} = ē_i ⊙ conj(e^{iθ_i}) = ē_i ⊙ e^{-iθ_i}. The adjoint of
            // multiply-by-e^{iθ} is multiply-by-its-conjugate. (Skipped after the
            // last step i=0; the loop simply ends.)
            if i > 0 {
                let theta_i = &self.theta[i];
                for (gv, &t) in e.iter_mut().zip(theta_i.iter()) {
                    *gv = *gv * Complex::from_phase(-t);
                }
                g = e;
            }
        }
        grads
    }
}

/// Forward the differentiable head over the cascade's sensor readout and return
/// `(loss, w_pix = ∂L/∂I)` while accumulating the head's `∂L/∂W,∂L/∂b`. The
/// digital tail (CE→softmax→head→(L2-norm)→avg-pool→∂L/∂I) is identical to the
/// single-plane [`crate::grad_train::sample_forward_backward`]; only the optical
/// readout `I` now comes from the cascade. `raw_pool` skips the L2-norm Jacobian.
#[allow(clippy::too_many_arguments)]
pub fn head_forward_backward(
    intensity: &[f32],
    pool: &Pooling,
    head: &LinearHead,
    label: usize,
    raw_pool: bool,
    grad_w: &mut [f32],
    grad_b: &mut [f32],
) -> (f32, Vec<f32>) {
    let n = intensity.len();

    // Pool -> features f (raw_pool=false L2-normalizes, matching the NCC eval).
    let f_raw = pool.forward(intensity);
    let norm = if raw_pool {
        1.0
    } else {
        f_raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9)
    };
    let mut f = f_raw.clone();
    if !raw_pool {
        l2_normalize(&mut f);
    }

    // Head forward: logits -> softmax -> CE.
    let z = head.logits(&f);
    let p = softmax(&z);
    let loss = -(p[label].max(1e-30)).ln();

    // Head backward: dL/dz = p - onehot; accumulate dL/dW, dL/db; dL/df = W^T dz.
    let mut dz = p.clone();
    dz[label] -= 1.0;
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

    // Pool-feature backward -> dL/df_raw (identity for raw_pool; else L2 Jacobian).
    let df_raw: Vec<f32> = if raw_pool {
        df
    } else {
        let dot: f32 = df.iter().zip(f.iter()).map(|(a, b)| a * b).sum();
        df.iter()
            .zip(f.iter())
            .map(|(&g, &fk)| (g - dot * fk) / norm)
            .collect()
    };

    // Average-pool backward: scatter dL/df_raw[box]*inv onto member pixels.
    // This vector is exactly w[j] = ∂L/∂I[j].
    let mut w_pix = vec![0.0f32; n];
    for (box_i, (idx, inv)) in pool.boxes.iter().enumerate() {
        let g = df_raw[box_i] * *inv;
        for &j in idx {
            w_pix[j] += g;
        }
    }
    (loss, w_pix)
}

/// One sample's incident field + label (mirrors [`crate::grad_train::GradSample`]).
pub struct CascadeSample {
    pub u0: Vec<Complex>,
    pub label: usize,
}

/// Hyperparameters for a cascade gradient-training run.
#[derive(Clone, Copy, Debug)]
pub struct CascadeTrainConfig {
    pub planes: usize,
    pub epochs: usize,
    pub batch: usize,
    pub lr_mask: f32,
    pub lr_head: f32,
    pub sensor: usize,
    pub seed: u64,
    pub raw_pool: bool,
    pub adam_eps: f32,
}

impl Default for CascadeTrainConfig {
    fn default() -> Self {
        Self {
            planes: 2,
            epochs: 40,
            batch: 64,
            lr_mask: 0.04,
            lr_head: 0.05,
            sensor: 8,
            seed: 0x6E157,
            raw_pool: false,
            adam_eps: 1e-7,
        }
    }
}

/// Trained cascade: per-plane phases + per-epoch mean CE.
pub struct CascadeTrainOutcome {
    pub theta: Vec<Vec<f32>>,
    pub loss_curve: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

/// Compute the analytic `(w_pix, per-plane θ-grads, loss)` for one sample —
/// shared by training and the FD grad-check so they exercise the SAME path.
fn sample_grads(
    cascade: &Cascade,
    pool: &Pooling,
    head: &LinearHead,
    s: &CascadeSample,
    raw_pool: bool,
    grad_w: &mut [f32],
    grad_b: &mut [f32],
) -> (f32, Vec<Vec<f32>>) {
    let (v_planes, u_k) = cascade.forward_fields(&s.u0);
    let intensity: Vec<f32> = u_k.iter().map(|c| c.norm_sqr()).collect();
    let (loss, w_pix) =
        head_forward_backward(&intensity, pool, head, s.label, raw_pool, grad_w, grad_b);
    let theta_grads = cascade.phase_grads(&v_planes, &u_k, &w_pix);
    (loss, theta_grads)
}

/// Mean cross-entropy over a sample set (no parameter update) — clean per-epoch
/// full-set loss, independent of minibatch noise.
fn mean_ce(
    cascade: &Cascade,
    pool: &Pooling,
    head: &LinearHead,
    data: &[CascadeSample],
    raw_pool: bool,
) -> f32 {
    let mut dummy_w = vec![0.0f32; head.w.len()];
    let mut dummy_b = vec![0.0f32; head.b.len()];
    let mut total = 0.0f32;
    for s in data {
        let intensity = cascade.intensity(&s.u0);
        let (loss, _) =
            head_forward_backward(&intensity, pool, head, s.label, raw_pool, &mut dummy_w, &mut dummy_b);
        total += loss;
    }
    total / data.len().max(1) as f32
}

/// Train a `K`-plane diffractive cascade by gradient descent on the differentiable
/// CE head, through the PROVEN composed adjoint. `theta0` holds `K` initial phase
/// planes (each length `w*h`); one shared linear head is trained alongside. The
/// minibatch order, reduction order, and Adam updates are all fixed, so the run
/// is bit-reproducible for a given seed + data + budget.
pub fn train_cascade_grad(
    prop: &Propagator,
    width: usize,
    height: usize,
    data: &[CascadeSample],
    theta0: &[Vec<f32>],
    cfg: &CascadeTrainConfig,
) -> CascadeTrainOutcome {
    let n = width * height;
    let pool = Pooling::new(width, height, cfg.sensor);
    let dim = pool.dim();
    let mut theta: Vec<Vec<f32>> = theta0.to_vec();
    let k = theta.len();
    let mut head = LinearHead::zeros(dim);

    // One Adam per plane (so per-plane moments stay independent) + head.
    let mut adam_theta: Vec<Adam> =
        (0..k).map(|_| Adam::with_eps(n, cfg.lr_mask, cfg.adam_eps)).collect();
    let mut adam_w = Adam::with_eps(head.w.len(), cfg.lr_head, cfg.adam_eps);
    let mut adam_b = Adam::with_eps(head.b.len(), cfg.lr_head, cfg.adam_eps);

    let mut loss_curve = Vec::with_capacity(cfg.epochs);
    let n_samples = data.len();

    for epoch in 0..cfg.epochs {
        let order = shuffled_indices(n_samples, cfg.seed ^ (epoch as u64).wrapping_mul(0x9E37_79B9));

        let mut start = 0;
        while start < n_samples {
            let end = (start + cfg.batch).min(n_samples);
            let bsz = (end - start) as f32;

            let mut g_theta: Vec<Vec<f32>> = (0..k).map(|_| vec![0.0f32; n]).collect();
            let mut g_w = vec![0.0f32; head.w.len()];
            let mut g_b = vec![0.0f32; head.b.len()];

            // Fixed reduction order over the (shuffled-but-fixed) batch indices.
            for idx in order.iter().take(end).skip(start) {
                // Borrow theta read-only for the forward/backward by cloning into
                // a Cascade view (theta is small: K * n f32). Keeps the proven
                // operators untouched and avoids aliasing the mutable Adam state.
                let cascade = Cascade::new(prop, theta.clone(), n);
                let s = &data[*idx];
                let (_loss, plane_grads) =
                    sample_grads(&cascade, &pool, &head, s, cfg.raw_pool, &mut g_w, &mut g_b);
                for (acc, gi) in g_theta.iter_mut().zip(plane_grads.iter()) {
                    for (a, g) in acc.iter_mut().zip(gi.iter()) {
                        *a += *g;
                    }
                }
            }

            // Mean over the batch (fixed scalar divide, no FMA).
            let inv = 1.0 / bsz;
            for plane in g_theta.iter_mut() {
                plane.iter_mut().for_each(|v| *v *= inv);
            }
            g_w.iter_mut().for_each(|v| *v *= inv);
            g_b.iter_mut().for_each(|v| *v *= inv);

            for p in 0..k {
                adam_theta[p].step(&mut theta[p], &g_theta[p]);
            }
            adam_w.step(&mut head.w, &g_w);
            adam_b.step(&mut head.b, &g_b);

            start = end;
        }

        let cascade = Cascade::new(prop, theta.clone(), n);
        loss_curve.push(mean_ce(&cascade, &pool, &head, data, cfg.raw_pool));
    }

    CascadeTrainOutcome { theta, loss_curve, width, height }
}

/// Deterministic Fisher–Yates shuffle of `0..n` from a seed (LCG stream) —
/// identical to the single-plane trainer's, so seed families line up.
fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = seed | 1;
    for i in (1..n).rev() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = (s >> 33) as usize % (i + 1);
        idx.swap(i, r);
    }
    idx
}
