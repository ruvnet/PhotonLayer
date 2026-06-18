//! End-to-end gradient check for the multi-plane diffractive cascade: the HARD
//! GATE that must pass before any cascade accuracy number is trusted (ADR-260).
//!
//! These validate that the cascade's composed adjoint (per-plane
//! [`Cascade::phase_grads`], built on `Propagator::backward_into` +
//! the proven `2·Im(conj·)` phase rule) is the exact gradient of the FULL CE
//! loss across BOTH phase planes, by central finite differences. They live as an
//! integration test (public API only) to keep `grad_cascade.rs` under the
//! 500-line limit, mirroring `photonlayer-core/tests/gradient_check.rs`.

use photonlayer_bench::grad_adam::{l2_normalize, softmax, Pooling};
use photonlayer_bench::grad_cascade::{
    head_forward_backward, train_cascade_grad, Cascade, CascadeSample, CascadeTrainConfig,
};
use photonlayer_bench::grad_train::LinearHead;
use photonlayer_core::complex::Complex;
use photonlayer_core::config::{OpticalConfig, PropagationMode};
use photonlayer_core::field::OpticalField;
use photonlayer_core::propagate::Propagator;

/// Seeded LCG in `[0,1)` (matches the single-plane FD check's generator).
fn rnd(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 40) as f32) / (1u32 << 24) as f32
}

/// THE HARD GATE: end-to-end finite-difference grad-check on the K=2 cascade CE
/// loss. Perturb θ entries across BOTH planes, central-difference the real CE
/// loss of a fixed (random) head + one labeled sample, and require the analytic
/// per-plane `∂L/∂θ` (composed adjoint) to match. Validates BOTH the
/// L2-normalized (eval-matched) and raw-pool feature paths. If this fails, no
/// cascade accuracy number is trustworthy.
#[test]
fn cascade_ce_gradient_matches_finite_difference() {
    let n_side = 16usize;
    let n = n_side * n_side;
    let mut cfg = OpticalConfig::demo(n_side, n_side);
    cfg.propagation = PropagationMode::AngularSpectrum;
    cfg.propagation_mm = 3.0;
    let prop = Propagator::new(n_side, n_side, &cfg).unwrap();
    let pool = Pooling::new(n_side, n_side, 4);
    let dim = pool.dim();

    let mut st = 0xC0FF_EE12_3456_789Au64;
    let u0: Vec<Complex> = (0..n)
        .map(|_| {
            let a = rnd(&mut st);
            let ph = (rnd(&mut st) - 0.5) * 0.5;
            Complex::from_phase(ph).scale(a)
        })
        .collect();
    // Two distinct phase planes.
    let theta0: Vec<f32> = (0..n).map(|_| rnd(&mut st) * core::f32::consts::TAU).collect();
    let theta1: Vec<f32> = (0..n).map(|_| rnd(&mut st) * core::f32::consts::TAU).collect();
    let planes = vec![theta0, theta1];

    let mut head = LinearHead::zeros(dim);
    head.w.iter_mut().for_each(|w| *w = (rnd(&mut st) - 0.5) * 0.8);
    head.b.iter_mut().for_each(|b| *b = (rnd(&mut st) - 0.5) * 0.2);
    let label = 1usize;

    for raw_pool in [false, true] {
        let cascade = Cascade::new(&prop, planes.clone(), n);
        let (v_planes, u_k) = cascade.forward_fields(&u0);
        let intensity: Vec<f32> = u_k.iter().map(|c| c.norm_sqr()).collect();
        let mut gw = vec![0.0f32; head.w.len()];
        let mut gb = vec![0.0f32; head.b.len()];
        let (_l0, w_pix) =
            head_forward_backward(&intensity, &pool, &head, label, raw_pool, &mut gw, &mut gb);
        let analytic = cascade.phase_grads(&v_planes, &u_k, &w_pix);

        // Pure CE loss as a function of the two θ planes (f64 reduction).
        let ce_loss = |p0: &[f32], p1: &[f32]| -> f64 {
            let c = Cascade::new(&prop, vec![p0.to_vec(), p1.to_vec()], n);
            let intensity = c.intensity(&u0);
            let mut f = pool.forward(&intensity);
            if !raw_pool {
                l2_normalize(&mut f);
            }
            let z = head.logits(&f);
            let p = softmax(&z);
            -(p[label].max(1e-30) as f64).ln()
        };

        // Central-difference loss with entry `kk` of plane `pl` perturbed.
        let fd_at = |pl: usize, kk: usize, eps: f64| -> f32 {
            let mut tp = planes.clone();
            let mut tm = planes.clone();
            tp[pl][kk] += eps as f32;
            tm[pl][kk] -= eps as f32;
            ((ce_loss(&tp[0], &tp[1]) - ce_loss(&tm[0], &tm[1])) / (2.0 * eps)) as f32
        };

        let eps = 1e-2f64;
        // Aggregate relative-L2 across BOTH planes, plus per-entry checks on the
        // dominant entries (matching the single-plane check's protocol).
        let gmax = analytic
            .iter()
            .flat_map(|g| g.iter())
            .fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(gmax > 1e-5, "raw_pool={raw_pool}: cascade gradient degenerate (all ~0)");
        let floor = 0.05 * gmax;

        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for plane in 0..2 {
            for kk in 0..n {
                let a = analytic[plane][kk];
                let fd = fd_at(plane, kk, eps);
                num += (a - fd) as f64 * (a - fd) as f64;
                den += a as f64 * a as f64;
                if a.abs() >= floor {
                    let rel = (a - fd).abs() / a.abs().max(fd.abs());
                    assert!(
                        rel <= 5e-2,
                        "raw_pool={raw_pool} plane={plane} k={kk}: analytic={a:e} fd={fd:e} rel={rel:e}"
                    );
                }
            }
        }
        let rel_l2 = (num / den.max(1e-30)).sqrt();
        eprintln!("[fd-grad-check] K=2 cascade raw_pool={raw_pool}: rel_L2={rel_l2:e} (gate <= 1e-2)");
        assert!(
            rel_l2 <= 1e-2,
            "raw_pool={raw_pool}: cascade end-to-end CE gradient disagrees with FD: rel_L2={rel_l2:e}"
        );
    }
}

/// Wiring guard (always on, no data): a few epochs of cascade training on a tiny
/// synthetic set must strictly reduce the mean CE — proves Adam, the per-plane
/// Adam state, batching, and the composed gradient SIGN are correct.
#[test]
fn cascade_training_reduces_loss() {
    use photonlayer_bench::synthetic::make_dataset;

    let n_side = 16usize;
    let n = n_side * n_side;
    let cfg = OpticalConfig::demo(n_side, n_side);
    let prop = Propagator::new(n_side, n_side, &cfg).unwrap();
    let train = make_dataset(n_side, 8, 1);
    let data: Vec<CascadeSample> = train
        .iter()
        .map(|s| CascadeSample {
            u0: OpticalField::from_image(&s.image, n_side, n_side).unwrap().data,
            label: s.label,
        })
        .collect();
    let theta0 = vec![vec![0.0f32; n], vec![0.0f32; n]];
    let cc = CascadeTrainConfig {
        planes: 2,
        epochs: 6,
        batch: 16,
        lr_mask: 0.05,
        lr_head: 0.1,
        sensor: 4,
        seed: 7,
        raw_pool: true,
        adam_eps: 1e-7,
    };
    let out = train_cascade_grad(&prop, n_side, n_side, &data, &theta0, &cc);
    let first = out.loss_curve.first().copied().unwrap();
    let last = out.loss_curve.last().copied().unwrap();
    assert!(last < first, "cascade loss did not decrease: {first} -> {last}");
}
