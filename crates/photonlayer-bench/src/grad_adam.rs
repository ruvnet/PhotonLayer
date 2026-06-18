//! Deterministic Adam optimizer (plain Rust, fixed update order).
//!
//! Kept dependency-free and SIMD/FMA-free so a gradient-training run is bit-
//! reproducible for a given seed + data + budget (the determinism invariant the
//! whole crate upholds). Split out of [`crate::grad_train`] so that module stays
//! under the 500-line cap.

/// Adam state for one parameter vector. `step` mutates the vector in place.
pub struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u64,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
}

impl Adam {
    /// Zero-initialized moments for an `n`-length parameter vector at rate `lr`.
    pub fn new(n: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
            b1: 0.9,
            b2: 0.999,
            eps: 1e-8,
        }
    }

    /// In-place Adam step on `param` given `grad` (same length). Fixed index
    /// order, no FMA — identical results across runs.
    pub fn step(&mut self, param: &mut [f32], grad: &[f32]) {
        self.t += 1;
        let bc1 = 1.0 - self.b1.powi(self.t as i32);
        let bc2 = 1.0 - self.b2.powi(self.t as i32);
        for i in 0..param.len() {
            let g = grad[i];
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g;
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * g * g;
            let mhat = self.m[i] / bc1;
            let vhat = self.v[i] / bc2;
            param[i] -= self.lr * mhat / (vhat.sqrt() + self.eps);
        }
    }
}
