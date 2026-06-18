//! Deterministic numeric primitives for gradient training: Adam, average
//! pooling, L2-normalize, softmax.
//!
//! Kept dependency-free and SIMD/FMA-free so a gradient-training run is bit-
//! reproducible for a given seed + data + budget (the determinism invariant the
//! whole crate upholds). Split out of [`crate::grad_train`] so that module stays
//! under the 500-line cap.

/// Average-pool a row-major `w×h` grid to `s×s`, replicating
/// [`crate::decoder::pool_features`]'s exact box boundaries. Stores each output
/// cell's source indices + `1/count` weight so the backward pass can scatter
/// `∂L/∂f_raw` onto pixels.
pub struct Pooling {
    /// `s*s` boxes; each box is `(src_indices, inv_count)`.
    pub boxes: Vec<(Vec<usize>, f32)>,
    sensor: usize,
}

impl Pooling {
    pub fn new(w: usize, h: usize, sensor: usize) -> Self {
        let mut boxes = Vec::with_capacity(sensor * sensor);
        for oy in 0..sensor {
            for ox in 0..sensor {
                let x0 = ox * w / sensor;
                let x1 = ((ox + 1) * w / sensor).max(x0 + 1).min(w);
                let y0 = oy * h / sensor;
                let y1 = ((oy + 1) * h / sensor).max(y0 + 1).min(h);
                let mut idx = Vec::new();
                for y in y0..y1 {
                    for x in x0..x1 {
                        idx.push(y * w + x);
                    }
                }
                let inv = if idx.is_empty() { 0.0 } else { 1.0 / idx.len() as f32 };
                boxes.push((idx, inv));
            }
        }
        Self { boxes, sensor }
    }

    pub fn dim(&self) -> usize {
        self.sensor * self.sensor
    }

    /// Forward pool: `intensity` (w*h) -> raw pooled means (s*s).
    pub fn forward(&self, intensity: &[f32]) -> Vec<f32> {
        self.boxes
            .iter()
            .map(|(idx, inv)| {
                let mut acc = 0.0f32;
                for &j in idx {
                    acc += intensity[j];
                }
                acc * *inv
            })
            .collect()
    }
}

/// L2-normalize in place, returning the pre-norm length (1.0 if degenerate) so
/// the backward pass can use it. Matches `decoder::pool_features`'s 1e-9 guard.
pub fn l2_normalize(v: &mut [f32]) -> f32 {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-9 {
        for x in v.iter_mut() {
            *x /= n;
        }
        n
    } else {
        1.0
    }
}

/// Numerically-stable softmax (subtract max), fixed order.
pub fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut e: Vec<f32> = z.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    let inv = 1.0 / s.max(1e-30);
    for v in &mut e {
        *v *= inv;
    }
    e
}

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
    /// Zero-initialized moments for an `n`-length parameter vector at rate `lr`,
    /// with the textbook `eps=1e-8`. Use [`Adam::with_eps`] to override `eps`
    /// (e.g. `1e-7` to avoid `v_hat` underflow freezing near-zero-grad cells).
    pub fn new(n: usize, lr: f32) -> Self {
        Self::with_eps(n, lr, 1e-8)
    }

    /// As [`Adam::new`] but with an explicit `eps`.
    pub fn with_eps(n: usize, lr: f32, eps: f32) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
            b1: 0.9,
            b2: 0.999,
            eps,
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
