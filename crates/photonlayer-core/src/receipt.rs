//! RVF-style experiment receipt (ADR-260 §15).
//!
//! A receipt binds every input that determines an experiment's output to a
//! set of content hashes plus environment provenance. Replaying the same
//! experiment must reproduce `output_hash`; otherwise the run is rejected as
//! tampered or non-deterministic (the determinism invariant, §21).

use crate::config::OpticalConfig;
use crate::detector::OpticalFrame;
use crate::field::InputImage;
use crate::hash::{hash_bytes, hash_f32, hash_join};
use crate::mask::PhaseMask;
use crate::metrics::MetricReport;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentReceipt {
    pub experiment_id: String,
    pub input_hash: String,
    pub mask_hash: String,
    pub config_hash: String,
    pub output_hash: String,
    pub metrics_hash: String,
    /// Provenance fields (ADR-260 §15).
    pub git_commit: String,
    pub rustc_version: String,
    pub feature_flags: Vec<String>,
    pub seed: u64,
    /// Digest over all of the above — the single anti-swap value.
    pub rvf_receipt_hash: String,
}

/// Provenance captured at build/run time.
#[derive(Clone, Debug, Default)]
pub struct Provenance {
    pub git_commit: String,
    pub rustc_version: String,
    pub feature_flags: Vec<String>,
}

pub fn hash_input(img: &InputImage) -> String {
    hash_f32("photonlayer.input.v1", &[img.width, img.height], &img.pixels)
}

pub fn hash_mask(mask: &PhaseMask) -> String {
    hash_f32(
        "photonlayer.mask.v1",
        &[mask.width, mask.height],
        &mask.phase_radians,
    )
}

pub fn hash_config(config: &OpticalConfig) -> String {
    // Canonical JSON keeps the digest stable across serde versions. `OpticalConfig`
    // is plain serializable data, so this cannot fail in practice — `expect` rather
    // than `unwrap_or_default`, which would silently hash empty bytes and let two
    // distinct (both-failing) configs collide.
    let json = serde_json::to_vec(config).expect("OpticalConfig must serialize");
    hash_bytes("photonlayer.config.v1", &json)
}

/// Build a fully-bound receipt for a finished experiment.
pub fn build_receipt(
    experiment_id: impl Into<String>,
    input: &InputImage,
    mask: &PhaseMask,
    config: &OpticalConfig,
    frame: &OpticalFrame,
    metrics: &MetricReport,
    prov: &Provenance,
) -> ExperimentReceipt {
    let experiment_id = experiment_id.into();
    let input_hash = hash_input(input);
    let mask_hash = hash_mask(mask);
    let config_hash = hash_config(config);
    let output_hash = frame.frame_hash.clone();
    let metrics_hash = metrics.metrics_hash();
    let flags = prov.feature_flags.join(",");

    let rvf_receipt_hash = hash_join(
        "photonlayer.receipt.v1",
        &[
            &experiment_id,
            &input_hash,
            &mask_hash,
            &config_hash,
            &output_hash,
            &metrics_hash,
            &prov.git_commit,
            &prov.rustc_version,
            &flags,
            &config.seed.to_string(),
        ],
    );

    ExperimentReceipt {
        experiment_id,
        input_hash,
        mask_hash,
        config_hash,
        output_hash,
        metrics_hash,
        git_commit: prov.git_commit.clone(),
        rustc_version: prov.rustc_version.clone(),
        feature_flags: prov.feature_flags.clone(),
        seed: config.seed,
        rvf_receipt_hash,
    }
}

/// Check the receipt's **internal consistency** only: recompute the binding
/// digest from the stored fields and compare it to `rvf_receipt_hash`.
///
/// This proves the stored fields have not been individually altered or swapped
/// (tamper-evidence) — it does **not** prove the experiment actually produced
/// `output_hash` from `input_hash`/`mask_hash`/`config_hash`. A fully
/// self-consistent receipt with fabricated content hashes will pass this check.
/// To prove the computation, replay it with [`replay_and_verify`], which re-runs
/// the pipeline and compares the freshly-computed hashes.
pub fn verify_receipt(r: &ExperimentReceipt) -> bool {
    let flags = r.feature_flags.join(",");
    let expected = hash_join(
        "photonlayer.receipt.v1",
        &[
            &r.experiment_id,
            &r.input_hash,
            &r.mask_hash,
            &r.config_hash,
            &r.output_hash,
            &r.metrics_hash,
            &r.git_commit,
            &r.rustc_version,
            &flags,
            &r.seed.to_string(),
        ],
    );
    expected == r.rvf_receipt_hash
}

/// **Full** verification: re-run the experiment from its inputs and confirm the
/// receipt's bound hashes match a fresh deterministic run — not merely that the
/// stored fields are internally consistent (all [`verify_receipt`] checks).
///
/// Returns `true` iff the receipt is self-consistent **and** replaying
/// `input + mask + config` reproduces `input_hash`, `mask_hash`, `config_hash`,
/// and `output_hash`. This is the check that turns a receipt from "these fields
/// were hashed together" into "this output really came from these inputs."
pub fn replay_and_verify(
    r: &ExperimentReceipt,
    input: &InputImage,
    mask: &PhaseMask,
    config: &OpticalConfig,
) -> bool {
    use crate::simulator::{OpticalSimulator, ScalarSimulator};
    if !verify_receipt(r) {
        return false;
    }
    let frame = match ScalarSimulator.simulate(input, mask, config) {
        Ok(f) => f,
        Err(_) => return false,
    };
    hash_input(input) == r.input_hash
        && hash_mask(mask) == r.mask_hash
        && hash_config(config) == r.config_hash
        && frame.frame_hash == r.output_hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OpticalConfig;
    use crate::detector::capture;
    use crate::field::{InputImage, OpticalField};
    use crate::propagate::propagate;

    fn run() -> ExperimentReceipt {
        let n = 16;
        let px: Vec<f32> = (0..n * n).map(|i| (i % 3) as f32 / 2.0).collect();
        let img = InputImage::from_norm_f32(n, n, px).unwrap();
        let field = OpticalField::from_image(&img, n, n).unwrap();
        let mask = PhaseMask::random(n, n, 7);
        let mut f2 = field.clone();
        mask.apply(&mut f2).unwrap();
        let cfg = OpticalConfig::demo(n, n);
        let out = propagate(&f2, &cfg).unwrap();
        let frame = capture(&out, &cfg);
        let metrics = MetricReport::default();
        build_receipt("exp-1", &img, &mask, &cfg, &frame, &metrics, &Provenance::default())
    }

    #[test]
    fn receipt_verifies() {
        let r = run();
        assert!(verify_receipt(&r));
    }

    #[test]
    fn tamper_breaks_receipt() {
        let mut r = run();
        r.output_hash.push('x');
        assert!(!verify_receipt(&r));
    }

    #[test]
    fn replay_is_deterministic() {
        let a = run();
        let b = run();
        assert_eq!(a.rvf_receipt_hash, b.rvf_receipt_hash);
    }

    /// Build a receipt together with the exact inputs that produced it.
    fn run_with_inputs() -> (ExperimentReceipt, InputImage, PhaseMask, OpticalConfig) {
        let n = 16;
        let px: Vec<f32> = (0..n * n).map(|i| (i % 3) as f32 / 2.0).collect();
        let img = InputImage::from_norm_f32(n, n, px).unwrap();
        let mask = PhaseMask::random(n, n, 7);
        let cfg = OpticalConfig::demo(n, n);
        let mut f2 = OpticalField::from_image(&img, n, n).unwrap();
        mask.apply(&mut f2).unwrap();
        let frame = capture(&propagate(&f2, &cfg).unwrap(), &cfg);
        let r = build_receipt(
            "exp-1", &img, &mask, &cfg, &frame, &MetricReport::default(), &Provenance::default(),
        );
        (r, img, mask, cfg)
    }

    #[test]
    fn replay_verifies_real_run() {
        let (r, img, mask, cfg) = run_with_inputs();
        assert!(replay_and_verify(&r, &img, &mask, &cfg));
    }

    #[test]
    fn replay_rejects_wrong_inputs() {
        let (r, img, _mask, cfg) = run_with_inputs();
        // A different mask must NOT reproduce output_hash, even though the
        // receipt is internally consistent (verify_receipt still passes).
        let other = PhaseMask::random(img.width, img.height, 9999);
        assert!(verify_receipt(&r));
        assert!(!replay_and_verify(&r, &img, &other, &cfg));
    }

    #[test]
    fn replay_rejects_fabricated_output_hash() {
        let (mut r, img, mask, cfg) = run_with_inputs();
        // Forge a self-consistent receipt with a bogus output: re-bind the digest
        // so verify_receipt passes, then show replay still catches it.
        r.output_hash = "deadbeef".into();
        r.rvf_receipt_hash = hash_join(
            "photonlayer.receipt.v1",
            &[
                &r.experiment_id, &r.input_hash, &r.mask_hash, &r.config_hash,
                &r.output_hash, &r.metrics_hash, &r.git_commit, &r.rustc_version,
                &r.feature_flags.join(","), &r.seed.to_string(),
            ],
        );
        assert!(verify_receipt(&r), "forged receipt is internally consistent");
        assert!(!replay_and_verify(&r, &img, &mask, &cfg), "replay must catch the forgery");
    }
}
