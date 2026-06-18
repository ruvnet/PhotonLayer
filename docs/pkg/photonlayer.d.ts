/* tslint:disable */
/* eslint-disable */

/**
 * One-call compression result for the browser playground (ADR-260 product demo).
 *
 * Bundles everything the side-by-side UI needs from a single optical run:
 * the original image (for display), the pooled sensor measurement rendered as
 * a noise-like image, the byte-count reduction, and the deterministic frame
 * hash that doubles as a verifiable receipt.
 */
export class CompressResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * BLAKE3 hex digest of the measurement frame — the verifiable receipt.
     */
    readonly frame_hash: string;
    /**
     * Raw byte count of the original grayscale image (`w * h`).
     */
    readonly input_bytes: number;
    /**
     * Height of the pooled sensor measurement.
     */
    readonly meas_height: number;
    /**
     * Width of the pooled sensor measurement.
     */
    readonly meas_width: number;
    /**
     * Pooled sensor measurement ("strange pattern"), normalized 0..255.
     */
    readonly measurement_buf: Uint8Array;
    /**
     * Byte count of the stored pooled measurement (`meas_w * meas_h`).
     */
    readonly measurement_bytes: number;
    /**
     * Original grayscale image, row-major (canvas-ready).
     */
    readonly orig_buf: Uint8Array;
    /**
     * Height of the original input image.
     */
    readonly orig_height: number;
    /**
     * Width of the original input image.
     */
    readonly orig_width: number;
}

/**
 * All five view buffers returned to JavaScript.
 *
 * Getters returning `Vec<u8>` copy the data into a fresh JS `Uint8Array`
 * each call — suitable for passing to `ImageData` or `canvas.putImageData`.
 */
export class WasmTraceResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * BLAKE3 hex digest of the sensor frame (anti-swap determinism proof).
     */
    readonly frame_hash: string;
    /**
     * Grid height in pixels.
     */
    readonly height: number;
    /**
     * View 1: amplitude of the incoming optical field, normalized to 0..255.
     */
    readonly incoming_buf: Uint8Array;
    /**
     * View 2: phase mask mapped 0..2π → 0..255.
     */
    readonly mask_buf: Uint8Array;
    /**
     * View 3: masked-field intensity normalized to 0..255.
     */
    readonly masked_intensity_buf: Uint8Array;
    /**
     * View 4: sensor capture ("strange pattern"), normalized to 0..255.
     */
    readonly sensor_buf: Uint8Array;
    /**
     * Grid width in pixels.
     */
    readonly width: number;
}

/**
 * Run the full optical-compression pipeline in one call (browser playground).
 *
 * Applies a learned/preset phase mask, propagates over `distance_mm`, captures
 * the sensor intensity, and average-pools it by `pool` (the data-reduction
 * lever) into a smaller measurement. Returns a [`CompressResult`] with both
 * images, the byte reduction, and the deterministic frame hash (receipt).
 *
 * # Parameters
 * * `image_bytes` — row-major grayscale u8 pixels (len must equal `w * h`).
 * * `w` / `h` — image dimensions (power-of-two, ≤ `MAX_GRID_DIM`).
 * * `mask_kind` — `"identity"`, `"random"`, or `"lens"`.
 * * `mask_seed` — seed for `"random"` masks.
 * * `mask_strength` — focal strength for `"lens"` masks.
 * * `distance_mm` — propagation distance in millimetres.
 * * `pool` — average-pool / binning factor (1 = no pooling). Must divide `w`/`h`.
 *
 * Re-running with identical arguments yields an identical `frame_hash`, which
 * is exactly what the UI's "verify" button checks.
 *
 * Throws a JS error string on any failure.
 */
export function compress(image_bytes: Uint8Array, w: number, h: number, mask_kind: string, mask_seed: bigint, mask_strength: number, distance_mm: number, pool: number): CompressResult;

/**
 * Return a JSON-serialized [`OpticalConfig::demo`] for the given dimensions.
 *
 * JavaScript can call this to obtain a valid starting config, then pass it
 * (possibly modified) back to `simulate`.
 */
export function default_config_json(width: number, height: number): string;

/**
 * Crate version, exported to verify the WASM module loads.
 */
export function photonlayer_version(): string;

/**
 * Run the five-view optical simulation pipeline.
 *
 * # Parameters
 * * `image_bytes` — row-major grayscale u8 pixels (len must equal `w * h`).
 * * `w` / `h` — image dimensions.
 * * `mask_kind` — `"identity"`, `"random"`, or `"lens"`.
 * * `mask_seed` — seed for `"random"` masks (ignored for others).
 * * `mask_strength` — focal strength for `"lens"` masks (ignored for others).
 * * `config_json` — JSON-serialized [`OpticalConfig`]; empty → `demo` config.
 *
 * Returns a [`WasmTraceResult`] whose getter methods supply canvas-ready
 * grayscale buffers for each of the five studio views.
 *
 * Throws a JS error string on any failure.
 */
export function simulate(image_bytes: Uint8Array, w: number, h: number, mask_kind: string, mask_seed: bigint, mask_strength: number, config_json: string): WasmTraceResult;

/**
 * Parse an [`ExperimentReceipt`] from JSON and verify its internal consistency.
 *
 * Returns `true` iff the receipt's `rvf_receipt_hash` matches a fresh
 * re-derivation over all bound fields — proving the output was not swapped.
 */
export function verify_receipt_json(json: string): boolean;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_compressresult_free: (a: number, b: number) => void;
    readonly __wbg_wasmtraceresult_free: (a: number, b: number) => void;
    readonly compress: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: bigint, i: number, j: number, k: number) => void;
    readonly compressresult_frame_hash: (a: number, b: number) => void;
    readonly compressresult_input_bytes: (a: number) => number;
    readonly compressresult_meas_height: (a: number) => number;
    readonly compressresult_meas_width: (a: number) => number;
    readonly compressresult_measurement_buf: (a: number, b: number) => void;
    readonly compressresult_measurement_bytes: (a: number) => number;
    readonly compressresult_orig_buf: (a: number, b: number) => void;
    readonly compressresult_orig_height: (a: number) => number;
    readonly compressresult_orig_width: (a: number) => number;
    readonly default_config_json: (a: number, b: number, c: number) => void;
    readonly photonlayer_version: (a: number) => void;
    readonly simulate: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: bigint, i: number, j: number, k: number) => void;
    readonly verify_receipt_json: (a: number, b: number) => number;
    readonly wasmtraceresult_frame_hash: (a: number, b: number) => void;
    readonly wasmtraceresult_height: (a: number) => number;
    readonly wasmtraceresult_incoming_buf: (a: number, b: number) => void;
    readonly wasmtraceresult_mask_buf: (a: number, b: number) => void;
    readonly wasmtraceresult_masked_intensity_buf: (a: number, b: number) => void;
    readonly wasmtraceresult_sensor_buf: (a: number, b: number) => void;
    readonly wasmtraceresult_width: (a: number) => number;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
