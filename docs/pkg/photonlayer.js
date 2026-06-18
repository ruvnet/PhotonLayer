/* @ts-self-types="./photonlayer.d.ts" */

/**
 * One-call compression result for the browser playground (ADR-260 product demo).
 *
 * Bundles everything the side-by-side UI needs from a single optical run:
 * the original image (for display), the pooled sensor measurement rendered as
 * a noise-like image, the byte-count reduction, and the deterministic frame
 * hash that doubles as a verifiable receipt.
 */
export class CompressResult {
    static __wrap(ptr) {
        const obj = Object.create(CompressResult.prototype);
        obj.__wbg_ptr = ptr;
        CompressResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CompressResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_compressresult_free(ptr, 0);
    }
    /**
     * BLAKE3 hex digest of the measurement frame — the verifiable receipt.
     * @returns {string}
     */
    get frame_hash() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.compressresult_frame_hash(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export3(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Raw byte count of the original grayscale image (`w * h`).
     * @returns {number}
     */
    get input_bytes() {
        const ret = wasm.compressresult_input_bytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Height of the pooled sensor measurement.
     * @returns {number}
     */
    get meas_height() {
        const ret = wasm.compressresult_meas_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Width of the pooled sensor measurement.
     * @returns {number}
     */
    get meas_width() {
        const ret = wasm.compressresult_meas_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Pooled sensor measurement ("strange pattern"), normalized 0..255.
     * @returns {Uint8Array}
     */
    get measurement_buf() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.compressresult_measurement_buf(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU8FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export3(r0, r1 * 1, 1);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Byte count of the stored pooled measurement (`meas_w * meas_h`).
     * @returns {number}
     */
    get measurement_bytes() {
        const ret = wasm.compressresult_measurement_bytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Original grayscale image, row-major (canvas-ready).
     * @returns {Uint8Array}
     */
    get orig_buf() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.compressresult_orig_buf(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU8FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export3(r0, r1 * 1, 1);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Height of the original input image.
     * @returns {number}
     */
    get orig_height() {
        const ret = wasm.compressresult_orig_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Width of the original input image.
     * @returns {number}
     */
    get orig_width() {
        const ret = wasm.compressresult_orig_width(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) CompressResult.prototype[Symbol.dispose] = CompressResult.prototype.free;

/**
 * All five view buffers returned to JavaScript.
 *
 * Getters returning `Vec<u8>` copy the data into a fresh JS `Uint8Array`
 * each call — suitable for passing to `ImageData` or `canvas.putImageData`.
 */
export class WasmTraceResult {
    static __wrap(ptr) {
        const obj = Object.create(WasmTraceResult.prototype);
        obj.__wbg_ptr = ptr;
        WasmTraceResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmTraceResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmtraceresult_free(ptr, 0);
    }
    /**
     * BLAKE3 hex digest of the sensor frame (anti-swap determinism proof).
     * @returns {string}
     */
    get frame_hash() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmtraceresult_frame_hash(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export3(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Grid height in pixels.
     * @returns {number}
     */
    get height() {
        const ret = wasm.wasmtraceresult_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * View 1: amplitude of the incoming optical field, normalized to 0..255.
     * @returns {Uint8Array}
     */
    get incoming_buf() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmtraceresult_incoming_buf(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU8FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export3(r0, r1 * 1, 1);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * View 2: phase mask mapped 0..2π → 0..255.
     * @returns {Uint8Array}
     */
    get mask_buf() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmtraceresult_mask_buf(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU8FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export3(r0, r1 * 1, 1);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * View 3: masked-field intensity normalized to 0..255.
     * @returns {Uint8Array}
     */
    get masked_intensity_buf() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmtraceresult_masked_intensity_buf(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU8FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export3(r0, r1 * 1, 1);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * View 4: sensor capture ("strange pattern"), normalized to 0..255.
     * @returns {Uint8Array}
     */
    get sensor_buf() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmtraceresult_sensor_buf(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU8FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export3(r0, r1 * 1, 1);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Grid width in pixels.
     * @returns {number}
     */
    get width() {
        const ret = wasm.wasmtraceresult_width(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) WasmTraceResult.prototype[Symbol.dispose] = WasmTraceResult.prototype.free;

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
 * @param {Uint8Array} image_bytes
 * @param {number} w
 * @param {number} h
 * @param {string} mask_kind
 * @param {bigint} mask_seed
 * @param {number} mask_strength
 * @param {number} distance_mm
 * @param {number} pool
 * @returns {CompressResult}
 */
export function compress(image_bytes, w, h, mask_kind, mask_seed, mask_strength, distance_mm, pool) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passArray8ToWasm0(image_bytes, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(mask_kind, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len1 = WASM_VECTOR_LEN;
        wasm.compress(retptr, ptr0, len0, w, h, ptr1, len1, mask_seed, mask_strength, distance_mm, pool);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return CompressResult.__wrap(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * Return a JSON-serialized [`OpticalConfig::demo`] for the given dimensions.
 *
 * JavaScript can call this to obtain a valid starting config, then pass it
 * (possibly modified) back to `simulate`.
 * @param {number} width
 * @param {number} height
 * @returns {string}
 */
export function default_config_json(width, height) {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.default_config_json(retptr, width, height);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export3(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Crate version, exported to verify the WASM module loads.
 * @returns {string}
 */
export function photonlayer_version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.photonlayer_version(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export3(deferred1_0, deferred1_1, 1);
    }
}

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
 * @param {Uint8Array} image_bytes
 * @param {number} w
 * @param {number} h
 * @param {string} mask_kind
 * @param {bigint} mask_seed
 * @param {number} mask_strength
 * @param {string} config_json
 * @returns {WasmTraceResult}
 */
export function simulate(image_bytes, w, h, mask_kind, mask_seed, mask_strength, config_json) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passArray8ToWasm0(image_bytes, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(mask_kind, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(config_json, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len2 = WASM_VECTOR_LEN;
        wasm.simulate(retptr, ptr0, len0, w, h, ptr1, len1, mask_seed, mask_strength, ptr2, len2);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return WasmTraceResult.__wrap(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * Parse an [`ExperimentReceipt`] from JSON and verify its internal consistency.
 *
 * Returns `true` iff the receipt's `rvf_receipt_hash` matches a fresh
 * re-derivation over all bound fields — proving the output was not swapped.
 * @param {string} json
 * @returns {boolean}
 */
export function verify_receipt_json(json) {
    const ptr0 = passStringToWasm0(json, wasm.__wbindgen_export, wasm.__wbindgen_export2);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.verify_receipt_json(ptr0, len0);
    return ret !== 0;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_ea4887a5f8f9a9db: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return addHeapObject(ret);
        },
    };
    return {
        __proto__: null,
        "./photonlayer_bg.js": import0,
    };
}

const CompressResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_compressresult_free(ptr, 1));
const WasmTraceResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmtraceresult_free(ptr, 1));

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function dropObject(idx) {
    if (idx < 1028) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getObject(idx) { return heap[idx]; }

let heap = new Array(1024).fill(undefined);
heap.push(undefined, null, true, false);

let heap_next = heap.length;

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('photonlayer_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
