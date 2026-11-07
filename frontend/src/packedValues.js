// Long lists of measured values travel packed: 32-bit floats in base64 instead of JSON text, about a third
// smaller, so a trace holds more. Missing values travel as NaN. The same format as trace_store.py.

const MIN_PACKED = 64;

function decode(base64) {
  const text = atob(base64);
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function encode(floats) {
  const bytes = new Uint8Array(floats.buffer);
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000)
    text += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

/** The value with every packed list turned back into numbers (null where none was measured). */
export function unpackValues(value) {
  if (Array.isArray(value)) return value.map(unpackValues);
  if (!value || typeof value !== "object") return value;
  if (value.__packed__ === "float32" && typeof value.data === "string") {
    const floats = decode(value.data);
    if (floats.length !== value.n) throw new Error("A packed list is damaged");
    const out = new Array(floats.length);
    // Single precision holds the 5 to 7 significant digits values are recorded with.
    for (let i = 0; i < floats.length; i++)
      out[i] = Number.isNaN(floats[i])
        ? null
        : Number(floats[i].toPrecision(7));
    return out;
  }
  const out = {};
  for (const key of Object.keys(value)) out[key] = unpackValues(value[key]);
  return out;
}

/** The value with long lists of measured (non-whole) numbers packed, for export. Whole-number lists stay as they are. */
export function packValues(value) {
  if (Array.isArray(value)) {
    const numeric =
      value.length >= MIN_PACKED &&
      value.every(
        (v) =>
          v === null ||
          (typeof v === "number" && Number.isFinite(v) && Math.abs(v) < 3e38),
      ) &&
      value.some((v) => typeof v === "number" && !Number.isInteger(v));
    if (!numeric) return value.map(packValues);
    const floats = new Float32Array(value.length);
    value.forEach((v, i) => (floats[i] = v === null ? NaN : v));
    return { __packed__: "float32", n: value.length, data: encode(floats) };
  }
  if (!value || typeof value !== "object") return value;
  const out = {};
  for (const key of Object.keys(value)) out[key] = packValues(value[key]);
  return out;
}
