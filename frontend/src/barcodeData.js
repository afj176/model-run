// The token barcode: every token of a text layer as one row, every recorded feature as a stripe.
// Pure functions here; TokenBarcode.jsx draws them.

/** Rows, features and the value spread of a token-grid event; null for anything else. */
export function barcodeModel(event) {
  if (!event?.token_rms || !Array.isArray(event.values)) return null;
  const [rows, cols] = event.shape;
  if (!rows || !cols || event.values.length !== rows * cols) return null;
  const magnitudes = [];
  for (const v of event.values)
    if (Number.isFinite(v)) magnitudes.push(Math.abs(v));
  magnitudes.sort((a, b) => a - b);
  const quantile = (q) =>
    magnitudes.length
      ? magnitudes[
          Math.min(magnitudes.length - 1, Math.floor(magnitudes.length * q))
        ]
      : 0;
  const max = magnitudes.at(-1) || 0;
  return {
    rows,
    cols,
    values: event.values,
    labels: Array.from(
      { length: rows },
      (_, r) => event.row_labels?.[r] ?? `#${event.token_indices?.[r] ?? r}`,
    ),
    features: Array.from(
      { length: cols },
      (_, c) =>
        event.feature_indices?.[c] ??
        Number(String(event.col_labels?.[c] ?? c).replace("#", "")),
    ),
    rms: event.token_rms,
    // The robust scale saturates at the 99th percentile, so one huge channel does not wash out the rest.
    p99: Math.max(quantile(0.99), 1e-6),
    max: Math.max(max, 1e-6),
  };
}

export const SCALES = [
  [
    "robust",
    "Robust",
    "Saturates at the 99th percentile of |value|; brighter outliers are marked",
  ],
  ["log", "Signed log", "sign × log(1 + |value|), so every size stays visible"],
  ["linear", "Linear", "Plain value over the largest |value|"],
];

/** A value on −1…1 for colouring. */
export function scaled(v, mode, model) {
  if (!Number.isFinite(v)) return null;
  if (mode === "linear") return v / model.max;
  if (mode === "log")
    return (Math.sign(v) * Math.log1p(Math.abs(v))) / Math.log1p(model.max);
  return Math.max(-1, Math.min(1, v / model.p99));
}

export const isOutlier = (v, model) =>
  Number.isFinite(v) && Math.abs(v) > model.p99;

/** A massive activation: far above its token's typical size (RMS over all its features). */
export const MASSIVE = 10;
export const isMassive = (v, rms) =>
  Number.isFinite(v) && rms > 0 && Math.abs(v) > MASSIVE * rms;

/** The feature column in [c0, c1) with the largest |value| in a row: what one pixel column shows. */
export function strongest(model, row, c0, c1) {
  let best = -1,
    size = -1;
  const base = row * model.cols;
  for (let c = Math.max(0, c0); c < Math.min(model.cols, c1); c++) {
    const v = model.values[base + c];
    if (Number.isFinite(v) && Math.abs(v) > size) {
      size = Math.abs(v);
      best = c;
    }
  }
  return best;
}

/** Channels far above their tokens' typical size (|value| over `times` × the token's RMS), most tokens first. */
export function outlierChannels(model, times = MASSIVE, limit = 3) {
  const found = new Map();
  for (let r = 0; r < model.rows; r++) {
    const floor = times * (model.rms[r] || 0);
    if (!floor) continue;
    for (let c = 0; c < model.cols; c++) {
      const v = model.values[r * model.cols + c];
      if (!Number.isFinite(v) || Math.abs(v) <= floor) continue;
      const entry = found.get(c) ?? {
        col: c,
        feature: model.features[c],
        tokens: 0,
        peak: 0,
        row: r,
      };
      entry.tokens += 1;
      if (Math.abs(v) > Math.abs(entry.peak))
        Object.assign(entry, { peak: v, row: r });
      found.set(c, entry);
    }
  }
  return [...found.values()]
    .sort((a, b) => b.tokens - a.tokens || Math.abs(b.peak) - Math.abs(a.peak))
    .slice(0, limit);
}

/** Pool each token's features into at most a few thousand bars, keeping each column's strongest value. */
export function poolTokenGrid(values, rows, features, budget = 3072) {
  const cols = Math.max(1, Math.min(features, 128, Math.floor(budget / rows)));
  const per = features / cols;
  const out = [],
    source = [];
  for (let r = 0; r < rows; r++)
    for (let b = 0; b < cols; b++) {
      const c0 = Math.floor(b * per),
        c1 = Math.max(c0 + 1, Math.floor((b + 1) * per));
      let best = c0;
      for (let c = c0; c < c1; c++) {
        const v = values[r * features + c],
          w = values[r * features + best];
        if (
          Number.isFinite(v) &&
          (!Number.isFinite(w) || Math.abs(v) > Math.abs(w))
        )
          best = c;
      }
      source.push(r * features + best);
      out.push(values[r * features + best]);
    }
  const toBar = (index) => {
    const r = Math.floor(index / features),
      c = index % features;
    let b = Math.min(cols - 1, Math.floor(c / per));
    // Floor rounding can put a feature one column early; step to the column that holds it.
    while (
      b < cols - 1 &&
      c >= Math.max(Math.floor(b * per) + 1, Math.floor((b + 1) * per))
    )
      b++;
    return r * cols + b;
  };
  return { values: out, source, cols, per, toBar };
}
