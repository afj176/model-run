// The layer × token weather map: how large each layer's output, and what its attention and MLP
// added, was at every token. Pure functions over a run's events; WeatherMap.jsx draws them.
import { layerOf } from "./insights.js";

const finite = (v) => typeof v === "number" && Number.isFinite(v);
const norms = (event) => {
  const rms = event?.slices?.token_rms ?? event?.token_rms;
  return Array.isArray(rms) ? rms.map((v) => (finite(v) ? v : null)) : null;
};

/** One row per captured layer, in layer order: the output's per-token RMS and each part's. */
export function layerTokenGrid(events) {
  const byLayer = new Map();
  for (const e of events || []) {
    const path = e.part ? e.part_of : e.module;
    const layer = layerOf(path);
    if (!Number.isInteger(layer)) continue;
    const row = norms(e);
    if (!row) continue;
    const entry = byLayer.get(layer) ?? {
      layer,
      path,
      output: null,
      attention: null,
      mlp: null,
    };
    if (e.part === "attention" || e.part === "mlp") entry[e.part] = row;
    else if (!entry.output) entry.output = row;
    byLayer.set(layer, entry);
  }
  const layers = [...byLayer.values()]
    .filter((l) => l.output || l.attention || l.mlp)
    .sort((a, b) => a.layer - b.layer);
  const tokens = layers.reduce(
    (n, l) =>
      Math.max(
        n,
        l.output?.length ?? 0,
        l.attention?.length ?? 0,
        l.mlp?.length ?? 0,
      ),
    0,
  );
  return {
    layers,
    tokens,
    measures: MEASURES.filter((m) => layers.some((l) => l[m.key])),
  };
}

export const MEASURES = [
  { key: "output", label: "layer output" },
  { key: "attention", label: "what attention added" },
  { key: "mlp", label: "what the MLP added" },
];

/** The grid's values for one measure on a shared 0…1 scale: each cell over the grid's
 * 98th percentile, so one massive token does not wash the rest out. */
export function scaledGrid(grid, measure) {
  const rows = grid.layers.map((l) => l[measure] ?? []);
  const all = rows
    .flat()
    .filter(finite)
    .sort((a, b) => a - b);
  const top = all.length
    ? all[Math.min(all.length - 1, Math.floor(all.length * 0.98))]
    : 1;
  const scale = Math.max(top, 1e-9);
  return {
    rows: rows.map((r) =>
      Array.from({ length: grid.tokens }, (_, t) =>
        finite(r[t]) ? Math.min(1, r[t] / scale) : null,
      ),
    ),
    scale,
  };
}

/** Per layer, the mean of a measure over tokens, and its spread (largest over the median). */
export function layerSummary(grid, measure) {
  return grid.layers.map((l) => {
    const row = (l[measure] ?? []).filter(finite);
    if (!row.length)
      return {
        layer: l.layer,
        mean: null,
        peak: null,
        median: null,
        spread: null,
      };
    const sorted = [...row].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const peak = sorted.at(-1);
    return {
      layer: l.layer,
      mean: row.reduce((a, b) => a + b, 0) / row.length,
      median,
      peak,
      spread: median > 0 ? peak / median : null,
      peakToken: (l[measure] ?? []).indexOf(peak),
    };
  });
}
