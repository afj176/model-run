// Prompt words by denoiser call: each word's share of the text attention at every recorded
// call. Pure functions; WordSteps.jsx draws them.

/** Padding runs (repeated special tokens) stay out of the grid. */
export const isPad = (labels, i) =>
  labels?.[i] !== undefined &&
  /^<\|?(endoftext|pad|register)\|?>$/i.test(labels[i]) &&
  i > 0 &&
  labels[i - 1] === labels[i];

/** Each prompt token's share of the text attention at each recorded denoiser call. */
export function wordStepGrid(events, labels) {
  const list = (events || []).filter((e) => Array.isArray(e.token_mass));
  if (!list.length) return null;
  const n = list[0].token_mass.length;
  const rows = [];
  for (let t = 0; t < n; t++) if (!isPad(labels, t)) rows.push(t);
  const shares = list.map((e) => {
    const total = rows.reduce((s, t) => s + (e.token_mass[t] || 0), 0) || 1;
    return rows.map((t) => (e.token_mass[t] || 0) / total);
  });
  const peak = Math.max(1e-9, ...shares.flat());
  return {
    rows,
    calls: list.map((e) => e.denoiser_call ?? e.step ?? null),
    cells: shares.map((col) => col.map((v) => v / peak)),
    shares,
    peak,
  };
}
