// Prompt words by denoiser call: each word's share of the text attention at every recorded
// call. Pure functions; WordSteps.jsx draws them.

/** Padding runs (repeated special tokens) stay out of the grid. */
export const isPad = (labels, i) =>
  labels?.[i] !== undefined &&
  /^<\|?(endoftext|pad|register)\|?>$/i.test(labels[i]) &&
  i > 0 &&
  labels[i - 1] === labels[i];

/** Each prompt token's share of the text attention at each recorded denoiser call.
 * Only the `most` tokens with the largest share at some call get a row of their own, in
 * prompt order; the rest are folded into one row, so a long encoder prompt stays one card. */
export function wordStepGrid(events, labels, most = 24) {
  const list = (events || []).filter((e) => Array.isArray(e.token_mass));
  if (!list.length) return null;
  const n = list[0].token_mass.length;
  const all = [];
  for (let t = 0; t < n; t++) if (!isPad(labels, t)) all.push(t);
  const byCall = list.map((e) => {
    const total = all.reduce((s, t) => s + (e.token_mass[t] || 0), 0) || 1;
    return new Map(all.map((t) => [t, (e.token_mass[t] || 0) / total]));
  });
  const peakOf = (t) => Math.max(...byCall.map((m) => m.get(t) || 0));
  const kept = new Set(
    [...all].sort((a, b) => peakOf(b) - peakOf(a)).slice(0, most),
  );
  const rows = all.filter((t) => kept.has(t));
  const others = all.filter((t) => !kept.has(t));
  const shares = byCall.map((m) => rows.map((t) => m.get(t) || 0));
  const rest = others.length
    ? {
        count: others.length,
        shares: byCall.map((m) =>
          others.reduce((s, t) => s + (m.get(t) || 0), 0),
        ),
      }
    : null;
  const peak = Math.max(1e-9, ...shares.flat());
  return {
    rows,
    total: all.length,
    rest,
    calls: list.map((e) => e.denoiser_call ?? e.step ?? null),
    cells: shares.map((col) => col.map((v) => v / peak)),
    shares,
    peak,
  };
}
