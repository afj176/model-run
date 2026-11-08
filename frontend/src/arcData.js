// Attention arcs: the few keys that carry one query's attention. Pure functions;
// AttentionArcs.jsx draws them.

/** The keys that hold `share` of one query's attention, strongest first, at most `most`. */
export function strongestKeys(row, share = 0.9, most = 8) {
  if (!Array.isArray(row)) return [];
  const keys = row
    .map((w, key) => ({ key, weight: Number.isFinite(w) ? w : 0 }))
    .filter((k) => k.weight > 0)
    .sort((a, b) => b.weight - a.weight);
  const out = [];
  let mass = 0;
  for (const k of keys) {
    out.push(k);
    mass += k.weight;
    if (mass >= share || out.length >= most) break;
  }
  return out;
}
