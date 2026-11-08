// The lens river: every candidate that reaches a layer's top list, with its probability per
// layer. Pure functions; LensRiver.jsx draws them.

/** The lanes of a logit lens: every token that reaches a layer's top list, with its probability per layer. */
export function riverLanes(lens, names, most = 8) {
  const layers = lens?.layers || [];
  const seen = new Map();
  layers.forEach((layer, l) => {
    for (const t of layer.top_tokens || []) {
      const lane = seen.get(t.token_id) ?? {
        token_id: t.token_id,
        text: names?.get?.(t.token_id) ?? t.text,
        p: new Array(layers.length).fill(0),
        peak: 0,
        last: 0,
      };
      lane.p[l] = t.probability;
      lane.peak = Math.max(lane.peak, t.probability);
      lane.last = t.probability;
      seen.set(t.token_id, lane);
    }
    for (const lane of seen.values()) if (lane.p[l] === 0) lane.last = 0;
  });
  const final = lens?.final_token_id;
  return [...seen.values()]
    .sort(
      (a, b) =>
        (a.token_id === final) - (b.token_id === final) || b.peak - a.peak,
    )
    .reverse()
    .slice(0, most)
    .map((lane, i) => ({ ...lane, final: lane.token_id === final }));
}
