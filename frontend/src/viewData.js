import { mostOf, leastOf } from "./extent.js";
/** What sampling settings would do to one measured distribution. The model itself never sampled with them. */
export function samplingWhatIf(view, names, { temperature, topK, topP }) {
  const grid = view?.temperatures || [];
  if (!grid.length || !view?.top?.length) return null;
  const point = grid.reduce((best, g) =>
    Math.abs(g.temperature - temperature) <
    Math.abs(best.temperature - temperature)
      ? g
      : best,
  );
  const scored = view.top.map((t) => ({
    token_id: t.token_id,
    text: names?.get?.(t.token_id),
    logit: t.logit,
    probability: Math.exp(t.logit / point.temperature - point.log_total),
  }));
  const seen = scored.reduce((sum, t) => sum + t.probability, 0);
  let running = 0;
  let kept = 0;
  for (const token of scored) {
    if (kept >= topK) break;
    kept += 1;
    running += token.probability;
    if (running >= topP) break;
  }
  const total = scored
    .slice(0, kept)
    .reduce((sum, t) => sum + t.probability, 0);
  const rows = scored.map((token, index) => ({
    ...token,
    kept: index < kept,
    filtered: index < kept && total > 0 ? token.probability / total : 0,
  }));
  const entropy = rows.reduce(
    (sum, r) =>
      r.filtered > 0 ? sum - r.filtered * Math.log2(r.filtered) : sum,
    0,
  );
  return {
    temperature: point.temperature,
    rows,
    kept,
    seen,
    entropy,
    // Top-P stops early only if enough probability sits inside the recorded tokens.
    reachedTopP: running >= topP || kept >= topK,
  };
}

/** The strongest links between tokens, skipping a token's link to itself. */
export function attentionArcs(values, limit = 16) {
  if (!Array.isArray(values)) return [];
  const arcs = [];
  values.forEach((row, from) => {
    row.forEach((weight, to) => {
      if (from !== to && Number.isFinite(weight))
        arcs.push({ from, to, weight });
    });
  });
  return arcs.sort((a, b) => b.weight - a.weight).slice(0, limit);
}

/** Where each head looks from one query token, so heads can be compared side by side. */
export function headFanOut(heads, query) {
  if (!Array.isArray(heads)) return [];
  return heads.map((matrix, head) => {
    const row = matrix?.[query] || [];
    const peak = row.reduce((best, v, i) => (v > row[best] ? i : best), 0);
    return { head, row, peak, weight: row[peak] ?? 0 };
  });
}

/** Lines for a chart of one measured series, with its range for scaling. */
export function seriesRange(values) {
  const finite = values.filter((v) => Number.isFinite(v));
  return finite.length
    ? { low: leastOf(finite), high: mostOf(finite) }
    : { low: 0, high: 1 };
}

/** One sampler step in words, using only what was recorded. */
export function denoisingStory(section, position) {
  const steps = section?.steps || [];
  const step = steps[position];
  if (!step) return "";
  const parts = [
    `Step ${step.index + 1}${section.total_steps ? ` of ${section.total_steps}` : ""}, at noise level ${step.sigma?.toFixed(2) ?? "unknown"}.`,
  ];
  const changes = steps.map((s) => s.estimate_change ?? -1);
  const biggest = changes.indexOf(mostOf(changes));
  if (biggest > 0)
    parts.push(
      biggest === position
        ? "The estimate of the final image moved more here than at any other step."
        : `The estimate moved most at step ${steps[biggest].index + 1}.`,
    );
  if (Number.isFinite(step.guidance_cosine))
    parts.push(
      `The positive and negative predictions are ${step.guidance_cosine >= 0.9995 ? "almost the same here" : `${(step.guidance_cosine * 100).toFixed(1)}% alike`}.`,
    );
  else parts.push("Guidance was off, so only one prediction was made.");
  return parts.join(" ");
}

/** Which tokens move one unit of one layer, strongest first. */
export function unitTokens(layer, unit, tokens) {
  const at = layer?.units?.indexOf(unit) ?? -1;
  if (at < 0) return [];
  return (layer.unit_values || [])
    .map((row, index) => ({
      index,
      text: tokens?.[index]?.text ?? `#${index}`,
      value: row[at],
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
}

/** Fit principal component points into a square, keeping their proportions so distances stay honest. */
export function fitPoints(layers, tokens, size = 100, margin = 6) {
  const all = (layers || []).flatMap((layer) =>
    layer.filter((_, i) => !tokens?.[i]?.special),
  );
  if (!all.length) return [];
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1] ?? 0);
  const [xLow, xHigh] = [leastOf(xs), mostOf(xs)];
  const [yLow, yHigh] = [leastOf(ys), mostOf(ys)];
  const span = Math.max(xHigh - xLow, yHigh - yLow, 1e-9);
  const scale = (size - 2 * margin) / span;
  const xOffset = (size - (xHigh - xLow) * scale) / 2;
  const yOffset = (size - (yHigh - yLow) * scale) / 2;
  return layers.map((layer) =>
    layer.map((p) => [
      xOffset + (p[0] - xLow) * scale,
      size - yOffset - ((p[1] ?? 0) - yLow) * scale,
    ]),
  );
}

/** Every recorded channel of a convolutional layer, for the positive prompt where the batch says which. */
export function featureSheet(event) {
  const slices = event?.slices;
  if (slices?.layout !== "features" || !Array.isArray(slices.values))
    return null;
  const size = slices.rows * slices.cols;
  const channelAt = slices.axes.findIndex((a) => a.name === "channel");
  if (channelAt < 0 || !size) return null;
  const channel = slices.axes[channelAt];
  const batchAt = slices.axes.findIndex((a) => a.name === "batch");
  const batch = batchAt >= 0 ? slices.axes[batchAt] : null;
  const batchPosition = batch
    ? Math.max(0, batch.indices.indexOf(batch.conditional))
    : 0;
  const stride = batch ? batch.indices.length : 1;
  const channels = channel.indices.map((index, position) => {
    const at =
      batchAt < channelAt
        ? batchPosition * channel.indices.length + position
        : position * stride + batchPosition;
    return {
      index,
      values: slices.values.slice(at * size, (at + 1) * size),
      mean: channel.mean_abs?.[position],
    };
  });
  return {
    rows: slices.rows,
    cols: slices.cols,
    size: channel.size,
    channels: channels.filter((c) => c.values.length === size),
  };
}

/** Each layer's all-channels map, in the order the layers ran. */
export function featureStrip(events) {
  const strip = [];
  const seen = new Set();
  for (const event of events || []) {
    const slices = event?.slices;
    if (slices?.layout !== "features" || seen.has(event.module)) continue;
    const size = slices.rows * slices.cols;
    if (!size || !Array.isArray(slices.aggregate)) continue;
    seen.add(event.module);
    const batch = slices.axes?.find((a) => a.name === "batch");
    const position = batch
      ? Math.max(0, batch.indices.indexOf(batch.conditional))
      : 0;
    strip.push({
      module: event.module,
      shape: event.shape,
      rows: slices.rows,
      cols: slices.cols,
      values: slices.aggregate.slice(position * size, (position + 1) * size),
    });
  }
  return strip;
}
