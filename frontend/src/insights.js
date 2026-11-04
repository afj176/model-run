/** Insight computations over recorded values. Every number here is derived from
 * captured data; nothing is inferred from architecture names or assumptions. */

const finite = (v) => typeof v === "number" && Number.isFinite(v);

export function entropyBits(probabilities) {
  let total = 0;
  for (const p of probabilities) if (p > 0) total -= p * Math.log2(p);
  return total;
}

/** Decoded token text that stays readable as an axis label. */
export function tokenLabel(token, index) {
  if (!token) return String(index);
  const text = (token.text ?? "").replace(/\n/g, "⏎").replace(/\t/g, "⇥");
  const trimmed = text.trim();
  if (!trimmed) return text.length ? "␣" : "∅";
  return trimmed.length > 12 ? `${trimmed.slice(0, 11)}…` : trimmed;
}

/** Profile one attention matrix (query × key, rows sum to about 1). */
export function attentionHeadProfile(matrix) {
  const rows = matrix?.length || 0;
  if (!rows || !Array.isArray(matrix[0])) return null;
  let sink = 0,
    diagonal = 0,
    previous = 0,
    entropy = 0,
    maxWeight = 0,
    counted = 0,
    previousCounted = 0;
  for (let r = 0; r < rows; r++) {
    const row = matrix[r];
    let rowEntropy = 0;
    for (let c = 0; c < row.length; c++) {
      const v = row[c];
      if (!finite(v)) continue;
      if (v > maxWeight) maxWeight = v;
      if (v > 0) rowEntropy -= v * Math.log2(v);
    }
    if (finite(row[0])) sink += row[0];
    if (finite(row[r])) diagonal += row[r];
    if (r > 0 && finite(row[r - 1])) {
      previous += row[r - 1];
      previousCounted++;
    }
    entropy += rowEntropy;
    counted++;
  }
  const profile = {
    rows,
    sinkMass: sink / counted,
    diagonalMass: diagonal / counted,
    previousTokenMass: previousCounted ? previous / previousCounted : 0,
    entropyBits: entropy / counted,
    maxWeight,
  };
  const uniform = Math.log2(Math.max(2, rows));
  profile.focus = 1 - Math.min(1, profile.entropyBits / uniform);
  return { ...profile, ...classifyHead(profile) };
}

export function classifyHead(p) {
  if (!p) return { label: "Unknown", description: "No recorded weights." };
  if (p.sinkMass >= 0.5)
    return {
      label: "Attention sink",
      description: `Most attention (${pct(p.sinkMass)} on average) lands on the first token regardless of the query; this head is largely parked.`,
    };
  if (p.previousTokenMass >= 0.4)
    return {
      label: "Previous-token head",
      description: `Each position looks mainly at the token right before it (${pct(p.previousTokenMass)} average weight), a local pattern used to build bigrams.`,
    };
  if (p.diagonalMass >= 0.4)
    return {
      label: "Self-attending head",
      description: `Queries keep ${pct(p.diagonalMass)} of their weight on their own position; this head mostly passes information through.`,
    };
  if (p.focus >= 0.6)
    return {
      label: "Focused head",
      description: `Attention is concentrated: each query spends about ${p.entropyBits.toFixed(1)} bits, so it picks out one or two specific tokens.`,
    };
  if (p.focus <= 0.25)
    return {
      label: "Diffuse head",
      description: `Attention spreads broadly (${p.entropyBits.toFixed(1)} bits per query); this head averages context rather than selecting from it.`,
    };
  return {
    label: "Mixed head",
    description: `Attention is partly concentrated (${p.entropyBits.toFixed(1)} bits per query) with ${pct(p.sinkMass)} parked on the first token.`,
  };
}

/** The strongest keys for one query row. */
export function rowSummary(matrix, row, tokens, limit = 3) {
  const values = matrix?.[row];
  if (!Array.isArray(values)) return [];
  return values
    .map((weight, key) => ({
      key,
      weight,
      label: tokenLabel(tokens?.[key], key),
    }))
    .filter((entry) => finite(entry.weight) && entry.weight > 0)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, limit);
}

/** Statistics of a captured value prefix. Nulls are non-finite values, not zeros. */
export function activationStats(values, { outliers = 5 } = {}) {
  const sample = (values || []).filter(finite);
  if (!sample.length) return null;
  let sum = 0,
    sumSquares = 0,
    min = Infinity,
    max = -Infinity;
  for (const v of sample) {
    sum += v;
    sumSquares += v * v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const n = sample.length;
  const rms = Math.sqrt(sumSquares / n);
  const threshold = rms * 1e-2;
  let near = 0,
    negative = 0;
  for (const v of sample) {
    if (Math.abs(v) <= threshold) near++;
    if (v < 0) negative++;
  }
  const ranked = (values || [])
    .map((v, index) => ({ index, value: v }))
    .filter((e) => finite(e.value))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, outliers);
  const maxAbs = Math.max(Math.abs(min), Math.abs(max));
  return {
    count: n,
    nonFinite: (values || []).length - n,
    mean: sum / n,
    rms,
    min,
    max,
    sparsity: near / n,
    negativeShare: negative / n,
    outlierRatio: rms > 0 ? maxAbs / rms : 0,
    outliers: ranked,
  };
}

export function describeActivation(stats, module) {
  if (!stats) return "No finite values were captured.";
  const name = module ? `${module}` : "This tensor";
  const parts = [];
  if (stats.outlierRatio >= 20)
    parts.push(
      `${name} has massive activations: its largest value is ${stats.outlierRatio.toFixed(0)}× the RMS, concentrated in index ${stats.outliers[0]?.index}.`,
    );
  else if (stats.outlierRatio >= 8)
    parts.push(
      `${name} has a few dominant channels (peak ${stats.outlierRatio.toFixed(0)}× the RMS).`,
    );
  else
    parts.push(
      `${name} is evenly scaled (peak ${stats.outlierRatio.toFixed(1)}× the RMS).`,
    );
  if (stats.sparsity >= 0.5)
    parts.push(
      `${pct(stats.sparsity)} of values are near zero, so the layer is sparse.`,
    );
  else if (stats.sparsity >= 0.2)
    parts.push(`${pct(stats.sparsity)} of values are near zero.`);
  if (stats.negativeShare >= 0.45 && stats.negativeShare <= 0.55)
    parts.push("Signs are balanced.");
  else if (stats.negativeShare < 0.2)
    parts.push(`Mostly positive (${pct(1 - stats.negativeShare)}).`);
  else if (stats.negativeShare > 0.8)
    parts.push(`Mostly negative (${pct(stats.negativeShare)}).`);
  return parts.join(" ");
}

const layerOf = (module) => {
  const m =
    /(?:layers|blocks|transformer_blocks|double_blocks|single_blocks|joint_blocks)\.(\d+)/.exec(
      module || "",
    );
  return m ? Number(m[1]) : null;
};

/** RMS of each captured event and the RMS change from the previous call of the same module. */
export function callDeltas(events) {
  const last = new Map();
  return (events || []).map((event, index) => {
    const key = event.module ?? "step";
    const ordinal = event.denoiser_call ?? event.step ?? index;
    const values = event.values || [];
    const stats = activationStats(values);
    let delta = null;
    const previous = last.get(key);
    if (previous && previous.values.length === values.length && values.length) {
      let sum = 0,
        n = 0;
      for (let i = 0; i < values.length; i++) {
        const a = values[i],
          b = previous.values[i];
        if (!finite(a) || !finite(b)) continue;
        sum += (a - b) * (a - b);
        n++;
      }
      delta = n ? Math.sqrt(sum / n) : null;
    }
    last.set(key, { values });
    return {
      index,
      module: event.module,
      ordinal,
      layer: layerOf(event.module),
      rms: stats?.rms ?? null,
      delta,
      relativeDelta: delta !== null && stats?.rms ? delta / stats.rms : null,
    };
  });
}

/** A per-layer metric series for the ordinal timeline. */
export function layerSeries(trace, events) {
  if (
    trace?.kind === "attention_capture" &&
    Array.isArray(trace.layers) &&
    trace.layers.length
  )
    return {
      label: "Attention entropy (bits, head mean)",
      points: trace.layers.map((l) => ({
        layer: l.index,
        value: l.entropy_bits,
      })),
      secondary: {
        label: "Sink mass",
        points: trace.layers.map((l) => ({
          layer: l.index,
          value: l.sink_mass,
        })),
      },
    };
  if (trace?.logit_lens?.layers?.length)
    return {
      label: "Probability of final token (logit lens)",
      points: trace.logit_lens.layers
        .filter((l) => l.index > 0)
        .map((l) => ({ layer: l.index - 1, value: l.final_token_probability })),
    };
  const perLayer = new Map();
  for (const event of events || []) {
    const layer = layerOf(event.module);
    if (layer === null) continue;
    const stats = activationStats(event.values);
    if (!stats) continue;
    if (!perLayer.has(layer)) perLayer.set(layer, stats.rms);
  }
  if (!perLayer.size) return null;
  return {
    label: "Activation RMS (captured prefix)",
    points: [...perLayer]
      .sort((a, b) => a[0] - b[0])
      .map(([layer, value]) => ({ layer, value })),
  };
}

/** Where the logit lens first settles on the final answer and how confidence grows. */
export function lensStory(lens, tokens) {
  const layers = lens?.layers;
  if (!Array.isArray(layers) || !layers.length) return null;
  const finalId = lens.final_token_id;
  const finalText = tokens?.get?.(finalId) ?? layers.at(-1).top_tokens[0]?.text;
  const ranksFirst = layers.map((l) => l.top_tokens[0]?.token_id === finalId);
  const firstIndex = ranksFirst.indexOf(true);
  let settled = null;
  for (let i = layers.length - 1; i >= 0; i--) {
    if (ranksFirst[i]) settled = i;
    else break;
  }
  const peak = layers.reduce(
    (best, l) => (l.entropy_bits > best.entropy_bits ? l : best),
    layers[0],
  );
  return {
    finalText,
    firstIndex: firstIndex < 0 ? null : firstIndex,
    firstLabel: firstIndex < 0 ? null : layers[firstIndex].label,
    settledIndex: settled,
    settledLabel: settled === null ? null : layers[settled].label,
    wavered: firstIndex >= 0 && settled !== null && firstIndex !== settled,
    layersTotal: layers.length,
    uncertainLabel: peak.label,
    finalProbability: layers.at(-1).final_token_probability,
    firstProbability: layers[0].final_token_probability,
  };
}

/** Pairwise comparison of two attention matrices with the same shape. */
export function compareMatrices(a, b) {
  if (
    !a?.length ||
    !b?.length ||
    a.length !== b.length ||
    a[0].length !== b[0].length
  )
    return null;
  let sum = 0,
    n = 0,
    maxDiff = 0,
    at = [0, 0];
  const diff = a.map((row, r) =>
    row.map((v, c) => {
      const w = b[r][c];
      if (!finite(v) || !finite(w)) return null;
      const d = v - w;
      sum += Math.abs(d);
      n++;
      if (Math.abs(d) > maxDiff) {
        maxDiff = Math.abs(d);
        at = [r, c];
      }
      return d;
    }),
  );
  return { diff, meanAbs: n ? sum / n : 0, maxDiff, at };
}

export function pct(v) {
  return `${Math.round((finite(v) ? v : 0) * 100)}%`;
}

/** Readable token text for narrative sentences. */
export function tokenText(text) {
  if (text === undefined || text === null) return "?";
  if (text === "") return "\u2205";
  return text.replace(/\n/g, "\u23ce").replace(/ /g, "\u2423");
}

/** What a head ablation sweep says about one head and about the layer as a whole. */
export function ablationStory(ablation, head) {
  const heads = ablation?.heads;
  if (!Array.isArray(heads) || !heads.length || !ablation.baseline) return null;
  const baseline = ablation.baseline.top_tokens[0];
  const ranked = [...heads].sort((a, b) => b.kl_bits - a.kl_bits);
  const peak = Math.max(1e-9, ...ranked.map((h) => h.kl_bits));
  const entry = heads.find((h) => h.head === head) || null;
  let text = null;
  let effect = null;
  if (entry) {
    const after = entry.top_tokens[0];
    const before = baseline.probability;
    const drop = before - entry.baseline_top_probability;
    effect =
      entry.kl_bits >= 1
        ? "decisive"
        : entry.kl_bits >= 0.1
          ? "noticeable"
          : "negligible";
    const rank = ranked.findIndex((h) => h.head === head) + 1;
    if (entry.top_token_changed)
      text = `Without head ${head} the answer changes: \u201c${tokenText(after.text)}\u201d takes over at ${pct(after.probability)} and \u201c${tokenText(baseline.text)}\u201d falls from ${pct(before)} to ${pct(entry.baseline_top_probability)}.`;
    else if (drop > 0.02)
      text = `Without head ${head} the answer holds, but confidence in \u201c${tokenText(baseline.text)}\u201d drops from ${pct(before)} to ${pct(entry.baseline_top_probability)}.`;
    else if (drop < -0.02)
      text = `Without head ${head} the model is more sure of \u201c${tokenText(baseline.text)}\u201d: ${pct(before)} rises to ${pct(entry.baseline_top_probability)}, so this head was pulling against the answer.`;
    else
      text = `Removing head ${head} barely moves the prediction: \u201c${tokenText(baseline.text)}\u201d stays at about ${pct(before)}.`;
    text += ` The distribution shifts by ${entry.kl_bits.toFixed(entry.kl_bits >= 0.1 ? 2 : 3)} bits${heads.length > 1 ? `, rank ${rank} of ${heads.length} measured heads` : ""}.`;
  }
  return {
    entry,
    effect,
    text,
    ranked,
    peak,
    baseline,
    complete: heads.length === ablation.head_count,
    strongest: ranked[0],
    changers: ranked.filter((h) => h.top_token_changed).map((h) => h.head),
  };
}

/** Ranking and narrative for one cross-attention event; ``token`` focuses one map. */
export function crossAttentionStory(event, labels, token = null) {
  const mass = event?.token_mass;
  if (!Array.isArray(mass) || !mass.length || !Array.isArray(event.maps))
    return null;
  const ranked = mass
    .map((m, index) => ({ index, mass: m }))
    .sort((a, b) => b.mass - a.mass);
  const peakMass = Math.max(1e-9, ranked[0].mass);
  const index = token ?? ranked[0].index;
  const map = event.maps[index] || [];
  const [rows, cols] = event.grid;
  let peak = 0,
    peakAt = 0,
    sum = 0;
  map.forEach((v, i) => {
    sum += v;
    if (v > peak) {
      peak = v;
      peakAt = i;
    }
  });
  const mean = map.length ? sum / map.length : 0;
  // Share of regions where this token gets at least half its peak weight.
  const spread = map.length
    ? map.filter((v) => v >= peak / 2).length / map.length
    : 0;
  const name = labels?.[index]
    ? `\u201c${tokenText(labels[index])}\u201d`
    : `token ${index}`;
  const rank = ranked.findIndex((r) => r.index === index) + 1;
  const row = Math.floor(peakAt / cols),
    col = peakAt % cols;
  const vertical =
    row < rows / 3 ? "top" : row >= (2 * rows) / 3 ? "bottom" : "middle";
  const horizontal =
    col < cols / 3 ? "left" : col >= (2 * cols) / 3 ? "right" : "centre";
  const where =
    vertical === "middle" && horizontal === "centre"
      ? "centre"
      : `${vertical} ${horizontal}`;
  const text =
    `${name} is read most at the ${where} (${pct(peak)} of that region's attention) and ` +
    (spread >= 0.5
      ? `broadly across ${pct(spread)} of the image`
      : `in a focused area covering ${pct(spread)} of the image`) +
    `; it ranks ${rank} of ${mass.length} tokens by mean attention (${pct(mean)}).`;
  const textShare =
    Array.isArray(event.text_mass) && event.text_mass.length
      ? event.text_mass.reduce((a, b) => a + b, 0) / event.text_mass.length
      : null;
  return {
    ranked,
    peakMass,
    index,
    peak,
    peakCell: [row, col],
    mean,
    spread,
    where,
    text,
    textShare,
  };
}
