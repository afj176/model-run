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

export const layerOf = (module) => {
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

/** Percent with two decimals below 1%, so small but real effects do not read as zero. */
export function pctFine(v) {
  const x = finite(v) ? v * 100 : 0;
  return `${Math.abs(x) < 1 && x !== 0 ? x.toFixed(2) : Math.round(x)}%`;
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

/** Special tokens that only fill the text sequence to a fixed length. */
export const isPaddingLabel = (label) =>
  typeof label === "string" && /^<\|?(endoftext|pad)\|?>$/i.test(label.trim());

/** Ranking and narrative for one cross-attention event; ``token`` focuses one map. */
export function crossAttentionStory(event, labels, token = null, frame = 0) {
  const mass = event?.token_mass;
  if (!Array.isArray(mass) || !mass.length || !Array.isArray(event.maps))
    return null;
  const padding = (i) => isPaddingLabel(labels?.[i]);
  const all = mass
    .map((m, index) => ({ index, mass: m }))
    .sort((a, b) => b.mass - a.mass);
  // Padding is reported as a share, not ranked against the words of the prompt.
  const words = all.filter((r) => !padding(r.index));
  const ranked = words.length ? words : all;
  const peakMass = Math.max(1e-9, ranked[0].mass);
  const total = mass.reduce((a, b) => a + b, 0);
  const paddingShare =
    total > 0
      ? mass.reduce((a, m, i) => a + (padding(i) ? m : 0), 0) / total
      : 0;
  const index = token ?? ranked[0].index;
  const [rows, cols] = event.grid;
  const frames = event.frames ?? 1;
  const whole = event.maps[index] || [];
  const map =
    frames > 1
      ? whole.slice(frame * rows * cols, (frame + 1) * rows * cols)
      : whole;
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
  const of = frames > 1 ? " of this frame" : " of the image";
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
    `${name} is read most at the ${where} (${pctFine(peak)} of that region's attention) and ` +
    (spread >= 0.5
      ? `broadly across ${pct(spread)}${of}`
      : `in a focused area covering ${pct(spread)}${of}`) +
    (rank > 0
      ? `; it ranks ${rank} of ${ranked.length} tokens by mean attention (${pctFine(mean)}).`
      : ".") +
    (paddingShare >= 0.05
      ? ` ${pct(paddingShare)} of this block's attention to text lands on padding positions, not on words of the prompt.`
      : "");
  const textShare =
    Array.isArray(event.text_mass) && event.text_mass.length
      ? event.text_mass.reduce((a, b) => a + b, 0) / event.text_mass.length
      : null;
  return {
    ranked,
    paddingShare,
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

/** Compact slab label: distinct per block family, steps as S<n>. */
export function railLabel(path, index) {
  const step = /^Step (\d+)$/.exec(path);
  if (step) return `S${step[1]}`;
  const m =
    /(?:^|\.)([a-z_]+)\.(\d+)$/.exec(path) || /(?:^|\.)(mid_block)$/.exec(path);
  if (!m) return String(index + 1);
  const family = m[1],
    n = m[2] ?? "";
  const prefix =
    {
      layers: "L",
      double_blocks: "DB",
      single_blocks: "SB",
      joint_blocks: "J",
      transformer_blocks: "T",
      down_blocks: "Dn",
      up_blocks: "Up",
      mid_block: "M",
      blocks: "B",
      h: "L",
    }[family] || family.replace(/_blocks?$/, "").slice(0, 2);
  return `${prefix}${n}`;
}

/** The raw prompt text as the tokenizer saw it (decoded tokens joined). */
export function promptFromTokens(tokens) {
  if (!Array.isArray(tokens) || !tokens.length) return "";
  return tokens.map((t) => t.text ?? "").join("");
}

/** Align two token lists by common prefix and suffix; the middle is what changed. */
export function tokenDiff(a = [], b = []) {
  const idA = a.map((t) => t.id ?? t.text),
    idB = b.map((t) => t.id ?? t.text);
  let prefix = 0;
  while (
    prefix < idA.length &&
    prefix < idB.length &&
    idA[prefix] === idB[prefix]
  )
    prefix++;
  let suffix = 0;
  while (
    suffix < idA.length - prefix &&
    suffix < idB.length - prefix &&
    idA[idA.length - 1 - suffix] === idB[idB.length - 1 - suffix]
  )
    suffix++;
  return {
    prefix,
    suffix,
    changedA: [prefix, idA.length - suffix],
    changedB: [prefix, idB.length - suffix],
    same: prefix === idA.length && idA.length === idB.length,
  };
}

/** Compare two matrices over their shared top-left block; reports truncation. */
export function compareOverlap(a, b) {
  if (!a?.length || !b?.length) return null;
  const rows = Math.min(a.length, b.length);
  const cols = Math.min(a[0].length, b[0].length);
  const result = compareMatrices(
    a.slice(0, rows).map((r) => r.slice(0, cols)),
    b.slice(0, rows).map((r) => r.slice(0, cols)),
  );
  if (!result) return null;
  return {
    ...result,
    rows,
    cols,
    truncated:
      rows !== a.length ||
      rows !== b.length ||
      cols !== a[0].length ||
      cols !== b[0].length,
  };
}

/** Which heads of the shared layer moved most between two attention captures. */
export function headReactions(traceA, traceB) {
  if (!Array.isArray(traceA?.heads) || !Array.isArray(traceB?.heads)) return [];
  const count = Math.min(traceA.heads.length, traceB.heads.length);
  const list = [];
  for (let h = 0; h < count; h++) {
    const diff = compareOverlap(traceA.heads[h], traceB.heads[h]);
    if (!diff) continue;
    const before = attentionHeadProfile(traceA.heads[h]);
    const after = attentionHeadProfile(traceB.heads[h]);
    list.push({
      head: h,
      meanAbs: diff.meanAbs,
      maxDiff: diff.maxDiff,
      before: before?.label,
      after: after?.label,
      relabelled: before?.label !== after?.label,
      entropyDelta: (after?.entropyBits ?? 0) - (before?.entropyBits ?? 0),
    });
  }
  return list.sort((x, y) => y.meanAbs - x.meanAbs);
}

/** Per-layer reactions from head-averaged layer summaries of two attention captures. */
export function layerReactions(traceA, traceB) {
  if (!Array.isArray(traceA?.layers) || !Array.isArray(traceB?.layers))
    return [];
  const byIndex = new Map(traceB.layers.map((l) => [l.index, l]));
  return traceA.layers
    .filter((l) => byIndex.has(l.index))
    .map((l) => {
      const o = byIndex.get(l.index);
      return {
        layer: l.index,
        entropyDelta: o.entropy_bits - l.entropy_bits,
        sinkDelta: o.sink_mass - l.sink_mass,
        previousDelta: o.previous_token_mass - l.previous_token_mass,
      };
    })
    .sort((x, y) => Math.abs(y.entropyDelta) - Math.abs(x.entropyDelta));
}

/** Where two logit-lens tracks stop agreeing on the top token. */
export function lensDivergence(lensA, lensB) {
  const a = lensA?.layers,
    b = lensB?.layers;
  if (!Array.isArray(a) || !Array.isArray(b) || !a.length || !b.length)
    return null;
  const count = Math.min(a.length, b.length);
  const agree = [];
  let first = null;
  for (let i = 0; i < count; i++) {
    const same = a[i].top_tokens[0]?.token_id === b[i].top_tokens[0]?.token_id;
    agree.push(same);
    if (!same && first === null) first = i;
  }
  let last = null;
  for (let i = count - 1; i >= 0; i--)
    if (!agree[i]) {
      last = i;
      break;
    }
  return {
    layers: count,
    agree,
    firstDisagreement: first,
    lastDisagreement: last,
    firstLabel: first === null ? null : a[first].label,
    agreeCount: agree.filter(Boolean).length,
  };
}

/** Narrative for a prompt-to-prompt comparison; every figure is read from the two traces. */
export function promptCompareStory(traceA, traceB) {
  if (!traceA || !traceB) return null;
  const parts = [];
  const diff = tokenDiff(traceA.tokens, traceB.tokens);
  if (traceA.tokens && traceB.tokens) {
    const removed = traceA.tokens
      .slice(...diff.changedA)
      .map((t) => t.text)
      .join("");
    const added = traceB.tokens
      .slice(...diff.changedB)
      .map((t) => t.text)
      .join("");
    if (diff.same) parts.push("Both prompts tokenise identically.");
    else if (removed && added)
      parts.push(
        `The prompt changed “${tokenText(removed)}” to “${tokenText(added)}” after ${diff.prefix} shared token${diff.prefix === 1 ? "" : "s"}.`,
      );
    else if (added)
      parts.push(
        `The prompt gained “${tokenText(added)}” after ${diff.prefix} shared tokens.`,
      );
    else if (removed)
      parts.push(
        `The prompt lost “${tokenText(removed)}” after ${diff.prefix} shared tokens.`,
      );
  }
  const topA = traceA.output?.top_tokens?.[0],
    topB = traceB.output?.top_tokens?.[0];
  if (topA && topB) {
    const under = traceB.output.top_tokens.find(
      (t) => t.token_id === topA.token_id,
    );
    if (topA.token_id === topB.token_id)
      parts.push(
        `The answer holds: “${tokenText(topA.text)}” goes from ${pct(topA.probability)} to ${pct(topB.probability)}.`,
      );
    else
      parts.push(
        `The answer changes from “${tokenText(topA.text)}” (${pct(topA.probability)}) to “${tokenText(topB.text)}” (${pct(topB.probability)})${under ? `; the old answer keeps ${pct(under.probability)}` : "; the old answer leaves the top ten"}.`,
      );
  }
  const lens = lensDivergence(traceA.logit_lens, traceB.logit_lens);
  if (lens)
    parts.push(
      lens.firstDisagreement === null
        ? `The logit lens agrees on the top token at every one of ${lens.layers} layers.`
        : `The logit lens first disagrees at ${lens.firstLabel} and agrees at ${lens.agreeCount} of ${lens.layers} layers.`,
    );
  const layers = layerReactions(traceA, traceB);
  if (layers.length) {
    const l = layers[0];
    parts.push(
      `Layer ${l.layer}'s attention reacted most: entropy ${l.entropyDelta >= 0 ? "up" : "down"} ${Math.abs(l.entropyDelta).toFixed(2)} bits, sink mass ${l.sinkDelta >= 0 ? "+" : ""}${pct(l.sinkDelta)}.`,
    );
  }
  const heads = headReactions(traceA, traceB);
  if (heads.length) {
    const h = heads[0];
    parts.push(
      `Head ${h.head} moved most (mean |Δ| ${h.meanAbs.toFixed(3)}${h.relabelled ? `, ${h.before} → ${h.after}` : ""}).`,
    );
  }
  return { text: parts.join(" "), diff, lens, layers, heads };
}

/** Narrative and ranking for an activation-patching sweep attached to the target trace. */
export function patchStory(block) {
  const layers = block?.layers;
  if (!Array.isArray(layers) || !layers.length) return null;
  const source = block.source_run.top_tokens[0];
  const target = block.target_run.top_tokens[0];
  const ranked = [...layers].sort((a, b) => b.recovery - a.recovery);
  const best = ranked[0];
  const flips = layers.filter((l) => l.top_is_source).map((l) => l.layer);
  const sameAnswer = source.token_id === target.token_id;
  const name = (t) => `\u201c${tokenText(t.text)}\u201d`;
  let text;
  if (sameAnswer)
    text = `Both prompts already answer ${name(source)}; patching moves its probability from ${pct(block.target_run.source_answer_probability)} by at most ${pct(Math.max(...layers.map((l) => l.source_answer_probability)) - block.target_run.source_answer_probability)} (layer ${best.layer}).`;
  else if (flips.length)
    text = `Copying A's residual stream into B at the ${block.positions === "all" ? "every position" : "last position"} flips the answer to ${name(source)} at ${flips.length === 1 ? `layer ${flips[0]}` : `${flips.length} layers (first ${Math.min(...flips)}, last ${Math.max(...flips)})`}; layer ${best.layer} recovers ${pct(best.recovery)} of the gap, taking ${name(source)} from ${pct(block.target_run.source_answer_probability)} to ${pct(best.source_answer_probability)}.`;
  else
    text = `No single layer flips B's answer to ${name(source)}; the strongest is layer ${best.layer}, which recovers ${pct(best.recovery)} of the gap (${pct(block.target_run.source_answer_probability)} \u2192 ${pct(best.source_answer_probability)}).`;
  const peak = Math.max(1e-9, ...layers.map((l) => Math.abs(l.recovery)));
  return { text, ranked, best, flips, peak, sameAnswer, source, target };
}

/** Per-step divergence between two diffusion runs over their retained latent prefixes. */
export function stepDeltas(eventsA = [], eventsB = []) {
  // Diffusers runs line up by step, ComfyUI runs by module and denoiser call.
  const keyOf = (e) =>
    e.step !== undefined
      ? `step:${e.step}`
      : e.denoiser_call !== undefined
        ? `${e.module}|${e.denoiser_call}`
        : null;
  const firstModule = (eventsA || []).find(
    (e) => e.denoiser_call !== undefined,
  )?.module;
  const byStep = new Map();
  for (const e of eventsB || []) if (keyOf(e)) byStep.set(keyOf(e), e);
  const out = [];
  for (const a of eventsA || []) {
    if (!keyOf(a) || (a.step === undefined && a.module !== firstModule))
      continue;
    const b = byStep.get(keyOf(a));
    if (!b || !Array.isArray(a.values) || !Array.isArray(b.values)) continue;
    const n = Math.min(a.values.length, b.values.length);
    let sum = 0,
      sumA = 0,
      count = 0;
    for (let i = 0; i < n; i++) {
      const x = a.values[i],
        y = b.values[i];
      if (!finite(x) || !finite(y)) continue;
      sum += (x - y) * (x - y);
      sumA += x * x;
      count++;
    }
    if (!count) continue;
    const rms = Math.sqrt(sum / count),
      rmsA = Math.sqrt(sumA / count);
    out.push({
      step: a.step ?? a.denoiser_call,
      module: a.step === undefined ? a.module : null,
      rms,
      rmsA,
      relative: rmsA > 0 ? rms / rmsA : 0,
    });
  }
  return out;
}

/** How each text token's mean attention moved between two cross-attention sections. */
export function tokenMassShift(sectionA, sectionB) {
  const a = sectionA?.events?.[0],
    b = sectionB?.events?.[0];
  if (!a?.token_mass || !b?.token_mass) return null;
  const labels = sectionA.text_labels || sectionB.text_labels || null;
  const n = Math.min(a.token_mass.length, b.token_mass.length);
  const ranked = [];
  for (let i = 0; i < n; i++) {
    if (isPaddingLabel(labels?.[i])) continue;
    ranked.push({
      index: i,
      label: labels?.[i] ?? null,
      before: a.token_mass[i],
      after: b.token_mass[i],
      delta: b.token_mass[i] - a.token_mass[i],
    });
  }
  ranked.sort((x, y) => Math.abs(y.delta) - Math.abs(x.delta));
  return {
    ranked,
    peak: Math.max(1e-9, ...ranked.map((r) => Math.abs(r.delta))),
  };
}

/** Narrative for two diffusion runs, usually a token-ablation pair. */
export function diffusionCompareStory({ ablation, pixels, deltas, shift }) {
  const parts = [];
  if (ablation?.labels?.length)
    parts.push(
      `B ran with ${ablation.labels.map((l) => `\u201c${tokenText(l)}\u201d`).join(", ")} masked out of every cross-attention call (${ablation.applied_attention_calls} attention calls recorded).`,
    );
  if (pixels) {
    const grid = pixels.grid || [];
    let peak = 0;
    for (let i = 1; i < grid.length; i++) if (grid[i] > grid[peak]) peak = i;
    const row = Math.floor(peak / 4),
      col = peak % 4;
    const where = `${row < 1 ? "top" : row > 2 ? "bottom" : "middle"} ${col < 1 ? "left" : col > 2 ? "right" : "centre"}`;
    parts.push(
      pixels.mean < 0.001
        ? "The images are practically identical (under 0.1% mean pixel difference)."
        : `The images differ by ${pctFine(pixels.mean)} of full scale on average, most at the ${where} (${pctFine(grid[peak])}).`,
    );
  }
  if (deltas?.length) {
    const first = deltas.find((d) => d.relative >= 0.05);
    const last = deltas.at(-1);
    const subject = last.module === null ? "Latents" : "The observed output";
    const unit = last.module === null ? "step" : "call";
    parts.push(
      first
        ? `${subject} ${last.module === null ? "diverge" : "diverges"} from ${unit} ${first.step} and ${last.module === null ? "end" : "ends"} ${pctFine(last.relative)} apart relative to A.`
        : `${subject} ${last.module === null ? "stay" : "stays"} within ${pctFine(Math.max(...deltas.map((d) => d.relative)))} of A throughout.`,
    );
  }
  if (shift?.ranked?.length) {
    const s = shift.ranked[0];
    parts.push(
      `Attention moved most for ${s.label ? `\u201c${tokenText(s.label)}\u201d` : `token ${s.index}`} (${s.delta >= 0 ? "+" : ""}${pctFine(s.delta)} mean per region).`,
    );
  }
  return parts.length ? { text: parts.join(" ") } : null;
}

/** The structure node that is decoder layer ``layer``, across the block families the app knows. */
export function layerNodePath(nodes, layer) {
  if (!Array.isArray(nodes) || !Number.isInteger(layer)) return null;
  const pattern = new RegExp(
    `(?:^|\\.)(?:layers|blocks|transformer_blocks|double_blocks|single_blocks|joint_blocks|h)\\.${layer}$`,
  );
  return nodes.find((n) => pattern.test(n.path))?.path ?? null;
}

/** Narrative for a greedy response: how sure the model was and where its tokens settled. */
export function responseStory(trace) {
  const steps = trace?.steps;
  if (!Array.isArray(steps) || !steps.length) return null;
  const probabilities = steps.map((s) => s.probability);
  const meanProbability =
    probabilities.reduce((a, b) => a + b, 0) / probabilities.length;
  let most = steps[0],
    least = steps[0];
  for (const s of steps) {
    if (s.probability > most.probability) most = s;
    if (s.probability < least.probability) least = s;
  }
  const settled = steps
    .map((s) => s.lens?.settled_layer)
    .filter((v) => Number.isInteger(v));
  const layerCount = steps[0].lens?.chosen_probability_by_layer?.length ?? 0;
  const parts = [];
  const text = (
    trace.response ?? steps.map((s) => s.text ?? "").join("")
  ).trim();
  parts.push(
    `The model wrote \u201c${tokenText(text.length > 160 ? `${text.slice(0, 157)}\u2026` : text)}\u201d.`,
  );
  parts.push(
    `It was surest of \u201c${tokenText(most.text)}\u201d (${pct(most.probability)}) and least sure of \u201c${tokenText(least.text)}\u201d (${pct(least.probability)}, ${least.entropy_bits.toFixed(1)} bits).`,
  );
  if (settled.length && layerCount > 1) {
    const sorted = [...settled].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const late = settled.filter((v) => v >= layerCount - 1).length;
    parts.push(
      `${settled.length === steps.length ? "Every token" : `${settled.length} of ${steps.length} tokens`} settled by the output; the typical token was decided at ${median === 0 ? "the embeddings" : `layer ${median - 1} of ${layerCount - 1}`}${late ? `, and ${late} only at the final layer` : ""}.`,
    );
  }
  const segments = responseSegments(steps);
  if (segments.thinking) {
    const [a, b] = segments.thinking;
    const think = steps.slice(a, b);
    const answer = segments.closed ? steps.slice(segments.answerStart) : [];
    const mean = (list) =>
      list.length
        ? list.reduce((x, s) => x + s.probability, 0) / list.length
        : null;
    parts.push(
      segments.closed
        ? `It thought for ${think.length} tokens before answering${answer.length ? ` with ${answer.length}` : ""}; thinking tokens averaged ${pct(mean(think))} confidence${answer.length ? ` and the answer ${pct(mean(answer))}` : ""}.`
        : `It was still thinking when the length limit hit (${think.length} thinking tokens); raise the response length to see the answer.`,
    );
  }
  // Close calls: where the runner-up nearly won.
  const closeCalls = steps
    .map((s) => ({
      index: s.index,
      text: s.text,
      probability: s.probability,
      runnerUp: s.top_tokens?.[1] ?? null,
    }))
    .filter((c) => c.runnerUp && c.probability - c.runnerUp.probability < 0.1)
    .sort(
      (x, y) =>
        x.probability -
        x.runnerUp.probability -
        (y.probability - y.runnerUp.probability),
    )
    .slice(0, 4);
  if (trace.stopped === "eos") parts.push("It chose to stop.");
  else if (trace.stopped === "context_limit")
    parts.push("It hit the context limit.");
  return {
    text: parts.join(" "),
    meanProbability,
    mostConfident: most,
    leastConfident: least,
    settledCount: settled.length,
    closeCalls,
    segments,
  };
}

/** Thinking span marked by the model's own <think> … </think> tokens, as step indices. */
export function responseSegments(steps = []) {
  const isTag = (s, tag) => (s?.text ?? "").trim() === tag;
  const open = steps.findIndex((s) => isTag(s, "<think>"));
  if (open < 0) return { thinking: null, closed: true };
  const close = steps.findIndex((s, i) => i > open && isTag(s, "</think>"));
  return {
    thinking: [open + 1, close < 0 ? steps.length : close],
    closed: close >= 0,
    answerStart: close < 0 ? null : close + 1,
  };
}
