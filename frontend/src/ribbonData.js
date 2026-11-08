// The prompt ribbon: one measure per prompt token, a reading window over them, and the
// sentence segments a long prompt folds into. Pure functions; PromptRibbon.jsx draws them.

const finite = (v) => typeof v === "number" && Number.isFinite(v);

/** The default reading window: about this many tokens are drawn in full. */
export const WINDOW_SIZE = 24;

/** A window [start, start + size) kept inside n tokens; a short prompt is one window. */
export function clampWindow(start, size, n) {
  const length = Math.max(1, Math.min(size, n || 1));
  const first = Math.max(0, Math.min(n - length, Math.round(start) || 0));
  return { start: first, size: length };
}

/** The window that shows token `index`, moving as little as possible from `current`. */
export function windowShowing(index, current, n, size = WINDOW_SIZE) {
  const length = Math.min(size, n);
  if (current && index >= current.start && index < current.start + current.size)
    return clampWindow(current.start, current.size, n);
  return clampWindow(index - Math.floor(length / 2), length, n);
}

/** Each token's size at one layer: its RMS over all features, from a layer event's token grid. */
export function tokenNorms(event) {
  const rms = event?.token_rms ?? event?.slices?.token_rms;
  return Array.isArray(rms) ? rms.map((v) => (finite(v) ? v : null)) : null;
}

/** How much the written answer looked at each prompt token: attention from every
 * generated step back onto the prompt, summed over steps (each step's row sums to 1). */
export function answerAttention(steps, promptLength) {
  if (!Array.isArray(steps) || !steps.length || !promptLength) return null;
  const out = new Array(promptLength).fill(0);
  let counted = 0;
  for (const step of steps) {
    const row = step?.attention;
    if (!Array.isArray(row)) continue;
    counted++;
    for (let i = 0; i < promptLength && i < row.length; i++)
      if (finite(row[i])) out[i] += row[i];
  }
  return counted ? out.map((v) => v / counted) : null;
}

/** How much attention each prompt token received inside the prompt: the column mean of one
 * head's matrix over the queries that can see it. A pooled matrix spreads each block's value
 * over the tokens the block covers (`pooled.ranges`), so the result is always per token. */
export function attentionReceived(matrix, pooled, n) {
  if (!Array.isArray(matrix) || !matrix.length) return null;
  const side = matrix.length;
  const column = new Array(side).fill(0);
  for (let c = 0; c < side; c++) {
    let total = 0,
      count = 0;
    for (let r = c; r < side; r++) {
      const v = matrix[r]?.[c];
      if (finite(v)) {
        total += v;
        count++;
      }
    }
    column[c] = count ? total / count : null;
  }
  const ranges = pooled?.ranges;
  if (!Array.isArray(ranges) || ranges.length !== side)
    return column.slice(0, n || side);
  const out = new Array(n || pooled.token_count || 0).fill(null);
  ranges.forEach(([a, b], c) => {
    for (let i = a; i < b && i < out.length; i++) out[i] = column[c];
  });
  return out;
}

/** Values scaled to 0…1 by rank-insensitive robust range: the 2nd to 98th percentile. */
export function normalize(values) {
  if (!values) return null;
  const sorted = values.filter(finite).sort((a, b) => a - b);
  if (!sorted.length) return values.map(() => null);
  const at = (q) =>
    sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];
  const low = at(0.02);
  const high = Math.max(at(0.98), low + 1e-9);
  return values.map((v) =>
    finite(v) ? Math.max(0, Math.min(1, (v - low) / (high - low))) : null,
  );
}

const SENTENCE_END = /[.!?。！？]["'”’)\]]*$/;
const CLAUSE_END = /[;:—]$|\n$/;

/** Sentences of the prompt as token ranges, from the decoded text: a long prompt folds into these.
 * A range ends at a token that ends a sentence, or a clause when a sentence runs past `most` tokens. */
export function sentenceSegments(tokens, most = 48) {
  const list = tokens || [];
  const out = [];
  let start = 0;
  for (let i = 0; i < list.length; i++) {
    const text = (list[i]?.text ?? "").trimEnd();
    const length = i - start + 1;
    const ends =
      SENTENCE_END.test(text) ||
      (length >= most && CLAUSE_END.test(text)) ||
      length >= most * 2 ||
      i === list.length - 1;
    if (!ends) continue;
    out.push({ start, end: i + 1 });
    start = i + 1;
  }
  return out.map((s, index) => ({
    ...s,
    index,
    text: list
      .slice(s.start, s.end)
      .map((t) => t?.text ?? "")
      .join("")
      .trim(),
  }));
}

/** Each segment's share of a per-token measure (its sum over the measure's total). */
export function segmentShares(segments, values) {
  if (!segments?.length || !values) return null;
  const sums = segments.map((s) => {
    let total = 0;
    for (let i = s.start; i < s.end; i++)
      if (finite(values[i])) total += values[i];
    return total;
  });
  const all = sums.reduce((a, b) => a + b, 0);
  return sums.map((v) => (all > 0 ? v / all : 0));
}

/** The segment holding a token. */
export const segmentAt = (segments, index) =>
  segments?.find((s) => index >= s.start && index < s.end) ?? null;
