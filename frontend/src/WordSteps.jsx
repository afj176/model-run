import { useMemo } from "react";
import { tokenText } from "./insights";
import { heat } from "./PromptRibbon";

/** Padding runs (repeated special tokens) stay out of the grid. */
const isPad = (labels, i) =>
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

/**
 * Prompt words down, denoiser calls across: when each word pulled on the image. Nouns light
 * up early, style words later, filler never. A click opens that call's map for that word.
 */
export default function WordSteps({
  events = [],
  labels = null,
  picked = 0,
  token = null,
  onPick,
}) {
  const grid = useMemo(() => wordStepGrid(events, labels), [events, labels]);
  if (!grid || grid.calls.length < 2) return null;
  const C = grid.calls.length;
  const label = (t) => (labels?.[t] ? tokenText(labels[t]) : `#${t}`);
  // Which word moved most across the calls: the largest rise or fall of its share.
  let mover = null;
  grid.rows.forEach((t, r) => {
    const series = grid.shares.map((col) => col[r]);
    const swing = Math.max(...series) - Math.min(...series);
    if (!mover || swing > mover.swing)
      mover = { t, swing, peakAt: series.indexOf(Math.max(...series)) };
  });
  const every = Math.max(1, Math.ceil(C / 12));
  return (
    <section className="word-steps" aria-label="Word by denoiser call">
      <h3>
        When each word shaped the output{" "}
        <small>
          {grid.rows.length} words × {C} calls · share of the text attention
        </small>
      </h3>
      <div
        className="word-steps-grid"
        style={{
          gridTemplateColumns: `minmax(60px, 110px) repeat(${C}, minmax(0, 1fr))`,
        }}
        role="grid"
      >
        {grid.rows.map((t, r) => (
          <div key={t} role="row" style={{ display: "contents" }}>
            <span
              className="word-steps-label"
              role="rowheader"
              title={label(t)}
            >
              {label(t)}
            </span>
            {grid.cells.map((col, i) => (
              <button
                key={i}
                role="gridcell"
                className={`word-steps-cell ${i === picked && t === token ? "picked" : i === picked ? "column" : ""}`}
                style={{ background: heat(col[r]) }}
                title={`call ${grid.calls[i]} · ${label(t)} · ${(grid.shares[i][r] * 100).toFixed(1)}% of the text attention`}
                onClick={() => onPick?.(i, t)}
              />
            ))}
          </div>
        ))}
      </div>
      <div
        className="word-steps-axis"
        style={{
          gridTemplateColumns: `minmax(60px, 110px) repeat(${C}, minmax(0, 1fr))`,
        }}
      >
        <span />
        {grid.calls.map((c, i) => (
          <span key={i}>{i % every === 0 || i === C - 1 ? c : ""}</span>
        ))}
      </div>
      <p className="micro">
        Each column is one recorded denoiser call; each cell is that word's
        share of the attention the image paid to the text, over the peak.{" "}
        {mover && mover.swing > 0.05
          ? `“${label(mover.t)}” moves most: strongest at call ${grid.calls[mover.peakAt]}.`
          : ""}{" "}
        Click a cell to open that call's map for that word.
      </p>
    </section>
  );
}
