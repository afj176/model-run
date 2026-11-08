import { useMemo } from "react";
import { layerSummary, layerTokenGrid } from "./weatherData";
import { tokenLabel } from "./insights";

/**
 * One bar per layer: the mean size of its output over all tokens, split into what attention
 * and the MLP added, with a whisker for the largest token. Reads which layers do the work
 * without opening any of them; its height is the model's, not the prompt's.
 */
export default function ResidualLadder({
  events = [],
  tokens = [],
  layer = null,
  onPick,
}) {
  const grid = useMemo(() => layerTokenGrid(events), [events]);
  const output = useMemo(() => layerSummary(grid, "output"), [grid]);
  const attention = useMemo(() => layerSummary(grid, "attention"), [grid]);
  const mlp = useMemo(() => layerSummary(grid, "mlp"), [grid]);
  const L = grid.layers.length;
  if (!L || !output.some((s) => s.mean !== null)) return null;
  const hasParts =
    attention.some((s) => s.mean !== null) || mlp.some((s) => s.mean !== null);
  const peak = Math.max(1e-9, ...output.map((s) => s.peak ?? 0));
  const meanPeak = Math.max(1e-9, ...output.map((s) => s.mean ?? 0));
  const W = 640,
    PAD_L = 36,
    PAD_R = 8,
    TOP = 8,
    H = 110,
    AXIS = 18;
  const bw = (W - PAD_L - PAD_R) / L;
  // Bars are scaled to the largest mean; whiskers beyond the top are clipped and marked.
  const y = (v) =>
    TOP + H - (Math.min(v, meanPeak * 1.6) / (meanPeak * 1.6)) * H;
  const massive = output
    .map((s, i) => ({ ...s, i }))
    .filter((s) => s.spread !== null && s.spread >= 8)
    .sort((a, b) => b.spread - a.spread)[0];
  const tickEvery = Math.max(1, Math.ceil(L / 10));
  return (
    <section className="ladder" aria-label="Residual ladder">
      <h3>
        How the stream grows{" "}
        <small>
          mean size of each layer's output over all tokens
          {hasParts ? ", split by what attention and the MLP added" : ""}
        </small>
      </h3>
      <svg
        viewBox={`0 0 ${W} ${TOP + H + AXIS}`}
        className="ladder-svg"
        role="img"
        aria-label="Residual stream size per layer"
      >
        {output.map((s, i) => {
          if (s.mean === null) return null;
          const x = PAD_L + i * bw;
          const a = attention[i]?.mean ?? null;
          const m = mlp[i]?.mean ?? null;
          const total = s.mean;
          // Parts are shown as their share of the output's size, so the bar still reads as one layer.
          const share =
            a !== null && m !== null && a + m > 0
              ? a / (a + m)
              : a !== null
                ? 1
                : 0;
          const aH = hasParts ? (TOP + H - y(total)) * share : 0;
          const picked = grid.layers[i].layer === layer;
          return (
            <g
              key={s.layer}
              className={picked ? "picked" : ""}
              onClick={() => onPick?.(s.layer)}
              style={{ cursor: onPick ? "pointer" : "default" }}
            >
              <title>
                L{s.layer}: output {total.toPrecision(3)} RMS
                {a !== null ? ` · attention added ${a.toPrecision(3)}` : ""}
                {m !== null ? ` · MLP added ${m.toPrecision(3)}` : ""}
                {s.peak !== null
                  ? ` · largest token ${s.peak.toPrecision(3)} (${Math.round(s.spread)}× the median)`
                  : ""}
              </title>
              <rect
                x={x + 1.5}
                y={y(total)}
                width={Math.max(1, bw - 3)}
                height={TOP + H - y(total) - aH}
                className="ladder-mlp"
              />
              {aH > 0 && (
                <rect
                  x={x + 1.5}
                  y={TOP + H - aH}
                  width={Math.max(1, bw - 3)}
                  height={aH}
                  className="ladder-attention"
                />
              )}
              {s.peak !== null && s.peak > total && (
                <line
                  x1={x + bw / 2}
                  x2={x + bw / 2}
                  y1={y(total)}
                  y2={y(s.peak)}
                  className={`ladder-whisker ${s.spread >= 8 ? "massive" : ""}`}
                />
              )}
              {s.peak !== null && s.peak > meanPeak * 1.6 && (
                <text
                  x={x + bw / 2}
                  y={TOP - 1}
                  textAnchor="middle"
                  className="ladder-clip"
                >
                  ▲
                </text>
              )}
              {picked && (
                <rect
                  x={x}
                  y={TOP}
                  width={bw}
                  height={H}
                  className="ladder-pick"
                />
              )}
              {(i % tickEvery === 0 || i === L - 1) && (
                <text
                  x={x + bw / 2}
                  y={TOP + H + 13}
                  textAnchor="middle"
                  className="ladder-tick"
                >
                  L{s.layer}
                </text>
              )}
            </g>
          );
        })}
        <text
          x={PAD_L - 6}
          y={TOP + H}
          textAnchor="end"
          className="ladder-tick"
        >
          0
        </text>
        <text
          x={PAD_L - 6}
          y={TOP + 10}
          textAnchor="end"
          className="ladder-tick"
        >
          {(meanPeak * 1.6).toPrecision(2)}
        </text>
      </svg>
      <p className="micro">
        {hasParts ? (
          <>
            <i className="ladder-key mlp" /> MLP added ·{" "}
            <i className="ladder-key attention" /> attention added ·{" "}
          </>
        ) : null}
        whiskers reach the largest token;{" "}
        {massive
          ? `at L${massive.layer} that is token ${massive.peakToken} “${tokenLabel(tokens[massive.peakToken], massive.peakToken)}”, ${Math.round(massive.spread)}× the median: a massive activation (▲ where it runs off the chart).`
          : "no token stands far above its layer's median."}
        {onPick ? " Click a bar to open that layer." : ""}
      </p>
    </section>
  );
}
