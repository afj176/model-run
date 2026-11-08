import { useEffect, useRef, useState } from "react";
import { tokenLabel } from "./insights";
import { strongestKeys } from "./arcData.js";

/**
 * One query token's attention as a few arcs over the reading window, instead of a whole matrix.
 * Keys outside the window land on a strip of the whole prompt, so nothing is hidden.
 * `ranges` maps pooled key blocks to token ranges; without pooling a key is a token.
 */
export default function AttentionArcs({
  row,
  query,
  ranges = null,
  tokens = [],
  window: win,
  onPick,
  caption = "",
}) {
  const box = useRef(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!box.current) return;
    const watch = new ResizeObserver(([entry]) =>
      setWidth(Math.floor(entry.contentRect.width)),
    );
    watch.observe(box.current);
    return () => watch.disconnect();
  }, []);
  const n = tokens.length;
  if (!row || !n || !win) return null;
  const rangeOf = (key) =>
    ranges ? (ranges[key] ?? [key, key + 1]) : [key, key + 1];
  const keyOfToken = (t) =>
    ranges
      ? Math.max(
          0,
          ranges.findIndex(([a, b]) => t >= a && t < b),
        )
      : t;
  const { start, size } = win;
  const end = Math.min(n, start + size);
  const edges = strongestKeys(row);
  const shown = edges.reduce((s, e) => s + e.weight, 0);
  const queryRange = rangeOf(query);

  const PAD = 10,
    ARC_H = 86,
    CHIP_Y = ARC_H + 6,
    CHIP_H = 26,
    STRIP_Y = CHIP_Y + CHIP_H + 30,
    STRIP_H = 10,
    H = STRIP_Y + STRIP_H + 16;
  const slotW = (width - 2 * PAD) / Math.max(1, end - start);
  const xToken = (t) => PAD + (t - start + 0.5) * slotW;
  const colW = (width - 2 * PAD) / n;
  const xStrip = (t) => PAD + (t + 0.5) * colW;
  const inWindow = (r) => r[0] < end && r[1] > start;
  const centre = (r) =>
    xToken((Math.max(r[0], start) + Math.min(r[1], end) - 1) / 2);
  const qx = inWindow(queryRange) ? centre(queryRange) : null;
  const queryTokens = new Set();
  for (let t = queryRange[0]; t < queryRange[1]; t++) queryTokens.add(t);
  const keyWeight = new Map(edges.map((e) => [e.key, e.weight]));
  const label = (t) => tokenLabel(tokens[t], t);

  return (
    <section className="attention-arcs" aria-label="Attention arcs" ref={box}>
      <header>
        <h3>
          Where “{label(queryRange[0])}” looks{" "}
          <small>
            {edges.length} key{edges.length === 1 ? "" : "s"} carry{" "}
            {Math.round(shown * 100)}% of its attention
            {caption ? ` · ${caption}` : ""}
          </small>
        </h3>
      </header>
      {width > 0 && (
        <svg
          width={width}
          height={H}
          role="img"
          aria-label="Attention from the selected token to the keys it reads"
        >
          {edges.map((e) => {
            const r = rangeOf(e.key);
            const w = 1 + e.weight * 10;
            if (qx !== null && inWindow(r)) {
              const kx = centre(r);
              if (Math.abs(kx - qx) < 1)
                return (
                  <g key={e.key}>
                    <circle
                      cx={qx}
                      cy={ARC_H - 6}
                      r={4 + e.weight * 10}
                      fill="none"
                      stroke="#7fd9c2"
                      strokeWidth={w}
                    />
                    <text
                      x={qx}
                      y={ARC_H - 30 - e.weight * 10}
                      textAnchor="middle"
                      className="arc-weight"
                    >
                      self {Math.round(e.weight * 100)}%
                    </text>
                  </g>
                );
              const rise = Math.min(ARC_H - 14, 18 + Math.abs(kx - qx) * 0.35);
              return (
                <g key={e.key}>
                  <path
                    d={`M${qx} ${ARC_H} C ${qx} ${ARC_H - rise}, ${kx} ${ARC_H - rise}, ${kx} ${ARC_H}`}
                    fill="none"
                    stroke="#7fd9c2"
                    strokeWidth={w}
                    opacity={0.9}
                  />
                  <text
                    x={(qx + kx) / 2}
                    y={ARC_H - rise * 0.75 - 4}
                    textAnchor="middle"
                    className="arc-weight"
                  >
                    {Math.round(e.weight * 100)}%
                  </text>
                </g>
              );
            }
            // Outside the window: down to the strip of the whole prompt.
            const sx = xStrip((r[0] + r[1] - 1) / 2);
            const from = qx ?? xStrip((queryRange[0] + queryRange[1] - 1) / 2);
            const fromY = qx !== null ? CHIP_Y + CHIP_H : STRIP_Y;
            return (
              <g key={e.key}>
                <path
                  d={`M${from} ${fromY} C ${from} ${STRIP_Y - 10}, ${sx} ${CHIP_Y + CHIP_H + 4}, ${sx} ${STRIP_Y}`}
                  fill="none"
                  stroke="#7fd9c2"
                  strokeWidth={w}
                  strokeDasharray="4 3"
                  opacity={0.85}
                />
                <rect
                  x={sx - 1.5}
                  y={STRIP_Y}
                  width={3}
                  height={STRIP_H}
                  fill="#7fd9c2"
                />
              </g>
            );
          })}
          {/* the window's tokens */}
          {Array.from({ length: end - start }, (_, k) => {
            const t = start + k;
            const key = keyOfToken(t);
            const w = keyWeight.get(key) ?? 0;
            const isQuery = queryTokens.has(t);
            const x = PAD + k * slotW;
            return (
              <g
                key={t}
                onClick={() => onPick?.(t)}
                style={{ cursor: "pointer" }}
              >
                <rect
                  x={x + 1}
                  y={CHIP_Y}
                  width={Math.max(1, slotW - 2)}
                  height={CHIP_H}
                  rx={4}
                  fill={
                    isQuery
                      ? "#ff3d8f"
                      : w > 0
                        ? `rgba(127,217,194,${0.25 + 0.75 * Math.min(1, w * 2)})`
                        : "#2b2452"
                  }
                />
                {slotW > 14 && (
                  <text
                    x={x + slotW / 2}
                    y={CHIP_Y + CHIP_H / 2 + 4}
                    textAnchor="middle"
                    className={`arc-token ${isQuery ? "query" : ""}`}
                  >
                    {clip(label(t), slotW)}
                  </text>
                )}
              </g>
            );
          })}
          {/* the strip of the whole prompt, with the window framed */}
          <rect
            x={PAD}
            y={STRIP_Y}
            width={width - 2 * PAD}
            height={STRIP_H}
            rx={2}
            fill="#2b2452"
          />
          <rect
            x={PAD + queryRange[0] * colW}
            y={STRIP_Y - 2}
            width={Math.max(2, (queryRange[1] - queryRange[0]) * colW)}
            height={STRIP_H + 4}
            fill="#ff3d8f"
          />
          <rect
            x={PAD + start * colW}
            y={STRIP_Y - 2}
            width={Math.max(4, (end - start) * colW)}
            height={STRIP_H + 4}
            fill="none"
            stroke="#ff3d8f"
            strokeWidth={1}
          />
        </svg>
      )}
      {edges.some((e) => !inWindow(rangeOf(e.key)) || qx === null) && (
        <p className="arc-outside">
          <span>Outside the window:</span>
          {edges
            .filter((e) => !inWindow(rangeOf(e.key)) || qx === null)
            .map((e) => {
              const r = rangeOf(e.key);
              return (
                <button key={e.key} onClick={() => onPick?.(r[0])}>
                  {Math.round(e.weight * 100)}% → {r[0]} “{label(r[0])}”
                </button>
              );
            })}
        </p>
      )}
      <p className="micro">
        Arcs are the strongest keys of this query, up to 90% of its row; dashed
        arcs reach keys outside the window and land on the strip of the whole
        prompt. Click a token to make it the query.
      </p>
    </section>
  );
}

function clip(text, slotW) {
  const most = Math.max(1, Math.floor((slotW - 6) / 6.5));
  return text.length > most ? `${text.slice(0, Math.max(1, most - 1))}…` : text;
}
