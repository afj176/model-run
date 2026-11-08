import { useEffect, useMemo, useRef, useState } from "react";
import { pct, tokenText as show } from "./insights";

const COLOURS = [
  "#9d8ff0",
  "#f0b35a",
  "#f16d83",
  "#6fb3ef",
  "#cbcb5c",
  "#ea86c9",
  "#8fd3c7",
  "#c9bdf0",
];

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

/**
 * The candidates for the next token as lanes across the layers, each as wide as its
 * probability: the answer widens toward the right, the runners-up cross and fade.
 */
export default function LensRiver({ lens, names, picked = null, onPick }) {
  const lanes = useMemo(() => riverLanes(lens, names), [lens, names]);
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
  const layers = lens?.layers || [];
  const L = layers.length;
  if (!L || !lanes.length) return null;
  const LABEL_W = 118,
    PAD_L = 28,
    H = 150,
    TOP = 10,
    AXIS = 20;
  const plotW = Math.max(60, width - LABEL_W - PAD_L);
  const x = (l) => PAD_L + (l / Math.max(1, L - 1)) * plotW;
  const y = (p) => TOP + (1 - p) * H;
  // Each layer's lanes share its height: a lane's width is its share of that layer's listed
  // candidates, so a takeover reads at every depth, even where the raw probabilities are tiny.
  const stacks = Array.from({ length: L }, (_, l) => {
    const total = lanes.reduce((sum, lane) => sum + lane.p[l], 0) || 1;
    let acc = 0;
    return lanes.map((lane) => {
      const bottom = acc;
      acc += lane.p[l] / total;
      return [bottom, acc];
    });
  });
  const area = (k) => {
    let top = "",
      bottom = "";
    for (let l = 0; l < L; l++)
      top += `${l ? "L" : "M"}${x(l)} ${y(stacks[l][k][1])} `;
    for (let l = L - 1; l >= 0; l--)
      bottom += `L${x(l)} ${y(stacks[l][k][0])} `;
    return top + bottom + "Z";
  };
  // Labels at the right edge, pushed apart so none overlap.
  let lastY = Infinity;
  const labels = lanes
    .map((lane, k) => ({
      lane,
      k,
      yy: y((stacks[L - 1][k][0] + stacks[L - 1][k][1]) / 2),
    }))
    .sort((a, b) => b.yy - a.yy)
    .map((e) => {
      const yy = Math.min(e.yy, lastY - 13);
      lastY = yy;
      return { ...e, yy };
    });
  const colour = (lane, k) =>
    lane.final ? "#ff3d8f" : COLOURS[k % COLOURS.length];
  const takeover =
    lens.final_token_id !== undefined
      ? layers.findIndex(
          (layer, l) =>
            layer.top_tokens?.[0]?.token_id === lens.final_token_id &&
            layers
              .slice(l)
              .every(
                (m) => m.top_tokens?.[0]?.token_id === lens.final_token_id,
              ),
        )
      : -1;
  const tickEvery = Math.max(1, Math.ceil(L / Math.max(4, plotW / 60)));
  return (
    <div className="lens-river" ref={box}>
      {width > 0 && (
        <svg
          width={width}
          height={TOP + H + AXIS}
          role="img"
          aria-label="Candidate tokens by layer"
        >
          {lanes.map((lane, k) => (
            <path
              key={lane.token_id}
              d={area(k)}
              fill={colour(lane, k)}
              opacity={lane.final ? 0.95 : 0.62}
            >
              <title>
                “{show(lane.text)}” · peaks at {pct(lane.peak)}
              </title>
            </path>
          ))}
          {layers.map((layer, l) => (
            <g key={layer.index}>
              <rect
                x={x(l) - plotW / (2 * Math.max(1, L - 1))}
                y={TOP}
                width={plotW / Math.max(1, L - 1)}
                height={H}
                fill="transparent"
                style={{ cursor: "pointer" }}
                onClick={() => onPick?.(layer.index)}
              >
                <title>
                  {layer.label}: top “
                  {show(
                    names?.get?.(layer.top_tokens?.[0]?.token_id) ??
                      layer.top_tokens?.[0]?.text,
                  )}
                  ” {pct(layer.top_tokens?.[0]?.probability)}
                </title>
              </rect>
              {(l % tickEvery === 0 || l === L - 1) && (
                <text
                  x={x(l)}
                  y={TOP + H + 14}
                  textAnchor="middle"
                  className="lens-river-tick"
                >
                  {layer.index === 0 ? "emb" : `L${layer.index - 1}`}
                </text>
              )}
            </g>
          ))}
          {picked !== null &&
            layers.some((layer) => layer.index === picked) && (
              <line
                x1={x(layers.findIndex((layer) => layer.index === picked))}
                x2={x(layers.findIndex((layer) => layer.index === picked))}
                y1={TOP}
                y2={TOP + H}
                className="lens-river-pick"
              />
            )}
          {takeover > 0 && (
            <g>
              <line
                x1={x(takeover)}
                x2={x(takeover)}
                y1={TOP - 6}
                y2={TOP + H}
                className="lens-river-takeover"
              />
              <text
                x={x(takeover)}
                y={TOP - 1}
                textAnchor="middle"
                className="lens-river-note"
              >
                leads from {layers[takeover].label}
              </text>
            </g>
          )}
          <text
            x={PAD_L - 6}
            y={y(1) + 4}
            textAnchor="end"
            className="lens-river-tick"
          >
            1.0
          </text>
          <text
            x={PAD_L - 6}
            y={y(0)}
            textAnchor="end"
            className="lens-river-tick"
          >
            0
          </text>
          {labels.map(({ lane, k, yy }) => (
            <text
              key={lane.token_id}
              x={PAD_L + plotW + 8}
              y={yy + 4}
              className={`lens-river-label ${lane.final ? "final" : ""}`}
              fill={colour(lane, k)}
            >
              {show(lane.text)} {pct(lane.p[L - 1])}
            </text>
          ))}
        </svg>
      )}
    </div>
  );
}
