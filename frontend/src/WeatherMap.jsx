import { useEffect, useMemo, useRef, useState } from "react";
import { layerSummary, layerTokenGrid, scaledGrid } from "./weatherData";
import { heat } from "./PromptRibbon";
import { tokenLabel } from "./insights";

/**
 * Layers down, the prompt across: one cell per layer and token holding how large that layer's
 * output (or what its attention or MLP added) was there. Fixed height whatever the prompt length;
 * the ribbon's window is framed, and a click picks that layer and token.
 */
export default function WeatherMap({
  events = [],
  tokens = [],
  window: win,
  layer = null,
  token = null,
  onPick,
}) {
  const grid = useMemo(() => layerTokenGrid(events), [events]);
  const [measure, setMeasure] = useState("output");
  const current = grid.measures.some((m) => m.key === measure)
    ? measure
    : (grid.measures[0]?.key ?? "output");
  const scaled = useMemo(() => scaledGrid(grid, current), [grid, current]);
  const summary = useMemo(() => layerSummary(grid, current), [grid, current]);
  const box = useRef(null);
  const canvas = useRef(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState(null);
  useEffect(() => {
    if (!box.current) return;
    const watch = new ResizeObserver(([entry]) =>
      setWidth(Math.floor(entry.contentRect.width)),
    );
    watch.observe(box.current);
    return () => watch.disconnect();
  }, []);
  const L = grid.layers.length;
  const n = grid.tokens;
  const LABEL_W = 36;
  const rowH = L ? Math.max(5, Math.min(12, Math.floor(200 / L))) : 0;
  const height = L * rowH;
  const plotW = Math.max(40, width - LABEL_W);
  useEffect(() => {
    const cv = canvas.current;
    if (!cv || !width || !L || !n) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(width * dpr);
    cv.height = Math.round(height * dpr);
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const px = Math.max(1, Math.round(plotW * dpr));
    const per = n / px;
    const gap = rowH >= 9 ? 1 : 0;
    // One device pixel per column: each shows the largest token it covers.
    const strip = new ImageData(px, L);
    for (let l = 0; l < L; l++) {
      const row = scaled.rows[l];
      for (let x = 0; x < px; x++) {
        const t0 = Math.floor(x * per);
        const t1 = Math.max(t0 + 1, Math.floor((x + 1) * per));
        let best = null;
        for (let t = t0; t < t1 && t < n; t++) {
          const v = row[t];
          if (v !== null && (best === null || v > best)) best = v;
        }
        const rgb =
          best === null ? [70, 70, 78] : heat(best).match(/\d+/g).map(Number);
        const o = (l * px + x) * 4;
        strip.data[o] = rgb[0];
        strip.data[o + 1] = rgb[1];
        strip.data[o + 2] = rgb[2];
        strip.data[o + 3] = 255;
      }
    }
    const off = document.createElement("canvas");
    off.width = px;
    off.height = L;
    off.getContext("2d").putImageData(strip, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.font = "10px Consolas, 'Cascadia Mono', monospace";
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (let l = 0; l < L; l++) {
      const y = l * rowH;
      ctx.drawImage(off, 0, l, px, 1, LABEL_W, y, plotW, rowH - gap);
      const every = Math.ceil(12 / rowH);
      if (l % every === 0 || l === L - 1 || grid.layers[l].layer === layer) {
        ctx.fillStyle = grid.layers[l].layer === layer ? "#ffd3e6" : "#c9c1e6";
        ctx.fillText(`L${grid.layers[l].layer}`, LABEL_W - 6, y + rowH / 2);
      }
    }
    const colW = plotW / n;
    if (win && win.size < n) {
      ctx.strokeStyle = "#ff3d8f";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(
        LABEL_W + win.start * colW + 0.5,
        0.5,
        Math.max(4, win.size * colW) - 1,
        height - 1,
      );
    }
    const at = grid.layers.findIndex((l) => l.layer === layer);
    if (at >= 0) {
      ctx.fillStyle = "#ff3d8f";
      ctx.fillRect(LABEL_W - 3, at * rowH, 2, rowH - gap);
    }
    if (token !== null && token < n) {
      ctx.fillStyle = "#ffffffb0";
      ctx.fillRect(LABEL_W + (token + 0.5) * colW - 0.5, 0, 1, height);
    }
  }, [scaled, width, height, rowH, plotW, L, n, win, layer, token, grid]);

  if (!L || !n) return null;
  const cellAt = (e) => {
    const b = canvas.current.getBoundingClientRect();
    const x = e.clientX - b.left,
      y = e.clientY - b.top;
    const l = Math.floor(y / rowH);
    const t = Math.floor(((x - LABEL_W) / plotW) * n);
    if (x < LABEL_W || l < 0 || l >= L || t < 0 || t >= n) return null;
    return { l, t, x, y };
  };
  const hv = hover ? grid.layers[hover.l][current]?.[hover.t] : null;
  const peakLayer = summary.reduce(
    (best, s) =>
      s.spread !== null && (best === null || s.spread > best.spread) ? s : best,
    null,
  );
  return (
    <section className="weather-map" aria-label="Layer by token map">
      <header>
        <div>
          <h3>
            Where each layer worked{" "}
            <small>
              {L} layers × {n.toLocaleString()} tokens
            </small>
          </h3>
          <p>
            {peakLayer && peakLayer.spread >= 8
              ? `L${peakLayer.layer} has a token ${Math.round(peakLayer.spread)}× its median (token ${peakLayer.peakToken}, “${tokenLabel(tokens[peakLayer.peakToken], peakLayer.peakToken)}”): a massive activation. `
              : ""}
            Click a cell to open that layer at that token.
          </p>
        </div>
        {grid.measures.length > 1 && (
          <div
            className="token-barcode-scale"
            role="group"
            aria-label="Measure"
          >
            {grid.measures.map((m) => (
              <button
                key={m.key}
                className={current === m.key ? "selected" : ""}
                aria-pressed={current === m.key}
                onClick={() => setMeasure(m.key)}
              >
                {m.label}
              </button>
            ))}
          </div>
        )}
      </header>
      <div className="weather-map-plot" ref={box}>
        <canvas
          ref={canvas}
          style={{ height }}
          role="img"
          aria-label={`${L} layers by ${n} tokens`}
          onMouseMove={(e) => setHover(cellAt(e))}
          onMouseLeave={() => setHover(null)}
          onClick={(e) => {
            const c = cellAt(e);
            if (c) onPick?.(grid.layers[c.l].layer, c.t);
          }}
        />
        {hover && (
          <div
            className="token-barcode-tip"
            style={{
              left: Math.max(0, Math.min(hover.x + 14, width - 260)),
              top: Math.max(0, hover.y - 34),
            }}
          >
            <b>L{grid.layers[hover.l].layer}</b> · {hover.t} “
            {tokenLabel(tokens[hover.t], hover.t)}” ·{" "}
            <b>{Number.isFinite(hv) ? hv.toPrecision(3) : "—"}</b> RMS
          </div>
        )}
      </div>
      <footer className="token-barcode-legend">
        <span>0</span>
        <i aria-hidden="true" className="weather-ramp" />
        <span>{scaled.scale.toPrecision(3)}</span>
        <small>
          RMS over all features of{" "}
          {grid.measures.find((m) => m.key === current)?.label}, per token; the
          colour tops out at the 98th percentile. The ribbon's window is framed.
        </small>
      </footer>
    </section>
  );
}
