import { useEffect, useMemo, useRef, useState } from "react";
import {
  SCALES,
  barcodeModel,
  isMassive,
  outlierChannels,
  scaled,
  strongest,
} from "./barcodeData";

const SCALE_KEY = "model-run.barcode-scale";
// Dark-stage diverging ramps: near zero sinks into the stage, positive rises to mint, negative to rose.
// Zero sits just above the card's own colour, so the standalone theme's stage works too.
const MID_DEFAULT = [38, 31, 66];
const POS = [
  [31, 168, 140],
  [167, 242, 219],
];
const NEG = [
  [214, 84, 116],
  [255, 184, 199],
];

function ramp(t, MID) {
  const [a, b] = t < 0 ? NEG : POS;
  // Small values sink into the stage so channels that stand out read as stripes, not static.
  const m = Math.pow(Math.min(1, Math.abs(t)), 1.6);
  const [from, to, p] = m < 0.6 ? [MID, a, m / 0.6] : [a, b, (m - 0.6) / 0.4];
  return from.map((c, i) => Math.round(c + (to[i] - c) * p));
}

const fmt = (n) =>
  Number.isFinite(n)
    ? new Intl.NumberFormat("en", { maximumSignificantDigits: 4 }).format(n)
    : "—";

function stageMid(node) {
  const hex = node
    ? getComputedStyle(node).getPropertyValue("--c-1a1436").trim()
    : "";
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) return MID_DEFAULT;
  return [0, 2, 4].map((i) =>
    Math.min(255, parseInt(m[1].slice(i, i + 2), 16) + 12),
  );
}

/** Where in the image a layer is strong: each patch tinted by its RMS over all features, over the input. */
function PatchMap({ model, grid, image, selected, onPick }) {
  const [prefix, rows, cols] = grid;
  const ref = useRef(null);
  const SIZE = 176;
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = SIZE * dpr;
    cv.height = SIZE * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const values = model.rms.slice(prefix, prefix + rows * cols);
    const finite = values.filter(Number.isFinite);
    const low = Math.min(...finite),
      high = Math.max(...finite, low + 1e-9);
    const cw = SIZE / cols,
      ch = SIZE / rows;
    const paint = () => {
      ctx.clearRect(0, 0, SIZE, SIZE);
      ctx.fillStyle = "#261f42";
      ctx.fillRect(0, 0, SIZE, SIZE);
      if (picture?.complete && picture.naturalWidth)
        ctx.drawImage(picture, 0, 0, SIZE, SIZE);
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          const v = values[r * cols + c];
          const t = Number.isFinite(v) ? (v - low) / (high - low) : 0;
          ctx.fillStyle = `rgba(31, 168, 140, ${(picture ? 0.08 : 0.25) + 0.72 * t})`;
          ctx.fillRect(c * cw, r * ch, cw, ch);
        }
      const at = (selected?.row ?? -1) - prefix;
      if (at >= 0 && at < rows * cols) {
        ctx.strokeStyle = "#ff4fa3";
        ctx.lineWidth = 2;
        ctx.strokeRect(
          (at % cols) * cw + 1,
          Math.floor(at / cols) * ch + 1,
          cw - 2,
          ch - 2,
        );
      }
    };
    let picture = null;
    if (image) {
      picture = new Image();
      picture.onload = paint;
      picture.src = image;
    }
    paint();
  }, [model, prefix, rows, cols, image, selected?.row]);
  return (
    <figure className="patch-map">
      <canvas
        ref={ref}
        style={{ width: SIZE, height: SIZE }}
        role="img"
        aria-label={`Each of the ${rows} × ${cols} patches tinted by its RMS. Click a patch to select its row.`}
        onClick={(e) => {
          const b = e.currentTarget.getBoundingClientRect();
          const c = Math.min(
            cols - 1,
            Math.floor(((e.clientX - b.left) / b.width) * cols),
          );
          const r = Math.min(
            rows - 1,
            Math.floor(((e.clientY - b.top) / b.height) * rows),
          );
          onPick(prefix + r * cols + c, selected?.col ?? 0);
        }}
      />
      <figcaption>Patch RMS over the input · brighter is stronger</figcaption>
    </figure>
  );
}

function readScale() {
  try {
    const saved = localStorage.getItem(SCALE_KEY);
    return SCALES.some(([id]) => id === saved) ? saved : "robust";
  } catch {
    return "robust";
  }
}

/** Every token of a text layer as a row and every recorded feature as a stripe, linked to the 3D window. */
export default function TokenBarcode({
  event,
  view,
  selected,
  onPick,
  image = null,
}) {
  // An image layer: [prefix tokens, patch rows, patch columns]. Rows are patches, not words.
  const patchGrid = event?.patch_grid ?? null;
  const unit = patchGrid ? "patches" : "tokens";
  const model = useMemo(() => barcodeModel(event), [event]);
  const channels = useMemo(
    () => (model ? outlierChannels(model) : []),
    [model],
  );
  const [scale, setScale] = useState(readScale);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState(null);
  const box = useRef(null);
  const canvas = useRef(null);

  useEffect(() => {
    if (!box.current) return;
    const watch = new ResizeObserver(([entry]) =>
      setWidth(Math.floor(entry.contentRect.width)),
    );
    watch.observe(box.current);
    return () => watch.disconnect();
  }, [model !== null]);

  const rowH = model
    ? Math.max(patchGrid ? 2 : 5, Math.min(22, Math.floor(380 / model.rows)))
    : 0;
  // Narrow screens give the stripes more of the width.
  const LABEL_W = width && width < 480 ? 72 : 104;
  const RMS_W = width && width < 480 ? 44 : 72;
  const plotW = Math.max(40, width - LABEL_W - RMS_W);
  const height = model ? model.rows * rowH : 0;
  const colAt = (x) => Math.floor(((x - LABEL_W) / plotW) * (model?.cols ?? 0));
  const xOf = (c) => LABEL_W + (c / model.cols) * plotW;
  // Patches are labelled once per row of the image (as often as labels fit); tokens while they fit.
  const patchEvery = patchGrid
    ? Math.max(1, Math.ceil(13 / (rowH * patchGrid[2])))
    : 1;
  const showLabel = (r) => {
    if (!patchGrid)
      return (
        rowH >= 11 || r % Math.ceil(11 / rowH) === 0 || r === selected?.row
      );
    const [prefix, , cols] = patchGrid;
    if (r < prefix) return r === 0;
    const offset = r - prefix;
    const row = offset / cols;
    // The first patch row sits right under the class token; leave it unlabelled when they would touch.
    return (
      offset % cols === 0 &&
      row % patchEvery === 0 &&
      (row > 0 || prefix * rowH >= 13)
    );
  };

  // The stripes change with the data, the scale and the width; the overlays redraw without them.
  const stripes = useMemo(() => {
    if (!model || !width || typeof document === "undefined") return null;
    const dpr = window.devicePixelRatio || 1;
    const px = Math.max(1, Math.round(plotW * dpr));
    const mid = stageMid(box.current);
    // One device pixel per column: each shows the strongest feature it covers.
    const strip = new ImageData(px, model.rows);
    for (let r = 0; r < model.rows; r++)
      for (let x = 0; x < px; x++) {
        const c0 = Math.floor((x / px) * model.cols);
        const c1 = Math.max(c0 + 1, Math.floor(((x + 1) / px) * model.cols));
        const c = strongest(model, r, c0, c1);
        const t =
          c < 0 ? null : scaled(model.values[r * model.cols + c], scale, model);
        const [red, green, blue] = t === null ? [70, 70, 78] : ramp(t, mid);
        const o = (r * px + x) * 4;
        strip.data[o] = red;
        strip.data[o + 1] = green;
        strip.data[o + 2] = blue;
        strip.data[o + 3] = 255;
      }
    const off = document.createElement("canvas");
    off.width = px;
    off.height = model.rows;
    off.getContext("2d").putImageData(strip, 0, 0);
    return off;
  }, [model, scale, width, plotW]);

  useEffect(() => {
    const cv = canvas.current;
    if (!cv || !model || !width) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(width * dpr);
    cv.height = Math.round(height * dpr);
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const off = stripes;
    if (!off) return;
    const px = off.width;
    ctx.imageSmoothingEnabled = false;
    const gap = rowH >= 10 ? 2 : rowH >= 7 ? 1 : 0;
    const maxRms = Math.max(...model.rms.filter(Number.isFinite), 1e-6);
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (let r = 0; r < model.rows; r++) {
      const y = r * rowH;
      ctx.drawImage(off, 0, r, px, 1, LABEL_W, y + gap / 2, plotW, rowH - gap);
      if (showLabel(r)) {
        ctx.fillStyle = r === selected?.row ? "#ffd3e6" : "#c9c1e6";
        ctx.textAlign = "right";
        const label = model.labels[r];
        ctx.fillText(
          label.length > 13 ? `${label.slice(0, 12)}…` : label,
          LABEL_W - 10,
          y + rowH / 2,
        );
      }
      // Each token's size over all of its features, not only the ones drawn.
      const w = ((model.rms[r] || 0) / maxRms) * (RMS_W - 22);
      ctx.fillStyle = "#7fd9c288";
      ctx.fillRect(LABEL_W + plotW + 10, y + rowH * 0.25, w, rowH * 0.5);
    }
    // Outliers beyond the robust scale get a bright tick, so saturation never hides them.
    if (scale === "robust") {
      ctx.fillStyle = "#ffffff";
      for (let r = 0; r < model.rows; r++)
        for (let c = 0; c < model.cols; c++)
          if (isMassive(model.values[r * model.cols + c], model.rms[r]))
            ctx.fillRect(
              xOf(c) - 0.5,
              r * rowH + gap / 2,
              1.5,
              Math.max(2, (rowH - gap) * 0.35),
            );
    }
    // The part the 3D stage shows, and the selected cell.
    if (view) {
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = "#ffffffb0";
      ctx.lineWidth = 1;
      const x0 = xOf(view.col0);
      ctx.strokeRect(
        x0 + 0.5,
        view.row0 * rowH + 0.5,
        Math.max(3, xOf(view.col0 + view.cols) - x0) - 1,
        view.rows * rowH - 1,
      );
      ctx.setLineDash([]);
    }
    if (selected) {
      ctx.strokeStyle = "#ff4fa3";
      ctx.lineWidth = 2;
      const x = xOf(selected.col);
      ctx.strokeRect(
        x - 2,
        selected.row * rowH,
        Math.max(4, plotW / model.cols + 3),
        rowH,
      );
    }
  }, [model, width, height, rowH, plotW, scale, view, selected, stripes]);

  if (!model) return null;

  function pointAt(e) {
    const b = canvas.current.getBoundingClientRect();
    const x = e.clientX - b.left,
      y = e.clientY - b.top;
    const r = Math.floor(y / rowH);
    if (x < LABEL_W || x >= LABEL_W + plotW || r < 0 || r >= model.rows)
      return null;
    const per = model.cols / plotW;
    const c0 = Math.floor((x - LABEL_W) * per);
    const c = per > 1 ? strongest(model, r, c0, Math.ceil(c0 + per)) : colAt(x);
    return c < 0
      ? null
      : {
          row: r,
          col: c,
          x,
          y,
          span:
            per > 1
              ? [c0, Math.min(model.cols, Math.ceil(c0 + per)) - 1]
              : null,
        };
  }
  function move(dr, dc) {
    const row = Math.max(
      0,
      Math.min(model.rows - 1, (selected?.row ?? 0) + dr),
    );
    const col = Math.max(
      0,
      Math.min(model.cols - 1, (selected?.col ?? 0) + dc),
    );
    onPick(row, col);
  }
  const hv = hover && model.values[hover.row * model.cols + hover.col];
  const ticks = (plotW < 320 ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).map((f) =>
    Math.min(model.cols - 1, Math.round(f * (model.cols - 1))),
  );
  const subsampled = event.source_shape?.at(-1) > model.cols;

  return (
    <section className="token-barcode" aria-label="Every token by feature">
      <header>
        <div>
          <h3>
            {patchGrid ? "Every patch × feature" : "Every token × feature"}
          </h3>
          <p>
            {patchGrid
              ? `${patchGrid[1]} × ${patchGrid[2]} patches${patchGrid[0] ? ` and ${patchGrid[0] === 1 ? "the class token" : `${patchGrid[0]} other tokens`}` : ""}`
              : `${model.rows} tokens`}{" "}
            · {model.cols.toLocaleString()}{" "}
            {subsampled
              ? `of ${event.source_shape.at(-1).toLocaleString()} features, evenly spaced`
              : "features"}{" "}
            · exact values.{" "}
            {view
              ? "Click to move the 3D window there."
              : "Click a row or a patch on the map to follow it."}
          </p>
        </div>
        <div
          className="token-barcode-scale"
          role="group"
          aria-label="Colour scale"
        >
          {SCALES.map(([id, label, title]) => (
            <button
              key={id}
              className={scale === id ? "selected" : ""}
              aria-pressed={scale === id}
              title={title}
              onClick={() => {
                setScale(id);
                try {
                  localStorage.setItem(SCALE_KEY, id);
                } catch {
                  // the choice just won't be remembered
                }
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </header>
      {channels.length > 0 && (
        <div className="token-barcode-channels">
          <span>Outlier channels</span>
          {channels.map((ch) => (
            <button
              key={ch.col}
              onClick={() => onPick(ch.row, ch.col)}
              title={`Feature #${ch.feature} is over 10× the RMS in ${ch.tokens} of ${model.rows} ${unit}. Click to select its largest value.`}
            >
              <b>#{ch.feature}</b> {ch.tokens} of {model.rows} {unit} · peak{" "}
              {fmt(ch.peak)}
            </button>
          ))}
        </div>
      )}
      <div className={`token-barcode-body ${patchGrid ? "with-map" : ""}`}>
        {patchGrid && (
          <PatchMap
            model={model}
            grid={patchGrid}
            image={image}
            selected={selected}
            onPick={onPick}
          />
        )}
        <div className="token-barcode-plot" ref={box}>
          <canvas
            ref={canvas}
            style={{ height }}
            tabIndex={0}
            role="img"
            aria-label={`${model.rows} tokens by ${model.cols} features. Arrow keys move the selected cell.`}
            onMouseMove={(e) => setHover(pointAt(e))}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => {
              const p = pointAt(e);
              if (p) onPick(p.row, p.col);
            }}
            onKeyDown={(e) => {
              const step = e.shiftKey
                ? Math.max(1, Math.round(model.cols / 32))
                : 1;
              const keys = {
                ArrowLeft: [0, -step],
                ArrowRight: [0, step],
                ArrowUp: [-1, 0],
                ArrowDown: [1, 0],
              };
              if (!keys[e.key]) return;
              e.preventDefault();
              move(...keys[e.key]);
            }}
          />
          {hover && (
            <div
              className="token-barcode-tip"
              style={{
                left: Math.max(0, Math.min(hover.x + 14, width - 300)),
                maxWidth: width,
                top: Math.max(0, hover.y - 34),
              }}
            >
              <b>“{model.labels[hover.row]}”</b> · #{model.features[hover.col]}{" "}
              <b>{fmt(hv)}</b>
              {model.rms[hover.row] > 0 &&
                ` · ${fmt(Math.abs(hv) / model.rms[hover.row])}× its RMS`}
              {hover.span &&
                ` · strongest of #${model.features[hover.span[0]]}–#${model.features[hover.span[1]]}`}
            </div>
          )}
          <div
            className="token-barcode-axis"
            style={{ marginLeft: LABEL_W, width: plotW }}
          >
            {ticks.map((c, i) => (
              <span
                key={i}
                style={{ left: `${(c / Math.max(1, model.cols - 1)) * 100}%` }}
              >
                #{model.features[c]}
              </span>
            ))}
            <em style={{ left: `calc(100% + ${RMS_W / 2}px)` }}>RMS</em>
          </div>
        </div>
      </div>
      <footer className="token-barcode-legend">
        <span className="neg">
          −
          {fmt(
            scale === "linear"
              ? model.max
              : scale === "robust"
                ? model.p99
                : model.max,
          )}
        </span>
        <i aria-hidden="true" />
        <span className="pos">
          +
          {fmt(
            scale === "linear"
              ? model.max
              : scale === "robust"
                ? model.p99
                : model.max,
          )}
        </span>
        <small>
          {scale === "robust"
            ? `Saturates at the 99th percentile (largest ${fmt(model.max)}); white ticks mark massive values, over 10× their token's RMS.`
            : scale === "log"
              ? "Signed log: every size stays visible; colour is not proportional."
              : "Linear: the largest value sets the scale."}{" "}
          Each pixel column shows its strongest feature.
        </small>
      </footer>
    </section>
  );
}
