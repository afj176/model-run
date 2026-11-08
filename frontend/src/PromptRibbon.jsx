import { useEffect, useMemo, useRef, useState } from "react";
import { CaretLeft, CaretRight } from "@phosphor-icons/react";
import {
  clampWindow,
  normalize,
  segmentAt,
  segmentShares,
  sentenceSegments,
} from "./ribbonData";
import { tokenLabel } from "./insights";

// The ribbon's heat: a quiet stage colour for small values, mint, then deep teal.
const LOW = [43, 36, 82];
const MID = [127, 217, 194];
const HIGH = [15, 159, 134];
export function heat(t) {
  if (t === null || t === undefined) return "rgb(70,70,78)";
  const [a, b, p] = t < 0.5 ? [LOW, MID, t * 2] : [MID, HIGH, (t - 0.5) * 2];
  return `rgb(${a.map((c, i) => Math.round(c + (b[i] - c) * p)).join(",")})`;
}

/** Prompts longer than this fold into sentences above the strip. */
const FOLD_AT = 48;

/**
 * Every token of the prompt as one thin column, coloured by a measure, with a reading window
 * drawn in full. The window is shared: the views under it draw only what it holds.
 */
export default function PromptRibbon({
  tokens = [],
  measures = [],
  measure,
  onMeasure,
  window: win,
  onWindow,
  selected = null,
  onPick,
  caption = null,
}) {
  const n = tokens.length;
  const current =
    measures.find((m) => m.key === measure) ?? measures[0] ?? null;
  const values = useMemo(() => normalize(current?.values ?? null), [current]);
  const segments = useMemo(
    () => (n > FOLD_AT ? sentenceSegments(tokens) : null),
    [tokens, n],
  );
  const shares = useMemo(
    () => (segments ? segmentShares(segments, current?.values) : null),
    [segments, current],
  );
  const strip = useRef(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!strip.current) return;
    const watch = new ResizeObserver(([entry]) =>
      setWidth(Math.floor(entry.contentRect.width)),
    );
    watch.observe(strip.current);
    return () => watch.disconnect();
  }, []);
  const canvas = useRef(null);
  const STRIP_H = 18;
  useEffect(() => {
    const cv = canvas.current;
    if (!cv || !width || !n) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(width * dpr);
    cv.height = Math.round(STRIP_H * dpr);
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "rgb(43,36,82)";
    ctx.fillRect(0, 0, width, STRIP_H);
    const colW = width / n;
    // One device pixel per column at least; a long prompt shows each column's strongest token.
    const px = Math.max(1, Math.round(width * dpr));
    const per = n / px;
    for (let x = 0; x < px; x++) {
      const i0 = Math.floor(x * per);
      const i1 = Math.max(i0 + 1, Math.floor((x + 1) * per));
      let best = null;
      for (let i = i0; i < i1 && i < n; i++) {
        const v = values?.[i] ?? null;
        if (v !== null && (best === null || v > best)) best = v;
      }
      ctx.fillStyle = heat(best);
      ctx.fillRect(x / dpr, 0, Math.max(colW, 1 / dpr), STRIP_H);
    }
  }, [values, width, n]);

  if (!n) return null;
  const size = win?.size ?? n;
  const start = win?.start ?? 0;
  const colW = width / n;
  const move = (to) => onWindow?.(clampWindow(to, size, n));

  function dragStart(e) {
    e.preventDefault();
    const box = strip.current.getBoundingClientRect();
    const at = (clientX) => Math.floor(((clientX - box.left) / box.width) * n);
    const grab = at(e.clientX);
    const inside = grab >= start && grab < start + size;
    const offset = inside ? grab - start : Math.floor(size / 2);
    move(grab - offset);
    const onMove = (ev) => move(at(ev.clientX) - offset);
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }
  function keys(e) {
    const step = e.shiftKey ? size : 1;
    if (e.key === "ArrowLeft") move(start - step);
    else if (e.key === "ArrowRight") move(start + step);
    else if (e.key === "Home") move(0);
    else if (e.key === "End") move(n);
    else return;
    e.preventDefault();
  }
  const windowTokens = tokens.slice(start, start + size);
  const picked = selected !== null ? segmentAt(segments, selected) : null;
  return (
    <section className="ribbon" aria-label="Prompt ribbon">
      <header className="ribbon-head">
        <span className="ribbon-title">
          Prompt <small>{n.toLocaleString()} tokens</small>
        </span>
        {measures.length > 1 ? (
          <label className="ribbon-measure">
            colour by
            <select
              value={current?.key}
              onChange={(e) => onMeasure?.(e.target.value)}
              aria-label="Ribbon measure"
            >
              {measures.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
        ) : current ? (
          <span className="ribbon-measure">{current.label}</span>
        ) : null}
        {caption && <span className="ribbon-caption">{caption}</span>}
        {size < n && (
          <span className="ribbon-window-label">
            <button
              aria-label="Previous window"
              onClick={() => move(start - size)}
              disabled={start === 0}
            >
              <CaretLeft size={14} />
            </button>
            window {start}–{start + size - 1}
            <button
              aria-label="Next window"
              onClick={() => move(start + size)}
              disabled={start + size >= n}
            >
              <CaretRight size={14} />
            </button>
          </span>
        )}
      </header>
      <div
        className="ribbon-window"
        role="list"
        aria-label={`Tokens ${start} to ${start + size - 1}`}
      >
        {windowTokens.map((t, k) => {
          const i = start + k;
          const v = values?.[i] ?? null;
          return (
            <button
              key={i}
              role="listitem"
              className={`ribbon-token ${i === selected ? "picked" : ""}`}
              style={{
                background: heat(v),
                color: v > 0.55 ? "#1a1436" : undefined,
              }}
              title={`${i}: “${tokenLabel(t, i)}”${current && Number.isFinite(current.values?.[i]) ? ` · ${current.label} ${current.values[i].toPrecision(3)}` : ""}`}
              onClick={() => onPick?.(i)}
            >
              {tokenLabel(t, i)}
            </button>
          );
        })}
      </div>
      {segments && (
        <div className="ribbon-segments" role="list" aria-label="Sentences">
          {segments.map((s, k) => (
            <button
              key={k}
              role="listitem"
              className={`ribbon-segment ${picked?.index === s.index ? "picked" : ""} ${start < s.end && start + size > s.start ? "shown" : ""}`}
              style={{
                flexGrow: s.end - s.start,
                background: heat(
                  shares
                    ? Math.min(1, shares[k] * segments.length * 0.5)
                    : null,
                ),
              }}
              title={`${k + 1}: ${s.text.length > 90 ? `${s.text.slice(0, 89)}…` : s.text} · ${s.end - s.start} tokens${shares ? ` · ${Math.round(shares[k] * 100)}% of ${current.label}` : ""}`}
              onClick={() => {
                onWindow?.(clampWindow(s.start, Math.max(size, 1), n));
                onPick?.(s.start);
              }}
            >
              {(s.end - s.start) * colW > 14 ? k + 1 : ""}
            </button>
          ))}
        </div>
      )}
      <div
        className="ribbon-strip"
        ref={strip}
        tabIndex={0}
        role="slider"
        aria-label="Reading window over the prompt"
        aria-valuemin={0}
        aria-valuemax={Math.max(0, n - size)}
        aria-valuenow={start}
        onPointerDown={dragStart}
        onKeyDown={keys}
      >
        <canvas ref={canvas} style={{ width, height: STRIP_H }} />
        {size < n && width > 0 && (
          <i
            className="ribbon-frame"
            style={{
              left: start * colW,
              width: Math.max(6, size * colW),
            }}
          />
        )}
        {selected !== null && width > 0 && (
          <i
            className="ribbon-mark"
            style={{ left: (selected + 0.5) * colW }}
          />
        )}
      </div>
    </section>
  );
}
