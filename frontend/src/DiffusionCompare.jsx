import { useEffect, useMemo, useRef, useState } from "react";
import { Lightbulb, Play } from "@phosphor-icons/react";
import {
  diffusionCompareStory,
  pct,
  pctFine,
  stepDeltas,
  tokenMassShift,
  tokenText,
} from "./insights";

const isPad = (labels, i) =>
  labels?.[i] !== undefined &&
  /^<\|?(endoftext|pad)\|?>$/i.test(labels[i]) &&
  i > 0 &&
  labels[i - 1] === labels[i];

/** Diffusion runs side by side: token ablation rerun, image difference, latent divergence. */
export default function DiffusionCompare({
  traceA,
  traceB,
  busy,
  canRun,
  onRunAblation,
}) {
  const labels = traceA?.cross_attention?.text_labels || null;
  const tokenCount =
    traceA?.cross_attention?.events?.[0]?.text_tokens ?? labels?.length ?? 0;
  const [picked, setPicked] = useState([]);
  const [manual, setManual] = useState("");
  useEffect(() => {
    setPicked([]);
    setManual("");
  }, [traceA]);
  const chosen = useMemo(() => {
    const typed = manual
      .replace(/,/g, " ")
      .split(/\s+/)
      .filter(Boolean)
      .map(Number)
      .filter((n) => Number.isInteger(n) && n >= 0 && n < 1024);
    return [...new Set([...picked, ...typed])].sort((a, b) => a - b);
  }, [picked, manual]);
  const canAblate =
    Boolean(onRunAblation) && canRun && Boolean(traceA?.sampling?.prompt);
  const [pixels, setPixels] = useState(null);
  const diffRef = useRef(null);
  const imageA = traceA?.preview?.base64,
    imageB = traceB?.preview?.base64;
  useEffect(() => {
    setPixels(null);
    if (!imageA || !imageB || !diffRef.current) return;
    let alive = true;
    const load = (b64) =>
      new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = `data:image/png;base64,${b64}`;
      });
    Promise.all([load(imageA), load(imageB)])
      .then(([a, b]) => {
        if (!alive) return;
        const w = Math.min(a.width, b.width, 256),
          h = Math.min(a.height, b.height, 256);
        const read = (img) => {
          const c = document.createElement("canvas");
          c.width = w;
          c.height = h;
          const ctx = c.getContext("2d");
          ctx.drawImage(img, 0, 0, w, h);
          return ctx.getImageData(0, 0, w, h).data;
        };
        const da = read(a),
          db = read(b);
        const canvas = diffRef.current;
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        const out = ctx.createImageData(w, h);
        const grid = new Array(16).fill(0),
          counts = new Array(16).fill(0);
        let total = 0;
        for (let i = 0; i < w * h; i++) {
          const at = i * 4;
          const d =
            (Math.abs(da[at] - db[at]) +
              Math.abs(da[at + 1] - db[at + 1]) +
              Math.abs(da[at + 2] - db[at + 2])) /
            3;
          total += d;
          const x = i % w,
            y = Math.floor(i / w);
          const cell =
            Math.min(3, Math.floor((y / h) * 4)) * 4 +
            Math.min(3, Math.floor((x / w) * 4));
          grid[cell] += d;
          counts[cell]++;
          const t = Math.min(1, d / 96);
          out.data[at] = Math.round(242 + (255 - 242) * t);
          out.data[at + 1] = Math.round(239 - (239 - 61) * t);
          out.data[at + 2] = Math.round(228 - (228 - 143) * t);
          out.data[at + 3] = 255;
        }
        ctx.putImageData(out, 0, 0);
        setPixels({
          mean: total / (w * h) / 255,
          grid: grid.map((v, i) => (counts[i] ? v / counts[i] / 255 : 0)),
        });
      })
      .catch(() => alive && setPixels(null));
    return () => {
      alive = false;
    };
  }, [imageA, imageB]);
  const deltas = useMemo(
    () => (traceB ? stepDeltas(traceA?.events, traceB?.events) : []),
    [traceA, traceB],
  );
  const shift = useMemo(
    () =>
      traceB
        ? tokenMassShift(traceA?.cross_attention, traceB?.cross_attention)
        : null,
    [traceA, traceB],
  );
  const story = useMemo(
    () =>
      traceB
        ? diffusionCompareStory({
            ablation: traceB.token_ablation || null,
            pixels,
            deltas,
            shift,
          })
        : null,
    [traceB, pixels, deltas, shift],
  );
  const peakDelta = Math.max(1e-9, ...deltas.map((d) => d.relative));
  return (
    <>
      {canAblate && (
        <div className="compare-card wide">
          <h3>
            Run without a token{" "}
            <small className="muted">
              token ablation · same prompt, seed, steps and size
            </small>
          </h3>
          <p className="micro">
            Masks the chosen text tokens out of every cross-attention call for a
            whole run, then pairs the result as B. An intervention on attention
            only: the tokens still passed through the text encoder.
          </p>
          {labels ? (
            <div className="xattn-strip ablate-strip" role="list">
              {labels.slice(0, tokenCount || labels.length).map((label, i) =>
                isPad(labels, i) ? null : (
                  <button
                    key={i}
                    role="listitem"
                    aria-pressed={picked.includes(i)}
                    className={picked.includes(i) ? "picked" : ""}
                    title={`token ${i}`}
                    onClick={() =>
                      setPicked((p) =>
                        p.includes(i) ? p.filter((x) => x !== i) : [...p, i],
                      )
                    }
                  >
                    {tokenText(label)}
                  </button>
                ),
              )}
            </div>
          ) : (
            <label className="ablate-manual">
              Token indices
              <input
                value={manual}
                onChange={(e) => setManual(e.target.value)}
                placeholder="e.g. 2 5"
                aria-label="Token indices to ablate"
              />
              <small>
                Capture with a cross-attention map to pick tokens by name.
              </small>
            </label>
          )}
          <div className="compare-patch-actions">
            <button
              disabled={busy || !chosen.length}
              onClick={() => onRunAblation(chosen)}
            >
              <Play size={14} weight="fill" />{" "}
              {busy
                ? "Running…"
                : chosen.length
                  ? `Run without ${chosen.length === 1 ? `token ${chosen[0]}` : `${chosen.length} tokens`}`
                  : "Pick tokens to ablate"}
            </button>
          </div>
        </div>
      )}
      {traceB && story?.text && (
        <p className="compare-story">
          <Lightbulb size={16} weight="duotone" /> {story.text}
        </p>
      )}
      {traceB && (
        <div className="compare-card wide">
          <h3>
            Images{" "}
            <small className="muted">
              {traceB.token_ablation
                ? `B ran without ${traceB.token_ablation.labels.map((l) => `“${tokenText(l)}”`).join(", ")}`
                : "two retained runs"}
            </small>
          </h3>
          <div className="diffusion-images">
            {[traceA, traceB].map((t, i) => (
              <figure key={i}>
                {t?.preview?.base64 ? (
                  <img
                    alt={`${i ? "B" : "A"} output`}
                    src={`data:image/png;base64,${t.preview.base64}`}
                  />
                ) : (
                  <div className="diffusion-missing">no preview</div>
                )}
                <figcaption>
                  <b>{i ? "B" : "A"}</b>{" "}
                  {t?.sampling
                    ? `seed ${t.sampling.seed} · ${t.sampling.steps} steps · ${t.sampling.size}px`
                    : t?.captured_calls
                      ? `${t.captured_calls} denoiser calls`
                      : ""}
                </figcaption>
              </figure>
            ))}
            <figure>
              <canvas
                ref={diffRef}
                className={imageA && imageB ? "" : "hidden"}
                aria-label="Pixel difference"
              />
              {!(imageA && imageB) && (
                <div className="diffusion-missing">needs both previews</div>
              )}
              <figcaption>
                <b>|A − B|</b>{" "}
                {pixels
                  ? `mean ${pctFine(pixels.mean)} of full scale`
                  : "pixel difference"}
              </figcaption>
            </figure>
          </div>
          {deltas.length > 0 && (
            <>
              <p className="matrix-insight-sub">
                Latent divergence per step{" "}
                <small>
                  (RMS of A − B relative to A's RMS, retained prefix)
                </small>
              </p>
              <div className="matrix-deltas compare-deltas">
                {deltas.map((d) => (
                  <button
                    key={d.step}
                    disabled
                    title={`step ${d.step}: Δ ${d.rms.toFixed(3)} (${pct(d.relative)} of A's RMS ${d.rmsA.toFixed(3)})`}
                  >
                    <i
                      style={{
                        height: `${Math.max(2, (d.relative / peakDelta) * 48)}px`,
                      }}
                    />
                    <small>{d.step}</small>
                  </button>
                ))}
              </div>
            </>
          )}
          {shift && shift.ranked.length > 0 && (
            <>
              <p className="matrix-insight-sub">
                Tokens whose attention changed most{" "}
                <small>(mean attention per region, B − A)</small>
              </p>
              <ul className="matrix-attends">
                {shift.ranked.slice(0, 6).map((s) => (
                  <li key={s.index}>
                    <button
                      disabled
                      title={`A ${pct(s.before)} → B ${pct(s.after)}`}
                    >
                      <span className="matrix-attends-token">
                        {s.label ? `“${tokenText(s.label)}”` : `#${s.index}`}
                      </span>
                      <span
                        className={`matrix-attends-bar ${s.delta < 0 ? "down" : ""}`}
                        style={{
                          width: `${Math.max(2, (Math.abs(s.delta) / shift.peak) * 100)}%`,
                        }}
                      />
                      <b>
                        {s.delta >= 0 ? "+" : ""}
                        {pctFine(s.delta)}
                      </b>
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
          <p className="micro">
            {traceB.token_ablation?.limitations?.[0] ||
              "Both runs must share prompt, seed, steps and size for the difference to mean anything."}
          </p>
        </div>
      )}
    </>
  );
}
