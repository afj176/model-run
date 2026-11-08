import { useEffect, useMemo, useRef, useState } from "react";
import { Lightbulb } from "@phosphor-icons/react";
import {
  pct,
  responseSegments,
  responseStory,
  tokenLabel,
  tokenText,
} from "./insights";
import { CacheGrowth } from "./TextPanels";

/** Answers longer than this collapse their sure stretches to plain text. */
const COLLAPSE_AT = 120;
const NEAR = 6;

/** The model's confidence across the whole answer as one line, dips marked; a click picks a token. */
function SurpriseLine({ steps, index, onPick, dips }) {
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
  const H = 44,
    PAD = 2;
  const n = steps.length;
  const x = (i) => PAD + ((i + 0.5) / n) * (width - 2 * PAD);
  const y = (p) => PAD + (1 - p) * (H - 2 * PAD);
  const path = steps
    .map((s, i) => `${i ? "L" : "M"}${x(i)} ${y(s.probability)}`)
    .join(" ");
  const pickAt = (e) => {
    const b = box.current.getBoundingClientRect();
    const i = Math.floor(((e.clientX - b.left - PAD) / (width - 2 * PAD)) * n);
    if (i >= 0 && i < n) onPick(i);
  };
  return (
    <div className="surprise" ref={box}>
      {width > 0 && (
        <svg
          width={width}
          height={H}
          role="img"
          aria-label="Probability of each written token, in order"
          onClick={pickAt}
        >
          <path
            d={`${path} L${x(n - 1)} ${H} L${x(0)} ${H} Z`}
            className="surprise-fill"
          />
          <path d={path} className="surprise-line" />
          {dips.map((i) => (
            <circle
              key={i}
              cx={x(i)}
              cy={y(steps[i].probability)}
              r={3}
              className="surprise-dip"
            />
          ))}
          {index !== null && (
            <line
              x1={x(index)}
              x2={x(index)}
              y1={0}
              y2={H}
              className="surprise-pick"
            />
          )}
        </svg>
      )}
      <span className="surprise-axis">
        <em>sure</em>
        <em>
          {n} tokens · click the line to pick a token; dots are the closest
          calls
        </em>
      </span>
    </div>
  );
}

/** The model's response and, for each token it wrote, where that token came from. */
export default function Response({ trace, doubtful = false, onRibbon = null }) {
  const steps = trace?.steps || [];
  const [picked, setPicked] = useState(null);
  const [showThinking, setShowThinking] = useState(true);
  const [opened, setOpened] = useState(() => []);
  useEffect(() => {
    setPicked(null);
    setOpened([]);
  }, [trace]);
  const story = useMemo(() => responseStory(trace), [trace]);
  const segments = useMemo(() => responseSegments(steps), [steps]);
  if (!steps.length) return null;
  const index = picked ?? story.leastConfident?.index ?? 0;
  const step = steps[index];
  const context = [
    ...(trace.tokens || []).map((t) => t.text),
    ...steps.slice(0, index).map((s) => s.text),
  ];
  const attention = step.attention || null;
  const peak = attention ? Math.max(1e-9, ...attention) : 1;
  let beyondSink = null;
  if (attention && attention.length > 1) {
    let best = 1;
    for (let k = 2; k < attention.length; k++)
      if (attention[k] > attention[best]) best = k;
    beyondSink = { index: best, value: attention[best] };
  }
  const lens = step.lens || null;
  const layerCount = lens ? lens.chosen_probability_by_layer.length : 0;
  const layersTotal =
    steps.find((s) => s.lens)?.lens?.chosen_probability_by_layer?.length ?? 0;
  // Where the model hesitated: the closest calls and every token under 30%.
  const dips = useMemo(() => {
    const set = new Set((story.closeCalls || []).map((c) => c.index));
    steps.forEach((s, i) => {
      if (s.probability < 0.3) set.add(i);
    });
    return [...set].sort((a, b) => a - b);
  }, [steps, story]);
  // How deep in the model each token was decided: 0 at the embeddings, 1 at the last layer.
  const depthOf = (s) =>
    s.lens && Number.isInteger(s.lens.settled_layer) && layersTotal > 1
      ? s.lens.settled_layer / (layersTotal - 1)
      : null;
  const collapse = steps.length > COLLAPSE_AT;
  const keep = useMemo(() => {
    if (!collapse) return null;
    const near = new Set();
    const mark = (i) => {
      for (
        let k = Math.max(0, i - NEAR);
        k <= Math.min(steps.length - 1, i + NEAR);
        k++
      )
        near.add(k);
    };
    dips.forEach(mark);
    mark(index);
    if (segments.thinking) {
      mark(segments.thinking[0]);
      mark(segments.thinking[1]);
    }
    for (const [a, b] of opened) for (let k = a; k < b; k++) near.add(k);
    return near;
  }, [collapse, dips, index, segments, opened, steps.length]);
  const segmentOf = (i) =>
    segments.thinking && i >= segments.thinking[0] && i < segments.thinking[1]
      ? "thinking"
      : "answer";
  const tokenButton = (s, i) => (
    <button
      key={i}
      role="listitem"
      className={`response-token ${i === index ? "picked" : ""} ${s.probability < 0.3 ? "unsure" : ""} ${depthOf(s) !== null ? "deep" : ""}`}
      style={{ "--p": s.probability, "--depth": depthOf(s) ?? 0 }}
      title={`token ${i}: “${tokenText(s.text)}” ${pct(s.probability)} · ${s.entropy_bits.toFixed(2)} bits${s.lens?.settled_layer !== undefined && s.lens?.settled_layer !== null ? ` · settled at ${s.lens.settled_layer === 0 ? "the embeddings" : `layer ${s.lens.settled_layer - 1}`}` : ""}`}
      onClick={() => setPicked(i)}
    >
      {s.text === "" ? "∅" : s.text}
    </button>
  );
  /** Tokens from `from` to `to`: every chip, or, in a long answer, chips near the dips and plain text between. */
  const renderRange = (from, to) => {
    if (!keep)
      return steps.slice(from, to).map((s, k) => tokenButton(s, from + k));
    const out = [];
    let i = from;
    while (i < to) {
      if (keep.has(i)) {
        out.push(tokenButton(steps[i], i));
        i++;
        continue;
      }
      let j = i;
      while (j < to && !keep.has(j)) j++;
      const run = [i, j];
      out.push(
        <button
          key={`run-${i}`}
          className="response-run"
          title={`${j - i} tokens the model was sure of (each at least 30%). Click to show them.`}
          onClick={() => setOpened((o) => [...o, run])}
        >
          {steps
            .slice(i, j)
            .map((s) => s.text)
            .join("")}
        </button>,
      );
      i = j;
    }
    return out;
  };
  return (
    <div className="response">
      <div className="prediction-head">
        <h2>
          What it wrote
          <small>
            {trace.decoding === "sampled"
              ? `sampled${trace.sampling?.temperature !== undefined ? ` at temperature ${trace.sampling.temperature}` : ""}`
              : "greedy"}{" "}
            · {steps.length} token{steps.length === 1 ? "" : "s"} · stopped:{" "}
            {trace.stopped === "eos"
              ? "end of text"
              : trace.stopped.replace(/_/g, " ")}
          </small>
        </h2>
        <span
          className={`prediction-confidence ${doubtful ? "undecided" : story.meanProbability >= 0.6 ? "confident" : story.meanProbability >= 0.3 ? "leaning" : "undecided"}`}
        >
          {doubtful
            ? "check the audio"
            : story.meanProbability >= 0.6
              ? "confident"
              : story.meanProbability >= 0.3
                ? "leaning"
                : "undecided"}
          <small>mean token probability {pct(story.meanProbability)}</small>
        </span>
      </div>
      <p className="prediction-story">
        <Lightbulb size={16} weight="duotone" /> {story.text}
      </p>
      {onRibbon && (
        <p className="response-ribbon-link">
          <button className="text-button" onClick={onRibbon}>
            Colour the prompt ribbon by how much this answer read each token
          </button>
        </p>
      )}
      <SurpriseLine
        steps={steps}
        index={index}
        onPick={setPicked}
        dips={dips}
      />
      <div className="response-text" role="list" aria-label="Generated tokens">
        <span className="response-prompt">
          {(trace.tokens || []).map((t) => t.text).join("")}
        </span>
        {segments.thinking ? (
          <>
            {renderRange(0, segments.thinking[0])}
            <span className="response-segment thinking">
              <button
                className="response-segment-toggle"
                onClick={() => setShowThinking((v) => !v)}
                aria-expanded={showThinking}
              >
                thinking · {segments.thinking[1] - segments.thinking[0]} tokens
                {segments.closed ? "" : " · unfinished"}{" "}
                {showThinking ? "▾" : "▸"}
              </button>
              {showThinking &&
                renderRange(segments.thinking[0], segments.thinking[1])}
            </span>
            {segments.closed && (
              <span className="response-segment answer">
                <span className="response-segment-label">answer</span>
                {renderRange(segments.thinking[1], steps.length)}
              </span>
            )}
          </>
        ) : (
          renderRange(0, steps.length)
        )}
      </div>
      <p className="micro">
        Click a token. Shade is the probability the model gave it; tokens under
        30% are outlined.
        {layersTotal > 1
          ? " The underline is how deep the token was decided: faint for one settled in the early layers, dark for one that only settled near the end."
          : ""}
        {collapse
          ? " Long answer: stretches the model was sure of are shown as plain text; click one to open it."
          : ""}{" "}
        The prompt is shown plain.
        {segments.thinking
          ? " Thinking tokens are what the model wrote between its own <think> markers; they are its visible reasoning, not a hidden state."
          : ""}
      </p>
      {story.closeCalls?.length > 0 && (
        <div className="response-forks">
          <span className="matrix-insight-sub">Closest calls</span>
          {story.closeCalls.map((c) => (
            <button
              key={c.index}
              className="response-fork"
              onClick={() => setPicked(c.index)}
              title={`token ${c.index}: ${pct(c.probability)} vs ${pct(c.runnerUp.probability)}`}
            >
              “{tokenText(c.text)}” <em>over</em> “{tokenText(c.runnerUp.text)}”
              <small>
                {pct(c.probability)} vs {pct(c.runnerUp.probability)}
              </small>
            </button>
          ))}
        </div>
      )}
      <CacheGrowth steps={steps} />
      <div className="response-detail">
        <div>
          <h3>
            Token {index}: “{tokenText(step.text)}”{" "}
            <small>
              {segmentOf(index) === "thinking" ? "thinking · " : ""}
              {pct(step.probability)} · {step.entropy_bits.toFixed(2)} bits of
              uncertainty
            </small>
          </h3>
          <ol className="prediction-list">
            {step.top_tokens.map((t, k) => (
              <li key={t.token_id}>
                <span className="prediction-rank">{k + 1}</span>
                <span className="prediction-token">
                  {t.text === "" ? "∅" : tokenText(t.text)}
                </span>
                <span className="prediction-bar">
                  <i
                    style={{ width: `${Math.max(1, t.probability * 100)}%` }}
                  />
                </span>
                <b>{pct(t.probability)}</b>
              </li>
            ))}
          </ol>
          {lens && (
            <>
              <p className="matrix-insight-sub">
                How it formed{" "}
                <small>
                  probability of “{tokenText(step.text)}” at each layer ·{" "}
                  {lens.settled_layer === null
                    ? "never on top before the output"
                    : lens.settled_layer === 0
                      ? "on top from the embeddings"
                      : `on top from layer ${lens.settled_layer - 1} of ${layerCount - 1}`}
                </small>
              </p>
              <div className="lens-track response-lens" role="list">
                {lens.chosen_probability_by_layer.map((p, l) => (
                  <span
                    key={l}
                    role="listitem"
                    className={`lens-cell ${lens.settled_layer !== null && l >= lens.settled_layer ? "final" : ""}`}
                    style={{ "--p": p }}
                    title={`${l === 0 ? "embeddings" : `layer ${l - 1}`}: ${pct(p)}`}
                  >
                    <i />
                    <small>{l === 0 ? "emb" : `L${l - 1}`}</small>
                  </span>
                ))}
              </div>
            </>
          )}
        </div>
        <div>
          {attention && (
            <>
              <p className="matrix-insight-sub">
                What it looked at{" "}
                <small>(every layer and head, averaged)</small>
              </p>
              <div className="matrix-attribution">
                {attention.map((v, k) => (
                  <span
                    key={k}
                    style={{ "--heat": Math.min(1, v / peak) }}
                    title={`${k}: ${pct(v)}`}
                    className={
                      k >= (trace.tokens || []).length ? "generated" : ""
                    }
                  >
                    {tokenLabel({ text: context[k] ?? String(k) }, k)}
                    <small>{pct(v)}</small>
                  </span>
                ))}
              </div>
              <p className="micro">
                {beyondSink && attention[0] >= 0.3
                  ? `${pct(attention[0])} sits on the first token (an attention sink); beyond it the strongest pull is “${tokenText(context[beyondSink.index] ?? String(beyondSink.index))}” at ${pct(beyondSink.value)}. `
                  : ""}
                Measured attention from the position that produced this token to
                every earlier token, prompt and generated alike. Attention is
                not a causal attribution.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
