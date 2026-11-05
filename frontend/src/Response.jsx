import { useEffect, useMemo, useState } from "react";
import { Lightbulb } from "@phosphor-icons/react";
import {
  pct,
  responseSegments,
  responseStory,
  tokenLabel,
  tokenText,
} from "./insights";

/** The model's greedy response and, for each token it wrote, where that token came from. */
export default function Response({ trace, doubtful = false }) {
  const steps = trace?.steps || [];
  const [picked, setPicked] = useState(null);
  const [showThinking, setShowThinking] = useState(true);
  useEffect(() => {
    setPicked(null);
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
  const segmentOf = (i) =>
    segments.thinking && i >= segments.thinking[0] && i < segments.thinking[1]
      ? "thinking"
      : "answer";
  const tokenButton = (s, i) => (
    <button
      key={i}
      role="listitem"
      className={`response-token ${i === index ? "picked" : ""} ${s.probability < 0.3 ? "unsure" : ""}`}
      style={{ "--p": s.probability }}
      title={`token ${i}: “${tokenText(s.text)}” ${pct(s.probability)} · ${s.entropy_bits.toFixed(2)} bits${s.lens?.settled_layer !== undefined && s.lens?.settled_layer !== null ? ` · settled at ${s.lens.settled_layer === 0 ? "the embeddings" : `layer ${s.lens.settled_layer - 1}`}` : ""}`}
      onClick={() => setPicked(i)}
    >
      {s.text === "" ? "∅" : s.text}
    </button>
  );
  return (
    <div className="response">
      <div className="prediction-head">
        <h2>
          What it wrote
          <small>
            greedy · {steps.length} token{steps.length === 1 ? "" : "s"} ·
            stopped:{" "}
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
      <div className="response-text" role="list" aria-label="Generated tokens">
        <span className="response-prompt">
          {(trace.tokens || []).map((t) => t.text).join("")}
        </span>
        {segments.thinking ? (
          <>
            {steps.slice(0, segments.thinking[0]).map(tokenButton)}
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
                steps
                  .slice(segments.thinking[0], segments.thinking[1])
                  .map((s, k) => tokenButton(s, segments.thinking[0] + k))}
            </span>
            {segments.closed && (
              <span className="response-segment answer">
                <span className="response-segment-label">answer</span>
                {steps
                  .slice(segments.thinking[1])
                  .map((s, k) => tokenButton(s, segments.thinking[1] + k))}
              </span>
            )}
          </>
        ) : (
          steps.map(tokenButton)
        )}
      </div>
      <p className="micro">
        Click a token. Shade is the probability the model gave it; tokens under
        30% are outlined. The prompt is shown plain.
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
