import { useMemo, useState } from "react";
import { Lightbulb } from "@phosphor-icons/react";
import { lensStory, pct, tokenLabel, tokenText as show } from "./insights";
import { ResidualStream } from "./TextPanels";
import LensRiver from "./LensRiver";

/** What the model predicted next, how sure it was, and how that answer formed layer by layer. */
export default function Prediction({ trace }) {
  const [lensLayer, setLensLayer] = useState(null);
  const top = trace?.output?.top_tokens || [];
  const lens = trace?.logit_lens;
  const names = useMemo(() => {
    const map = new Map();
    for (const t of top) map.set(t.token_id, t.text);
    for (const layer of lens?.layers || [])
      for (const t of layer.top_tokens)
        if (t.text !== undefined) map.set(t.token_id, t.text);
    return map;
  }, [top, lens]);
  if (!top.length) return null;
  const story = lensStory(lens, names);
  const entropy = trace.output.entropy_bits;
  const first = top[0];
  const lastToken = trace.tokens?.at(-1);
  const confidence =
    first.probability >= 0.6
      ? "confident"
      : first.probability >= 0.3
        ? "leaning"
        : "undecided";
  const picked = lens?.layers?.find((l) => l.index === lensLayer) || null;
  return (
    <div className="prediction">
      <div className="prediction-head">
        <h2>
          What it predicted next
          <small>
            Raw softmax · after “
            {lastToken ? tokenLabel(lastToken) : "last token"}”
          </small>
        </h2>
        <span className={`prediction-confidence ${confidence}`}>
          {confidence}
          {Number.isFinite(entropy) && (
            <small>{entropy.toFixed(2)} bits of uncertainty</small>
          )}
        </span>
      </div>
      <p className="prediction-story">
        <Lightbulb size={16} weight="duotone" /> The model puts{" "}
        <b>{pct(first.probability)}</b> on “{show(first.text)}”
        {top[1] && (
          <>
            , then {pct(top[1].probability)} on “{show(top[1].text)}”
          </>
        )}
        .{" "}
        {story?.settledLabel &&
          (story.settledIndex === 0
            ? " The embeddings alone already favoured this token."
            : story.wavered
              ? ` It first surfaced as the top guess at ${story.firstLabel}, lost the lead, and held it from ${story.settledLabel} of ${story.layersTotal - 1} to the output.`
              : ` It first became the top guess at ${story.settledLabel} of ${story.layersTotal - 1} and held through to the output.`)}
        {story && story.settledIndex === null && (
          <>
            {" "}
            No earlier layer ranked this token first; the final layer decided
            it.
          </>
        )}
      </p>
      <ol className="prediction-list">
        {top.slice(0, 5).map((t, i) => (
          <li key={t.token_id}>
            <span className="prediction-rank">{i + 1}</span>
            <span className="prediction-token" title={`Token ${t.token_id}`}>
              {show(t.text)}
            </span>
            <span className="prediction-bar">
              <i style={{ width: `${Math.max(1, t.probability * 100)}%` }} />
            </span>
            <b>{pct(t.probability)}</b>
          </li>
        ))}
      </ol>
      {lens?.layers?.length > 0 && (
        <div className="lens">
          <h3>
            How the answer formed{" "}
            <small>logit lens · last position · top token per layer</small>
          </h3>
          <LensRiver
            lens={lens}
            names={names}
            picked={lensLayer}
            onPick={(index) => setLensLayer(lensLayer === index ? null : index)}
          />
          <p className="micro">
            Each lane is a candidate that reaches some layer's top five; at each
            layer the lanes share the height by probability, so the widest lane
            is that layer's favourite. The pink lane is the answer. Click a
            layer to see its top five below.
          </p>
          <div className="lens-track" role="list">
            {lens.layers.map((layer) => {
              const guess = layer.top_tokens[0];
              const isFinal = guess?.token_id === lens.final_token_id;
              return (
                <button
                  key={layer.index}
                  role="listitem"
                  className={`lens-cell ${isFinal ? "final" : ""} ${lensLayer === layer.index ? "picked" : ""}`}
                  style={{ "--p": layer.final_token_probability }}
                  title={`${layer.label}: top “${show(names.get(guess?.token_id) ?? guess?.text)}” ${pct(guess?.probability)} · final token ${pct(layer.final_token_probability)} · ${layer.entropy_bits.toFixed(2)} bits`}
                  onClick={() =>
                    setLensLayer(lensLayer === layer.index ? null : layer.index)
                  }
                >
                  <i />
                  <span>{show(names.get(guess?.token_id) ?? guess?.text)}</span>
                  <small>
                    {layer.index === 0 ? "emb" : `L${layer.index - 1}`}
                  </small>
                </button>
              );
            })}
          </div>
          <p className="micro">
            Bar height is the probability of the final answer at that layer
            (from {pct(story?.firstProbability ?? 0)} to{" "}
            {pct(story?.finalProbability ?? 0)}). Pink cells already rank the
            final token first. Most uncertain at {story?.uncertainLabel}.
          </p>
          <ResidualStream lens={lens} />
          {picked && (
            <div className="lens-detail">
              <b>{picked.label}</b>
              <span>
                residual norm {picked.residual_norm.toFixed(1)} ·{" "}
                {picked.entropy_bits.toFixed(2)} bits
              </span>
              <div className="lens-detail-tokens">
                {picked.top_tokens.map((t) => (
                  <span key={t.token_id}>
                    “{show(names.get(t.token_id) ?? t.text)}”{" "}
                    {pct(t.probability)}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
