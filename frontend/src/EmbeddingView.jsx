import { tokenText } from "./insights";

const POOLING = {
  mean_tokens: "mean of tokens",
  cls_token: "first token",
  max_tokens: "max over tokens",
  lasttoken: "last token",
};

/** The sentence vector an embedding model made, and which tokens point its way. */
export default function EmbeddingView({ trace }) {
  const record = trace?.embedding;
  if (!record) return null;
  const values = record.values || [];
  const peak = Math.max(1e-12, ...values.map((v) => Math.abs(v)));
  const tokens = trace.tokens || [];
  const alignment = record.token_alignment || [];
  const shown = values.slice(0, 512);
  const steps = [
    POOLING[record.pooling] || record.pooling,
    ...record.dense_layers.map(([out, inp]) => `dense ${inp}→${out}`),
    ...(record.normalized ? ["normalized"] : []),
  ];
  const prompts = Object.entries(record.prompts || {});
  return (
    <div className="embedding-view">
      <div className="image-classification-head">
        <h2>
          Sentence vector
          <small>
            {record.dimension.toLocaleString()} values · {steps.join(" · ")} ·
            length {record.norm_before_normalize.toFixed(2)} before
            normalizing
          </small>
        </h2>
      </div>
      <div
        className="embedding-values"
        role="img"
        aria-label={`First ${shown.length} of ${record.dimension} values; largest magnitude ${peak.toFixed(3)}`}
      >
        {shown.map((v, i) => (
          <i
            key={i}
            title={`Value ${i}: ${v.toFixed(4)}`}
            className={v < 0 ? "negative" : undefined}
            style={{ height: `${(Math.abs(v) / peak) * 50}%` }}
          />
        ))}
      </div>
      <small className="muted">
        {shown.length < record.dimension
          ? `First ${shown.length} of ${record.dimension} values, `
          : ""}
        scaled to the largest magnitude, {peak.toFixed(3)}. Above the line is
        positive.
      </small>
      <h3>Tokens that point the way the sentence does</h3>
      <ol className="image-labels embedding-tokens">
        {alignment.map((a, i) => (
          <li key={i}>
            <span title={`Position ${i}`}>{tokenText(tokens[i]?.text)}</span>
            <b>{a.toFixed(2)}</b>
            <i
              className={a < 0 ? "negative" : undefined}
              style={{ width: `${Math.abs(a) * 100}%` }}
            />
          </li>
        ))}
      </ol>
      {prompts.length > 0 && (
        <small className="muted">
          This package suggests prefixes for its tasks, such as{" "}
          {prompts
            .slice(0, 2)
            .map(([name, text]) => `${name}: "${text}"`)
            .join(", ")}
          . The text above was embedded exactly as typed.
        </small>
      )}
    </div>
  );
}
