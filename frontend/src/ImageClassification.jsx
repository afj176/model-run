import { useEffect, useMemo, useState } from "react";
import { imageAttentionStory, pct, pctFine, validIndex } from "./insights";

/** What an image model predicted, and where each layer's class token looked on the input. */
export default function ImageClassification({
  trace,
  stageLayer = null,
  onLayer,
}) {
  const record = trace?.image_classification;
  const layers = record?.attention_layers || [];
  const [picked, setPicked] = useState(null);
  useEffect(() => setPicked(null), [trace]);
  // The layer the stage shows is the layer whose attention this shows.
  useEffect(() => {
    const i = layers.findIndex((l) => l.layer === stageLayer);
    if (i >= 0) setPicked(i);
  }, [stageLayer, layers]);
  const index = validIndex(picked, layers.length, layers.length - 1);
  const layer = layers[index];
  const story = useMemo(
    () => (layer ? imageAttentionStory(layer.patches, record.grid) : null),
    [layer, record],
  );
  if (!record) return null;
  const [rows, cols] = record.grid || [1, 1];
  const peak = story?.peak || 1;
  const first = record.top_labels[0];
  return (
    <div className="image-classification">
      <div className="image-classification-head">
        <h2>
          What it saw
          <small>
            Raw softmax over {record.class_count?.toLocaleString() ?? "?"}{" "}
            classes · {record.entropy_bits.toFixed(2)} bits of uncertainty
          </small>
        </h2>
      </div>
      <div className="image-classification-body">
        <div className="image-attention">
          {record.input_image ? (
            <img
              alt="Input as the model received it"
              src={`data:${record.input_image.mime};base64,${record.input_image.base64}`}
            />
          ) : (
            <div className="image-attention-missing">No input image kept</div>
          )}
          {layer && (
            <div
              className="image-attention-grid"
              style={{
                gridTemplateColumns: `repeat(${cols}, 1fr)`,
                gridTemplateRows: `repeat(${rows}, 1fr)`,
              }}
              aria-hidden="true"
            >
              {layer.patches.map((v, i) => (
                <i
                  key={i}
                  style={{ opacity: Math.min(0.85, ((v ?? 0) / peak) * 0.85) }}
                />
              ))}
            </div>
          )}
        </div>
        <div className="image-classification-side">
          <ol className="image-labels">
            {record.top_labels.map((entry) => (
              <li key={entry.index}>
                <span title={entry.label}>{entry.label}</span>
                <b>{pctFine(entry.probability)}</b>
                <i
                  style={{
                    width: `${(entry.probability / first.probability) * 100}%`,
                  }}
                />
              </li>
            ))}
          </ol>
          {layers.length > 0 ? (
            <>
              <label className="image-layer">
                Layer {layer.layer}
                <input
                  type="range"
                  min="0"
                  max={layers.length - 1}
                  value={index}
                  aria-label="Attention layer"
                  onChange={(e) => {
                    const i = Number(e.target.value);
                    setPicked(i);
                    onLayer?.(layers[i]?.layer);
                  }}
                />
              </label>
              <p className="muted">
                The class token puts {pct(story.top4Share)} of its patch
                attention on the 4 strongest of {story.count} patches, peaking
                at row {story.row}, column {story.col}. It keeps{" "}
                {pct(layer.self ?? 0)} on itself.
              </p>
            </>
          ) : (
            <p className="muted">{record.attention_unavailable}</p>
          )}
        </div>
      </div>
      <p className="micro">{record.limitations?.join(" ")}</p>
    </div>
  );
}
