import { useEffect, useState } from "react";
import { CaretLeft, CaretRight, Pause, Play } from "@phosphor-icons/react";
import { axisKey, axisLabel, resolveSlice } from "./matrixData";

const num = (v) => (Number.isFinite(v) ? v.toPrecision(3) : "non-finite");
const FRAME_MS = 450;

/** One axis: a bar per recorded index, sized by the mean magnitude measured over that index. */
function AxisStrip({ axis, position, allowAll, onPick }) {
  const peak = Math.max(
    ...axis.mean_abs.filter((v) => Number.isFinite(v)),
    1e-12,
  );
  const count = axis.indices.length;
  const name = axisLabel(axis);
  const step = (delta) => {
    const from = position ?? (delta > 0 ? -1 : count);
    onPick(axis.indices[(from + delta + count) % count]);
  };
  return (
    <div className="slice-axis">
      <span className="slice-axis-name">
        <b>{name}</b>
        <small>
          {position === null
            ? `all ${axis.size.toLocaleString()}`
            : `${axis.indices[position].toLocaleString()} of ${axis.size.toLocaleString()}`}
          {count < axis.size ? ` · ${count} kept` : ""}
        </small>
      </span>
      {allowAll && (
        <button
          className={position === null ? "slice-all selected" : "slice-all"}
          aria-pressed={position === null}
          title={`Mean magnitude across every ${name}`}
          onClick={() => onPick("all")}
        >
          All
        </button>
      )}
      <button
        className="slice-step"
        aria-label={`Previous ${name}`}
        onClick={() => step(-1)}
      >
        <CaretLeft size={13} />
      </button>
      <div className="slice-bars" role="group" aria-label={`${name} index`}>
        {axis.indices.map((index, i) => (
          <button
            key={index}
            className={i === position ? "selected" : ""}
            aria-pressed={i === position}
            aria-label={`${name} ${index}`}
            title={`${name} ${index}: mean |value| ${num(axis.mean_abs[i])}, peak ${num(axis.max_abs[i])}`}
            onClick={() => onPick(index)}
          >
            <i
              style={{
                height: `${Math.max(8, ((axis.mean_abs[i] ?? 0) / peak) * 100)}%`,
              }}
            />
          </button>
        ))}
      </div>
      <button
        className="slice-step"
        aria-label={`Next ${name}`}
        onClick={() => step(1)}
      >
        <CaretRight size={13} />
      </button>
    </div>
  );
}

/** Moves through a tensor's recorded frames and channels. Shared by both stages. */
export default function SliceNavigator({ event, wanted, onChange }) {
  const slices = event?.slices;
  const [playing, setPlaying] = useState(false);
  const axes = slices?.axes;
  const usable = Array.isArray(axes);
  const positions = usable ? resolveSlice(slices, wanted) : [];
  const frameAt = usable ? axes.findIndex((a) => a.name === "frame") : -1;
  const frame = frameAt >= 0 ? axes[frameAt] : null;
  const framePosition = frameAt >= 0 ? positions[frameAt] : 0;
  useEffect(() => {
    if (!playing || !frame) return undefined;
    const timer = setTimeout(() => {
      const next = (framePosition + 1) % frame.indices.length;
      onChange(axisKey(frame), frame.indices[next]);
    }, FRAME_MS);
    return () => clearTimeout(timer);
  }, [playing, frame, framePosition, onChange]);
  if (slices?.unavailable)
    return <p className="slice-nav slice-note">{slices.unavailable}</p>;
  if (!usable || !axes.length) return null;
  return (
    <div className="slice-nav" aria-label="Tensor slices">
      {frame && frame.indices.length > 1 && !event.volume_layout && (
        <button
          className="slice-play"
          aria-label={playing ? "Pause frames" : "Play frames"}
          aria-pressed={playing}
          onClick={() => setPlaying((p) => !p)}
        >
          {playing ? (
            <Pause size={13} weight="fill" />
          ) : (
            <Play size={13} weight="fill" />
          )}
        </button>
      )}
      {axes.map((axis, i) =>
        axisKey(axis) === event.volume_layout?.axisKey ? null : (
          <AxisStrip
            key={axisKey(axis)}
            axis={axis}
            position={positions[i]}
            allowAll={i === slices.aggregate_axis}
            onPick={(value) => onChange(axisKey(axis), value)}
          />
        ),
      )}
    </div>
  );
}
