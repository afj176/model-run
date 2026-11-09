import { useMemo } from "react";
import { headCensus, ROLES } from "./censusData";

const ROLE_COLOUR = {
  prev: [157, 143, 240],
  sink: [240, 179, 90],
  self: [111, 179, 239],
  focused: [127, 217, 194],
  diffuse: [120, 112, 150],
  mixed: [84, 72, 140],
  unknown: [70, 70, 78],
};
const cell = (role, focus) => {
  const [r, g, b] = ROLE_COLOUR[role] ?? ROLE_COLOUR.unknown;
  const t = 0.35 + 0.65 * Math.max(0, Math.min(1, focus));
  return `rgb(${Math.round(43 + (r - 43) * t)},${Math.round(36 + (g - 36) * t)},${Math.round(82 + (b - 82) * t)})`;
};

/**
 * Every head of the capture on one card: layers down, heads across, coloured by habit and
 * brightened by focus, ringed by ablation effect where a sweep ran. A click opens that head.
 */
export default function HeadCensus({ trace, layer, head, onPick }) {
  const census = useMemo(() => headCensus(trace), [trace]);
  if (!census?.layers.length) return null;
  const H = Math.max(1, census.headCount || 1);
  const cols = H;
  const withHeads = census.layers.some((l) => l.heads);
  const roles = [
    ...new Set(
      census.layers.flatMap((l) => (l.heads || [l.mean]).map((h) => h.role)),
    ),
  ];
  return (
    <section className="head-census" aria-label="Head census">
      <h3>
        Every head{" "}
        <small>
          {census.layers.length} of {census.layerCount ?? census.layers.length}{" "}
          layers
          {withHeads
            ? census.layers.every((l) => l.heads)
              ? ` × ${H} heads`
              : ` · ${H} heads were kept, the head mean elsewhere`
            : " · head means"}
        </small>
      </h3>
      <div
        className="head-census-grid"
        style={{
          gridTemplateColumns: `34px repeat(${withHeads ? cols : 1}, minmax(0, 1fr))`,
        }}
        role="grid"
      >
        {census.layers.map((l) => (
          <div
            key={l.index}
            className="head-census-row"
            role="row"
            style={{ display: "contents" }}
          >
            <span
              className={`head-census-layer ${l.index === layer ? "current" : ""}`}
              role="rowheader"
            >
              L{l.index}
            </span>
            {l.heads ? (
              l.heads.map((h) => (
                <button
                  key={h.head}
                  role="gridcell"
                  className={`head-census-cell ${l.index === layer && h.head === head ? "picked" : ""}`}
                  style={{ background: cell(h.role, h.focus) }}
                  title={`L${l.index} · H${h.head} · ${h.roleLabel} · focus ${Math.round(h.focus * 100)}%${h.effect !== null ? ` · ablation ${h.effect.toFixed(2)} bits` : ""}`}
                  onClick={() => onPick?.(l.index, h.head)}
                >
                  {h.effect !== null && h.effect >= 0.1 && (
                    <i
                      style={{
                        width: `${Math.min(100, 30 + h.effect * 40)}%`,
                        height: `${Math.min(100, 30 + h.effect * 40)}%`,
                      }}
                    />
                  )}
                </button>
              ))
            ) : (
              <button
                role="gridcell"
                className={`head-census-cell mean ${l.index === layer ? "picked" : ""}`}
                style={{
                  gridColumn: "2 / -1",
                  background: cell(l.mean.role, l.mean.focus),
                }}
                title={`L${l.index} · head mean · ${l.mean.roleLabel} · focus ${Math.round(l.mean.focus * 100)}%`}
                onClick={() => onPick?.(l.index, head)}
              >
                head mean
              </button>
            )}
          </div>
        ))}
      </div>
      <div className="head-census-legend">
        {roles.map((r) => (
          <span key={r}>
            <i style={{ background: cell(r, 1) }} />
            {Object.values(ROLES).find((x) => x.key === r)?.label ?? r}
          </span>
        ))}
        <span className="head-census-note">
          brighter = more focused
          {census.layers.some((l) => l.swept)
            ? " · ring = ablation effect"
            : ""}
        </span>
      </div>
      {census.strongest.length > 0 && (
        <p className="micro">
          Largest ablation effects:{" "}
          {census.strongest.map((h, i) => (
            <button
              key={`${h.layer}-${h.head}`}
              className="text-button"
              onClick={() => onPick?.(h.layer, h.head)}
            >
              L{h.layer}·H{h.head} {h.effect.toFixed(2)} bits
              {i < census.strongest.length - 1 ? "," : ""}
            </button>
          ))}
        </p>
      )}
    </section>
  );
}
