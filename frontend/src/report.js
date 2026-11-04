/** Self-contained HTML insight report built from one trace. Every number is read
 * from the recorded trace; heatmaps are drawn from recorded values. */
import {
  ablationStory,
  responseStory,
  patchStory,
  activationStats,
  attentionHeadProfile,
  callDeltas,
  crossAttentionStory,
  describeActivation,
  layerSeries,
  lensStory,
  pct,
  rowSummary,
  tokenLabel,
  tokenText,
} from "./insights.js";

const esc = (s) =>
  String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
const num = (n, d = 3) => (Number.isFinite(n) ? n.toFixed(d) : "—");

const STOPS = [
  [242, 239, 228],
  [127, 217, 194],
  [15, 159, 134],
];
function mixRgb(t) {
  const p = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  const [a, b] = t < 0.5 ? [STOPS[0], STOPS[1]] : [STOPS[1], STOPS[2]];
  return a.map((v, i) => Math.round(v + (b[i] - v) * p));
}

/** Heatmap as a PNG data URL when a canvas exists, otherwise inline SVG. */
export function heatmap(
  rows,
  { width = 160, signed = false, title = "" } = {},
) {
  const n = rows.length,
    m = rows[0]?.length || 0;
  if (!n || !m) return "";
  let peak = 1e-9;
  for (const row of rows)
    for (const v of row)
      if (Number.isFinite(v)) peak = Math.max(peak, Math.abs(v));
  const color = (v) => {
    if (!Number.isFinite(v)) return [60, 56, 90];
    if (!signed) return mixRgb(Math.min(1, Math.max(0, v) / peak));
    const t = Math.min(1, Math.abs(v) / peak);
    return v >= 0
      ? [
          Math.round(242 - 227 * t),
          Math.round(239 - 80 * t),
          Math.round(228 - 94 * t),
        ]
      : [
          Math.round(242 + 13 * t),
          Math.round(239 - 140 * t),
          Math.round(228 - 105 * t),
        ];
  };
  const height = Math.round((width * n) / m);
  if (typeof document !== "undefined" && document.createElement) {
    const canvas = document.createElement("canvas");
    canvas.width = m;
    canvas.height = n;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      const img = ctx.createImageData(m, n);
      rows.forEach((row, r) =>
        row.forEach((v, c) => {
          const [R, G, B] = color(v);
          const at = (r * m + c) * 4;
          img.data[at] = R;
          img.data[at + 1] = G;
          img.data[at + 2] = B;
          img.data[at + 3] = 255;
        }),
      );
      ctx.putImageData(img, 0, 0);
      return `<img class="heat" width="${width}" height="${height}" alt="${esc(title)}" title="${esc(title)}" src="${canvas.toDataURL("image/png")}">`;
    }
  }
  const cells = rows
    .map((row, r) =>
      row
        .map((v, c) => {
          const [R, G, B] = color(v);
          return `<rect x="${c}" y="${r}" width="1" height="1" fill="rgb(${R},${G},${B})"/>`;
        })
        .join(""),
    )
    .join("");
  return `<svg class="heat" width="${width}" height="${height}" viewBox="0 0 ${m} ${n}" shape-rendering="crispEdges" role="img" aria-label="${esc(title)}">${cells}</svg>`;
}

function bars(items, { max = null, unit = "" } = {}) {
  const peak = max ?? Math.max(1e-9, ...items.map((i) => i.value));
  return `<ul class="bars">${items
    .map(
      (i) =>
        `<li><span class="bar-label">${esc(i.label)}</span><span class="bar"><i style="width:${Math.max(1, (i.value / peak) * 100)}%"></i></span><b>${esc(i.text ?? num(i.value, 3))}${unit}</b></li>`,
    )
    .join("")}</ul>`;
}

function sparkline(points, label) {
  if (!points?.length) return "";
  const values = points.map((p) => p.value).filter(Number.isFinite);
  const lo = Math.min(...values),
    hi = Math.max(...values);
  const w = 320,
    h = 48;
  const path = points
    .map((p, i) => {
      const x = (i / Math.max(1, points.length - 1)) * w;
      const y = h - 4 - ((p.value - lo) / Math.max(1e-9, hi - lo)) * (h - 8);
      return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return `<figure class="spark"><figcaption>${esc(label)} · min ${num(lo)} · max ${num(hi)}</figcaption><svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><path d="${path}"/></svg><div class="spark-axis"><span>${esc(points[0].layer)}</span><span>${esc(points.at(-1).layer)}</span></div></figure>`;
}

function section(title, body, note = "") {
  if (!body) return "";
  return `<section><h2>${esc(title)}</h2>${body}${note ? `<p class="note">${esc(note)}</p>` : ""}</section>`;
}

const show = (t) => esc(tokenText(t));

function attentionSections(trace) {
  const out = [];
  const tokens = trace.tokens;
  const labelAt = (i) => (tokens?.[i] ? tokenLabel(tokens[i], i) : String(i));
  const profile = attentionHeadProfile(trace.values);
  const lastRow = trace.values.length - 1;
  const attends = rowSummary(trace.values, lastRow, tokens, 4);
  out.push(
    section(
      `Layer ${trace.layer} · head ${trace.head}: ${profile.label}`,
      `<div class="split">${heatmap(trace.values, { width: 240, title: `Attention L${trace.layer} H${trace.head}` })}<div><p>${esc(profile.description)}</p><p class="chips"><span>sink ${pct(profile.sinkMass)}</span><span>prev-token ${pct(profile.previousTokenMass)}</span><span>self ${pct(profile.diagonalMass)}</span><span>${num(profile.entropyBits, 2)} bits</span></p>${
        attends.length
          ? `<p class="sub">“${esc(labelAt(lastRow))}” attends to</p>${bars(
              attends.map((a) => ({
                label: tokens ? `“${a.label}”` : `key ${a.key}`,
                value: a.weight,
                text: pct(a.weight),
              })),
              { max: 1 },
            )}`
          : ""
      }</div></div>`,
      "Rows are query positions, columns key positions; weights above the diagonal are zero under the causal mask.",
    ),
  );
  const received = trace.attribution?.last_position_attention;
  if (Array.isArray(received)) {
    const peak = Math.max(1e-9, ...received);
    out.push(
      section(
        "What the last token looked at",
        `<p class="strip">${received
          .map(
            (v, i) =>
              `<span style="--heat:${(v / peak).toFixed(3)}">${esc(labelAt(i))}<small>${pct(v)}</small></span>`,
          )
          .join("")}</p>`,
        `Measured attention from the final query position to each key, averaged over all ${trace.layer_count ?? ""} layers and ${trace.head_count ?? ""} heads. Attention is not a causal attribution.`,
      ),
    );
  }
  const ablation = ablationStory(trace.ablation, trace.head);
  if (ablation) {
    const body = `${ablation.text ? `<p>${esc(ablation.text)}</p>` : ""}<table><tr><th></th><th>Top token</th><th>Probability</th></tr><tr><td>With head ${trace.head}</td><td>“${show(ablation.baseline.text)}”</td><td>${pct(ablation.baseline.probability)}</td></tr>${
      ablation.entry
        ? `<tr><td>Without head ${trace.head}</td><td>“${show(ablation.entry.top_tokens[0].text)}”</td><td>${pct(ablation.entry.top_tokens[0].probability)}</td></tr>`
        : ""
    }</table>${
      ablation.ranked.length > 1
        ? `<p class="sub">Heads of layer ${trace.layer} ranked by shift (bits)</p>${bars(ablation.ranked.slice(0, 12).map((h) => ({ label: `H${h.head}${h.top_token_changed ? " · flips" : ""}`, value: h.kl_bits, text: num(h.kl_bits, 3) })))}`
        : ""
    }`;
    out.push(
      section(
        "What happens without this head",
        body,
        trace.ablation.limitations?.join(" "),
      ),
    );
  }
  if (Array.isArray(trace.heads)) {
    out.push(
      section(
        `All ${trace.heads.length} heads of layer ${trace.layer}`,
        `<div class="gallery">${trace.heads
          .map((m, i) => {
            const p = attentionHeadProfile(m);
            return `<figure class="${i === trace.head ? "picked" : ""}">${heatmap(m, { width: 96, title: `Head ${i}` })}<figcaption>H${i}<br><small>${esc(p?.label)}</small></figcaption></figure>`;
          })
          .join("")}</div>`,
      ),
    );
  }
  const series = layerSeries(trace, []);
  if (series)
    out.push(
      section(
        "Across layers",
        sparkline(series.points, series.label) +
          (series.secondary
            ? sparkline(series.secondary.points, series.secondary.label)
            : ""),
      ),
    );
  return out.join("");
}

function predictionSection(trace) {
  const top = trace.output?.top_tokens;
  if (!top?.length) return "";
  const names = new Map();
  for (const t of top) names.set(t.token_id, t.text);
  for (const l of trace.logit_lens?.layers || [])
    for (const t of l.top_tokens)
      if (t.text !== undefined) names.set(t.token_id, t.text);
  const story = lensStory(trace.logit_lens, names);
  const first = top[0];
  let text = `The model puts ${pct(first.probability)} on “${tokenText(first.text)}”${top[1] ? `, then ${pct(top[1].probability)} on “${tokenText(top[1].text)}”` : ""}.`;
  if (story?.settledLabel)
    text +=
      story.settledIndex === 0
        ? " The embeddings alone already favoured this token."
        : story.wavered
          ? ` It first surfaced at ${story.firstLabel}, lost the lead, and held it from ${story.settledLabel} to the output.`
          : ` It first became the top guess at ${story.settledLabel} of ${story.layersTotal - 1} and held through to the output.`;
  let body = `<p>${esc(text)}${Number.isFinite(trace.output.entropy_bits) ? ` <span class="muted">${num(trace.output.entropy_bits, 2)} bits of uncertainty.</span>` : ""}</p>${bars(
    top.slice(0, 5).map((t) => ({
      label: `“${tokenText(t.text)}”`,
      value: t.probability,
      text: pct(t.probability),
    })),
    { max: 1 },
  )}`;
  const lens = trace.logit_lens?.layers;
  if (lens?.length) {
    body += `<p class="sub">How the answer formed (logit lens, last position)</p><div class="lens">${lens
      .map((l) => {
        const g = l.top_tokens[0];
        const final = g?.token_id === trace.logit_lens.final_token_id;
        return `<div class="${final ? "final" : ""}" title="${esc(l.label)}"><i style="height:${Math.round(6 + l.final_token_probability * 70)}px"></i><span>${show(names.get(g?.token_id) ?? g?.text)}</span><small>${l.index === 0 ? "emb" : `L${l.index - 1}`}</small></div>`;
      })
      .join(
        "",
      )}</div><p class="note">Bar height is the probability of the final answer at that layer (${pct(story?.firstProbability)} → ${pct(story?.finalProbability)}). Pink cells already rank the final token first. ${esc(trace.logit_lens.limitations?.join(" ") || "")}</p>`;
  }
  return section("What it predicted next", body);
}

function activationSections(trace) {
  const events = (trace.events || []).filter(
    (e) => e.status === "captured" || e.values,
  );
  if (!events.length) return "";
  const rows = events.slice(0, 48).map((e) => {
    const s = activationStats(e.values);
    const name = e.module ?? (e.step !== undefined ? `step ${e.step}` : "");
    return `<tr><td>${esc(name)}${e.denoiser_call !== undefined ? ` <small>call ${e.denoiser_call}</small>` : ""}</td><td>${esc((e.shape || []).join("×"))}</td><td>${num(s?.rms)}</td><td>${num(s?.outlierRatio, 1)}×</td><td>${pct(s?.sparsity)}</td><td class="text">${esc(describeActivation(s, null).replace(/^This tensor /, ""))}</td></tr>`;
  });
  let body = `<table><tr><th>Module</th><th>Shape</th><th>RMS</th><th>Peak / RMS</th><th>Near zero</th><th>Reading</th></tr>${rows.join("")}</table>`;
  const deltas = events.some(
    (e) => e.denoiser_call !== undefined || e.step !== undefined,
  )
    ? callDeltas(events).filter((d) => d.delta !== null)
    : [];
  if (deltas.length)
    body += `<p class="sub">Change between consecutive calls (RMS of the difference)</p>${bars(deltas.slice(0, 32).map((d) => ({ label: `${d.module ? d.module.split(".").slice(-2).join(".") + " " : ""}call ${d.ordinal}`, value: d.delta, text: num(d.delta) })))}`;
  const series = layerSeries(trace, events);
  if (series) body += sparkline(series.points, series.label);
  const first = events[0];
  body += `<div class="split">${heatmap(toMatrix(first.values), { width: 240, signed: true, title: "First captured tensor, flattened prefix" })}<p class="note">First captured prefix (${esc(first.module ?? "step " + first.step)}) folded into a square; teal positive, pink negative, grey non-finite. Flattened prefixes are not representative samples.</p></div>`;
  return section("Activations", body);
}

function toMatrix(values, limit = 4096) {
  const v = (values || []).slice(0, limit);
  const m = Math.max(1, Math.ceil(Math.sqrt(v.length)));
  const rows = [];
  for (let r = 0; r * m < v.length; r++)
    rows.push(
      Array.from({ length: m }, (_, c) =>
        r * m + c < v.length ? v[r * m + c] : null,
      ),
    );
  return rows;
}

function crossAttentionSection(trace) {
  const ca = trace.cross_attention;
  if (!ca) return "";
  const event = ca.events?.[0];
  if (!event)
    return section(
      "Where the prompt is read",
      `<p>No attention call matched ${esc(ca.selection)}. ${esc(ca.skipped?.[0]?.reason || "")}</p>`,
    );
  const labels = ca.text_labels || null;
  const story = crossAttentionStory(event, labels);
  const isPad = (i) =>
    labels?.[i] !== undefined &&
    /^<\|?(endoftext|pad)\|?>$/i.test(labels[i]) &&
    i > 0 &&
    labels[i - 1] === labels[i];
  const ranked = story.ranked.filter((r) => !isPad(r.index)).slice(0, 6);
  const [rows, cols] = event.grid;
  const grid = (values) =>
    Array.from({ length: rows }, (_, r) =>
      values.slice(r * cols, (r + 1) * cols),
    );
  const label = (i) => (labels?.[i] ? `“${tokenText(labels[i])}”` : `#${i}`);
  const maps = ranked
    .map((r) => {
      const f = crossAttentionStory(event, labels, r.index);
      return `<figure>${heatmap(grid(event.maps[r.index]), { width: 150, title: label(r.index) })}<figcaption>${esc(label(r.index))}<br><small>peak ${pct(f.peak)} · ${esc(f.where)}</small></figcaption></figure>`;
    })
    .join("");
  let body = `<p>${esc(story.text)}</p><p class="muted">${esc(ca.selection)} · ${event.attention_kind} attention · ${event.heads} heads averaged · ${event.text_tokens} text tokens · ${event.source_grid[0]}×${event.source_grid[1]} image tokens pooled to ${rows}×${cols} · ${esc(event.batch)} · denoiser call ${event.denoiser_call}${ca.events.length > 1 ? ` of ${ca.events.length} recorded` : ""}</p><div class="gallery">${maps}</div>`;
  body += `<p class="sub">Tokens by how much the image reads them</p>${bars(ranked.map((r) => ({ label: label(r.index), value: r.mass, text: pct(r.mass) })))}`;
  if (event.attention_kind === "joint" && story.textShare !== null)
    body += `<div class="split">${heatmap(grid(event.text_mass), { width: 150, title: "Text share" })}<p class="note">Text share: how much of each region's attention goes to text at all (mean ${pct(story.textShare)}).</p></div>`;
  return section("Where the prompt is read", body, ca.limitations?.join(" "));
}

/** Build the report HTML for one trace. */
export function buildReport(trace, { name = "", generated = new Date() } = {}) {
  if (!trace) throw new Error("No trace to report");
  const title =
    name || trace.model_name || trace.model_class || "Model Run trace";
  const kind =
    {
      attention_capture: "Attention capture",
      pytorch_module_capture: "Module capture",
      diffusion_step_capture: "Diffusion step capture",
      head_ablation: "Head ablation",
    }[trace.kind] || trace.kind;
  const facts = [
    ["Kind", kind],
    ["Model class", trace.model_class],
    [
      "Source",
      trace.source || (trace.adapter_id ? `adapter ${trace.adapter_id}` : ""),
    ],
    ["Torch", trace.torch_version],
    trace.input?.token_count ? ["Input tokens", trace.input.token_count] : null,
    trace.sampling
      ? [
          "Sampling",
          `${trace.sampling.steps} steps · ${trace.sampling.size}px · seed ${trace.sampling.seed}`,
        ]
      : null,
    trace.captured_calls ? ["Denoiser calls", trace.captured_calls] : null,
    trace.token_ablation
      ? [
          "Ablated tokens",
          trace.token_ablation.labels
            .map((l) => `“${tokenText(l)}”`)
            .join(", "),
        ]
      : null,
    trace.status ? ["Status", trace.status] : null,
  ].filter((f) => f && f[1] !== undefined && f[1] !== "");
  let body = "";
  if (trace.tokens?.length)
    body += section(
      "Prompt tokens",
      `<p class="strip plain">${trace.tokens.map((t, i) => `<span title="position ${i} · id ${t.id}">${esc(t.text === "" ? "∅" : t.text)}</span>`).join("")}</p>`,
    );
  if (trace.preview?.mime === "image/png")
    body += section(
      "Generated output",
      `<img class="preview" alt="Decoded pipeline output" src="data:image/png;base64,${trace.preview.base64}">`,
    );
  if (trace.kind === "attention_capture") body += attentionSections(trace);
  body += predictionSection(trace);
  body += responseSection(trace);
  body += patchingSection(trace);
  body += crossAttentionSection(trace);
  if (trace.kind !== "attention_capture") body += activationSections(trace);
  const limitations = [...new Set(trace.limitations || [])];
  body += section(
    "Limitations recorded with this trace",
    `<ul class="plain-list">${limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`,
  );
  const when =
    generated instanceof Date
      ? generated.toISOString().replace("T", " ").slice(0, 16) + " UTC"
      : String(generated);
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)} · Model Run report</title>
<style>${CSS}</style></head>
<body><main>
<header><p class="eyebrow">Model Run · insight report</p><h1>${esc(title)}</h1><p class="lede">${esc(kind)} · generated ${esc(when)}. Every number below is read from recorded values; nothing is inferred from the architecture name.</p>
<dl class="facts">${facts.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl></header>
${body}
<footer>Generated by Model Run. Attention weights describe where information is read, not what caused an output; ablations are interventions. Keep the trace JSON alongside this report for the full data.</footer>
</main></body></html>`;
}

const CSS = `
:root{--ink:#1e1a46;--muted:#6b6489;--paper:#f6f3fc;--card:#fff;--line:#e3dbf5;--indigo:#3f31a3;--pink:#ff3d8f;--mint:#0f9f86}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 20px 48px}
header{margin-bottom:20px}.eyebrow{margin:0;font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--indigo);font-weight:700}
h1{margin:4px 0 6px;font-size:30px}h2{margin:0 0 10px;font-size:19px}.lede{margin:0 0 14px;color:var(--muted)}
.facts{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px 16px;margin:0}.facts div{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px}
dt{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}dd{margin:0;font-weight:600;word-break:break-word}
section{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin:14px 0}
p{margin:6px 0}.sub{margin:14px 0 6px;font-weight:600;color:#514a77;font-size:13px}.note,.muted{color:var(--muted);font-size:12.5px}
.split{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}.split>div{flex:1;min-width:240px}
.heat{image-rendering:pixelated;border-radius:8px;background:#1a1437;display:block}
.chips span{display:inline-block;font-size:12px;padding:3px 9px;border-radius:999px;background:#ece7fa;color:#3a3270;margin:2px 4px 2px 0}
.bars{list-style:none;margin:0;padding:0;max-width:560px}.bars li{display:grid;grid-template-columns:minmax(90px,170px) 1fr 60px;gap:10px;align-items:center;font-size:13px;margin:3px 0}
.bar-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:Consolas,"Cascadia Mono",monospace}.bar{display:block;height:11px;border-radius:6px;background:#ece7fa;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,#7fd9c2,var(--mint));border-radius:6px}.bars b{text-align:right}
.strip{display:flex;flex-wrap:wrap;gap:4px}.strip span{--heat:0;display:inline-flex;flex-direction:column;align-items:center;padding:4px 7px 3px;border-radius:7px;font-size:13px;background:color-mix(in srgb,var(--mint) calc(var(--heat)*85%),#f2efe4);border:1px solid #0f9f8633}
.strip span small{font-size:10px;opacity:.8}.strip.plain span{background:#ece7fa;border-color:transparent;white-space:pre}
.gallery{display:flex;flex-wrap:wrap;gap:12px}.gallery figure{margin:0;text-align:center;font-size:12px;padding:6px;border:1px solid var(--line);border-radius:10px}.gallery figure.picked{border-color:var(--pink);box-shadow:0 0 0 2px #ff3d8f44}.gallery small{color:var(--muted)}
table{border-collapse:collapse;font-size:13px;width:100%}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}td.text{color:#2f2a55}td small{color:var(--muted)}
.lens{display:flex;align-items:flex-end;gap:3px;height:118px;border-bottom:1px solid #d9d0e8;padding-bottom:4px;overflow-x:auto}.lens div{flex:1 0 34px;max-width:60px;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;gap:3px}.lens i{display:block;width:100%;border-radius:4px 4px 0 0;background:#c9c1e6}.lens .final i{background:linear-gradient(180deg,#ff6fae,var(--pink))}.lens span{font-family:Consolas,monospace;font-size:10px;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.lens small{font-size:9px;color:var(--muted)}
.spark{margin:12px 0 0}.spark figcaption{font-size:12px;color:var(--muted)}.spark svg path{fill:none;stroke:#6a52d6;stroke-width:1.5}.spark-axis{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);max-width:320px}
.preview{max-height:260px;border-radius:8px}.plain-list{margin:0;padding-left:18px;font-size:13px;color:#2f2a55}footer{margin-top:20px;font-size:12px;color:var(--muted)}
.rtext{font-family:Consolas,"Cascadia Mono",monospace;font-size:14px;line-height:1.9;white-space:pre-wrap}.rprompt{color:var(--muted)}.rtok{--p:0;border-radius:4px;padding:1px 2px;background:color-mix(in srgb,var(--mint) calc(var(--p)*70%),#f2efe4)}
@media print{section{break-inside:avoid}body{background:#fff}}
`;

function patchingSection(trace) {
  const block = trace.patching;
  const story = patchStory(block);
  if (!story) return "";
  const rows = block.layers
    .map(
      (l) =>
        `<tr class="${l.top_is_source ? "flip" : ""}"><td>L${l.layer}</td><td>“${esc(tokenText(l.top_tokens[0]?.text))}” ${pct(l.top_tokens[0]?.probability)}</td><td>${pct(l.source_answer_probability)}</td><td>${pct(l.recovery)}</td><td>${num(l.kl_bits, 2)}</td></tr>`,
    )
    .join("");
  const body = `<p>${esc(story.text)}</p><p class="muted">Source prompt: ${esc(
    (block.source_run.tokens || []).map((t) => t.text).join(""),
  )} · source answer “${esc(tokenText(story.source.text))}” ${pct(story.source.probability)} · target answer “${esc(tokenText(story.target.text))}” ${pct(story.target.probability)} · patched at the ${block.positions === "all" ? "same positions" : "last position"}</p><table><tr><th>Layer</th><th>Top token after patch</th><th>P(source answer)</th><th>Recovery</th><th>KL bits</th></tr>${rows}</table>`;
  return section(
    "Which layer carries the source answer",
    body,
    block.limitations?.join(" "),
  );
}

function responseSection(trace) {
  const story = responseStory(trace);
  if (!story) return "";
  const prompt = esc((trace.tokens || []).map((t) => t.text).join(""));
  const tokens = trace.steps
    .map(
      (s) =>
        `<span class="rtok" style="--p:${s.probability.toFixed(3)}" title="${esc(`${pct(s.probability)} · ${s.entropy_bits.toFixed(2)} bits`)}">${esc(s.text === "" ? "∅" : s.text)}</span>`,
    )
    .join("");
  const rows = trace.steps
    .map((s) => {
      const settled = s.lens?.settled_layer;
      return `<tr><td>${s.index}</td><td>“${esc(tokenText(s.text))}”</td><td>${pct(s.probability)}</td><td>${num(s.entropy_bits, 2)}</td><td>${settled === null || settled === undefined ? "—" : settled === 0 ? "embeddings" : `L${settled - 1}`}</td><td>${esc(
        s.top_tokens
          .slice(1, 3)
          .map((t) => `“${tokenText(t.text)}” ${pct(t.probability)}`)
          .join(", "),
      )}</td></tr>`;
    })
    .join("");
  return section(
    "What it wrote",
    `<p>${esc(story.text)}</p><p class="rtext"><span class="rprompt">${prompt}</span>${tokens}</p><table><tr><th>#</th><th>Token</th><th>P</th><th>Bits</th><th>Settled</th><th>Runners-up</th></tr>${rows}</table>`,
    (trace.limitations || []).slice(0, 2).join(" "),
  );
}
