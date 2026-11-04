import { useEffect, useMemo, useState } from "react";
import { GitDiff, Lightbulb, Play } from "@phosphor-icons/react";
import { api } from "./api";
import {
  activationStats,
  attentionHeadProfile,
  compareOverlap,
  headReactions,
  layerReactions,
  lensDivergence,
  layerOf,
  patchStory,
  pct,
  promptCompareStory,
  promptFromTokens,
  tokenDiff,
  tokenLabel,
  tokenText,
} from "./insights";
import { mix } from "./MatrixWorkspace";
import DiffusionCompare from "./DiffusionCompare";

const short = (n) => (Number.isFinite(n) ? n.toPrecision(3) : "—");

function describe(trace) {
  if (!trace) return "—";
  if (trace.kind === "attention_capture")
    return `${trace.model_name || "model"} · L${trace.layer} H${trace.head} · ${trace.shape.join("×")}`;
  return `${trace.model_name || "model"} · ${trace.events?.length || 0} events`;
}

/** Side-by-side comparison of two retained traces, with a one-edit prompt rerun.
 * All numbers come from the recorded values of the two traces. */
export default function Compare({
  traceA,
  traces,
  traceId,
  pairedId = "",
  busy = false,
  canRun = false,
  canRunDiffusionModel = false,
  onRunVariant,
  onRunPatch,
  onRunAblation,
  fallbackPrompt = "",
}) {
  const [otherId, setOtherId] = useState("");
  const [traceB, setTraceB] = useState(null);
  const [error, setError] = useState("");
  const [promptB, setPromptB] = useState("");
  const candidates = (traces || []).filter((t) => t.id !== traceId);
  const rawPrompt = useMemo(
    () =>
      traceA?.tokens && (traceA.input_mode || "raw") === "raw"
        ? promptFromTokens(traceA.tokens)
        : fallbackPrompt,
    [traceA, fallbackPrompt],
  );
  useEffect(() => {
    setPromptB(rawPrompt);
  }, [rawPrompt]);
  useEffect(() => {
    if (pairedId) setOtherId(pairedId);
  }, [pairedId]);
  const revisionB = (traces || []).find((t) => t.id === otherId)?.revision ?? 0;
  useEffect(() => {
    if (!otherId) {
      setTraceB(null);
      return;
    }
    let alive = true;
    setError("");
    api(`traces/${otherId}`)
      .then((t) => alive && setTraceB(t))
      .catch((e) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [otherId, revisionB]);
  const promptOf = (t) =>
    t?.tokens && (t.input_mode || "raw") === "raw"
      ? promptFromTokens(t.tokens)
      : "";
  const patching = traceB?.patching || null;
  const patchStoryData = useMemo(() => patchStory(patching), [patching]);
  const patchSourceMatches =
    patching &&
    traceA?.tokens &&
    Array.isArray(patching.source_run.tokens) &&
    patching.source_run.tokens.length === traceA.tokens.length &&
    patching.source_run.tokens.every((t, i) => t.id === traceA.tokens[i].id);
  const canRunDiffusion =
    traceA?.kind === "diffusion_step_capture" &&
    canRunDiffusionModel &&
    traceA?.sampling?.prompt;
  const canPatch =
    Boolean(onRunPatch) &&
    canRun &&
    traceA?.tokens &&
    traceB?.tokens &&
    traceB.kind !== "diffusion_step_capture" &&
    (traceA.input_mode || "raw") === (traceB.input_mode || "raw") &&
    promptOf(traceA) &&
    promptOf(traceB);
  const attention =
    traceA?.kind === "attention_capture" &&
    traceB?.kind === "attention_capture";
  const diff = useMemo(
    () => (attention ? compareOverlap(traceA.values, traceB.values) : null),
    [attention, traceA, traceB],
  );
  const profiles = useMemo(
    () =>
      attention
        ? [
            attentionHeadProfile(traceA.values),
            attentionHeadProfile(traceB.values),
          ]
        : null,
    [attention, traceA, traceB],
  );
  const story = useMemo(
    () => (traceA && traceB ? promptCompareStory(traceA, traceB) : null),
    [traceA, traceB],
  );
  const heads = useMemo(() => headReactions(traceA, traceB), [traceA, traceB]);
  const layers = useMemo(
    () => layerReactions(traceA, traceB),
    [traceA, traceB],
  );
  const lens = useMemo(
    () => lensDivergence(traceA?.logit_lens, traceB?.logit_lens),
    [traceA, traceB],
  );
  const tokens = useMemo(
    () =>
      traceA?.tokens && traceB?.tokens
        ? tokenDiff(traceA.tokens, traceB.tokens)
        : null,
    [traceA, traceB],
  );
  const layerRms = (trace) => {
    const map = new Map();
    for (const e of trace?.events || []) {
      const layer = layerOf(e.module);
      const stats = activationStats(e.values);
      if (layer === null || !stats || map.has(layer)) continue;
      map.set(layer, stats.rms);
    }
    return map;
  };
  const rmsA = useMemo(() => layerRms(traceA), [traceA]);
  const rmsB = useMemo(() => layerRms(traceB), [traceB]);
  const sharedLayers = [...rmsA.keys()]
    .filter((l) => rmsB.has(l))
    .sort((a, b) => a - b);
  const rmsRanked = sharedLayers
    .map((l) => ({ layer: l, ratio: rmsB.get(l) / rmsA.get(l) }))
    .sort((x, y) => Math.abs(Math.log(y.ratio)) - Math.abs(Math.log(x.ratio)));
  const topA = traceA?.output?.top_tokens || [];
  const topB = traceB?.output?.top_tokens || [];
  const strips = [traceA, traceB].map((t) =>
    Array.isArray(t?.attribution?.last_position_attention)
      ? t.attribution.last_position_attention
      : null,
  );
  const changed = promptB.trim() && promptB.trim() !== rawPrompt.trim();
  return (
    <section className="compare panel" aria-label="Compare traces">
      <header>
        <h2>
          <GitDiff size={22} /> Compare
        </h2>
        <div className="compare-pick">
          <span>
            <b>A</b> {describe(traceA)}
          </span>
          <label>
            <b>B</b>
            <select
              value={otherId}
              onChange={(e) => setOtherId(e.target.value)}
            >
              <option value="">Choose a retained trace…</option>
              {candidates.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name} · {t.kind.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>
      {error && <p className="error">{error}</p>}
      {!traceA && (
        <p className="muted">Open a capture first, then pick a second one.</p>
      )}
      {traceA && traceA.kind !== "diffusion_step_capture" && onRunVariant && (
        <form
          className="compare-variant"
          onSubmit={(e) => {
            e.preventDefault();
            onRunVariant?.(promptB.trim());
          }}
        >
          <label>
            Prompt B · edit one word and rerun the same capture
            <textarea
              value={promptB}
              onChange={(e) => setPromptB(e.target.value)}
              maxLength={8192}
              aria-label="Variant prompt"
            />
          </label>
          <button
            type="submit"
            disabled={busy || !canRun || !promptB.trim() || !onRunVariant}
            title={
              canRun
                ? "Run the same capture with prompt B and compare"
                : "Load the model that produced trace A first"
            }
          >
            <Play size={16} weight="fill" /> {busy ? "Running…" : "Run B"}
          </button>
          <p className="micro">
            Reruns{" "}
            {traceA.kind === "attention_capture"
              ? `the attention probe at L${traceA.layer} H${traceA.head}`
              : "the module capture with the same components"}{" "}
            on the loaded model, keeps A open and pairs the new trace as B.
            {changed ? "" : " Prompt B currently matches A."}
          </p>
        </form>
      )}
      {traceA?.kind === "diffusion_step_capture" && (
        <DiffusionCompare
          traceA={traceA}
          traceB={traceB?.kind === "diffusion_step_capture" ? traceB : null}
          busy={busy}
          canRun={Boolean(canRunDiffusion)}
          onRunAblation={onRunAblation}
        />
      )}
      {traceA && !traceB && (
        <p className="muted">
          Pick a second retained trace or run prompt B. Attention captures
          compare weight by weight and head by head; module captures compare
          per-layer activation scale and the logit lens; both compare next-token
          predictions when present.
        </p>
      )}
      {story?.text && (
        <p className="compare-story">
          <Lightbulb size={16} weight="duotone" /> {story.text}
        </p>
      )}
      {tokens && (
        <div className="compare-card wide">
          <h3>Prompt tokens</h3>
          {[traceA, traceB].map((t, i) => {
            const [from, to] = i ? tokens.changedB : tokens.changedA;
            return (
              <div key={i} className="compare-tokens-diff">
                <b>{i ? "B" : "A"}</b>
                {t.tokens.map((tok, k) => (
                  <span
                    key={k}
                    className={k >= from && k < to ? "changed" : ""}
                    title={`position ${k} · id ${tok.id}`}
                  >
                    {tok.text === "" ? "∅" : tok.text}
                  </span>
                ))}
              </div>
            );
          })}
          <p className="micro">
            Pink tokens differ between the prompts; matching prefix and suffix
            are aligned by token id.
          </p>
        </div>
      )}
      {attention && (
        <div className="compare-grid">
          {[traceA, traceB].map((t, i) => (
            <div key={i} className="compare-card">
              <h3>
                {i ? "B" : "A"} · L{t.layer} H{t.head}
              </h3>
              <Thumb matrix={t.values} />
              <p className="compare-label">{profiles[i].label}</p>
              <div className="matrix-chips">
                <span>sink {pct(profiles[i].sinkMass)}</span>
                <span>prev {pct(profiles[i].previousTokenMass)}</span>
                <span>{profiles[i].entropyBits.toFixed(2)} bits</span>
              </div>
            </div>
          ))}
          <div className="compare-card">
            <h3>A − B</h3>
            {diff ? (
              <>
                <Thumb matrix={diff.diff} signed />
                <p className="compare-label">
                  mean |Δ| {short(diff.meanAbs)} · largest {short(diff.maxDiff)}{" "}
                  at query {diff.at[0]}
                  {traceA.tokens?.[diff.at[0]] &&
                    ` “${tokenLabel(traceA.tokens[diff.at[0]])}”`}{" "}
                  → key {diff.at[1]}
                  {traceA.tokens?.[diff.at[1]] &&
                    ` “${tokenLabel(traceA.tokens[diff.at[1]])}”`}
                </p>
                <p className="micro">
                  Teal where A attends more, coral where B attends more.
                  {diff.truncated &&
                    ` Shapes differ (${traceA.shape.join("×")} vs ${traceB.shape.join("×")}); only the shared ${diff.rows}×${diff.cols} block is subtracted.`}
                </p>
              </>
            ) : (
              <p className="muted">No overlapping weights to subtract.</p>
            )}
          </div>
        </div>
      )}
      {traceB && (heads.length > 0 || layers.length > 0) && (
        <div className="compare-card wide">
          <h3>Where the attention reacted</h3>
          <div className="compare-react">
            {heads.length > 0 && (
              <div>
                <p className="matrix-insight-sub">
                  Heads of layer {traceA.layer} by mean |Δ| weight
                </p>
                <ul className="matrix-attends">
                  {heads.slice(0, 8).map((h) => (
                    <li key={h.head}>
                      <button
                        disabled
                        title={`largest single change ${h.maxDiff.toFixed(3)} · entropy Δ ${h.entropyDelta.toFixed(2)} bits`}
                      >
                        <span className="matrix-attends-token">
                          H{h.head}
                          {h.relabelled && (
                            <small>
                              {" "}
                              {h.before} → {h.after}
                            </small>
                          )}
                        </span>
                        <span
                          className="matrix-attends-bar"
                          style={{
                            width: `${Math.max(2, (h.meanAbs / Math.max(1e-9, heads[0].meanAbs)) * 100)}%`,
                          }}
                        />
                        <b>{h.meanAbs.toFixed(3)}</b>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {layers.length > 0 && (
              <div>
                <p className="matrix-insight-sub">
                  Layers by change in attention entropy (B − A)
                </p>
                <ul className="matrix-attends">
                  {layers.slice(0, 8).map((l) => (
                    <li key={l.layer}>
                      <button
                        disabled
                        title={`sink mass Δ ${pct(l.sinkDelta)} · previous-token Δ ${pct(l.previousDelta)}`}
                      >
                        <span className="matrix-attends-token">L{l.layer}</span>
                        <span
                          className={`matrix-attends-bar ${l.entropyDelta < 0 ? "down" : ""}`}
                          style={{
                            width: `${Math.max(2, (Math.abs(l.entropyDelta) / Math.max(1e-9, Math.abs(layers[0].entropyDelta))) * 100)}%`,
                          }}
                        />
                        <b>
                          {l.entropyDelta >= 0 ? "+" : ""}
                          {l.entropyDelta.toFixed(2)}
                        </b>
                      </button>
                    </li>
                  ))}
                </ul>
                <p className="micro">
                  Head-averaged over each layer's full matrix; teal bars spread
                  out, pink bars focused.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
      {traceB && strips[0] && strips[1] && (
        <div className="compare-card wide">
          <h3>What the last token looked at</h3>
          <div className="compare-strips">
            {strips.map((strip, i) => {
              const t = i ? traceB : traceA;
              const peak = Math.max(...strip, 1e-9);
              return (
                <div key={i}>
                  <p className="matrix-insight-sub">
                    {i ? "B" : "A"} · “
                    {tokenLabel(t.tokens?.at(-1), strip.length - 1)}”
                  </p>
                  <div className="matrix-attribution">
                    {strip.map((v, k) => (
                      <span
                        key={k}
                        style={{ "--heat": Math.min(1, v / peak) }}
                        title={`${pct(v)}`}
                      >
                        {t.tokens?.[k] ? tokenLabel(t.tokens[k], k) : k}
                        <small>{pct(v)}</small>
                      </span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
          <p className="micro">
            Measured attention from the final query position, averaged over
            every layer and head of each run.
          </p>
        </div>
      )}
      {traceB && sharedLayers.length > 0 && (
        <div className="compare-card wide">
          <h3>Activation RMS per layer</h3>
          {rmsRanked.length > 0 && (
            <p className="micro">
              Most changed: L{rmsRanked[0].layer} at {short(rmsRanked[0].ratio)}
              × A
              {rmsRanked[1] &&
                `, then L${rmsRanked[1].layer} at ${short(rmsRanked[1].ratio)}×`}
              .
            </p>
          )}
          <p className="micro">
            RMS of each tensor's retained prefix (
            {traceA.capture?.max_tensor_elements?.toLocaleString() ??
              "the first"}{" "}
            values), not the whole tensor; under causal attention a shared
            prompt prefix makes those leading positions identical.
          </p>
          <table className="compare-table">
            <thead>
              <tr>
                <th>Layer</th>
                <th>A</th>
                <th>B</th>
                <th>B / A</th>
              </tr>
            </thead>
            <tbody>
              {sharedLayers.map((l) => (
                <tr key={l}>
                  <td>L{l}</td>
                  <td>{short(rmsA.get(l))}</td>
                  <td>{short(rmsB.get(l))}</td>
                  <td
                    className={
                      Math.abs(Math.log(rmsB.get(l) / rmsA.get(l))) >
                      Math.log(1.5)
                        ? "hot"
                        : ""
                    }
                  >
                    {short(rmsB.get(l) / rmsA.get(l))}×
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {traceB && lens && (
        <div className="compare-card wide">
          <h3>Logit lens agreement</h3>
          <div className="compare-lens" role="list">
            {lens.agree.map((same, i) => (
              <span
                key={i}
                role="listitem"
                className={same ? "" : "differs"}
                title={`${traceA.logit_lens.layers[i].label}: A “${tokenText(traceA.logit_lens.layers[i].top_tokens[0]?.text)}” · B “${tokenText(traceB.logit_lens.layers[i].top_tokens[0]?.text)}”`}
              >
                {i === 0 ? "e" : i - 1}
              </span>
            ))}
          </div>
          <p className="micro">
            Green layers rank the same top token for both prompts; pink layers
            differ.{" "}
            {lens.firstDisagreement === null
              ? "The two answers form the same way at every layer."
              : `They part at ${lens.firstLabel}${lens.lastDisagreement !== null && lens.lastDisagreement !== lens.firstDisagreement ? ` and last differ at ${traceA.logit_lens.layers[lens.lastDisagreement].label}` : ""}.`}
          </p>
        </div>
      )}
      {traceB && (canPatch || patching) && (
        <div className="compare-card wide">
          <h3>
            Which layer carries A's answer{" "}
            <small className="muted">activation patching, A → B</small>
          </h3>
          {patching && patchSourceMatches && patchStoryData ? (
            <>
              <p className="compare-story">
                <Lightbulb size={16} weight="duotone" /> {patchStoryData.text}
              </p>
              <div className="patch-track" role="list">
                {patching.layers.map((l) => (
                  <span
                    key={l.layer}
                    role="listitem"
                    className={`${l.top_is_source ? "flip" : ""} ${l.layer === patchStoryData.best.layer ? "best" : ""}`}
                    style={{
                      "--r": Math.max(0, Math.min(1, l.recovery)),
                    }}
                    title={`Layer ${l.layer}: “${tokenText(patchStoryData.source.text)}” ${pct(l.source_answer_probability)} (recovery ${pct(l.recovery)}) · top “${tokenText(l.top_tokens[0]?.text)}” ${pct(l.top_tokens[0]?.probability)} · ${l.kl_bits.toFixed(2)} bits`}
                  >
                    <i />
                    <small>{l.layer}</small>
                  </span>
                ))}
              </div>
              <p className="micro">
                Bar height is how much of the gap between B's and A's
                probability of “{tokenText(patchStoryData.source.text)}” the
                patch at that layer restores; pink bars flip B's top token to
                A's answer. Patched at the{" "}
                {patching.positions === "all"
                  ? "same positions"
                  : "last position"}
                , one layer at a time. {patching.limitations?.[1]}
              </p>
            </>
          ) : patching && !patchSourceMatches ? (
            <p className="muted">
              B carries a patching result from a different source prompt. Run
              patching again to measure against this A.
            </p>
          ) : (
            <p className="muted">
              Copy A's residual stream into B's forward pass at the last
              position, one decoder layer at a time, and measure how much of A's
              answer comes back. This is a causal test of where the answer is
              carried.
            </p>
          )}
          {canPatch && (
            <div className="compare-patch-actions">
              <button
                disabled={busy}
                onClick={() =>
                  onRunPatch(
                    promptOf(traceA),
                    promptOf(traceB),
                    otherId,
                    "last",
                    traceB.input_mode || "raw",
                  )
                }
              >
                <Play size={14} weight="fill" />{" "}
                {busy ? "Running…" : patching ? "Patch again" : "Patch A → B"}
              </button>
              {traceA.tokens.length === traceB.tokens.length && (
                <button
                  disabled={busy}
                  onClick={() =>
                    onRunPatch(
                      promptOf(traceA),
                      promptOf(traceB),
                      otherId,
                      "all",
                      traceB.input_mode || "raw",
                    )
                  }
                  title="Patch every position (prompts have the same length)"
                >
                  Patch all positions
                </button>
              )}
              <span className="micro">
                One forward pass per layer ({traceB.structure?.nodes ? "" : ""}
                {patching?.layer_count ?? "all"} layers) on the loaded model.
              </span>
            </div>
          )}
        </div>
      )}
      {traceB && (topA.length || topB.length) > 0 && (
        <div className="compare-card wide">
          <h3>Next-token prediction</h3>
          <div className="compare-tokens">
            {[topA, topB].map((list, i) => (
              <div key={i}>
                <b>{i ? "B" : "A"}</b>
                {list.slice(0, 5).map((t) => {
                  const other = (i ? topA : topB).find(
                    (o) => o.token_id === t.token_id,
                  );
                  return (
                    <span key={t.token_id} className={other ? "" : "only"}>
                      “{tokenText(t.text ?? String(t.token_id))}”{" "}
                      {pct(t.probability)}
                      {!other && <small>only here</small>}
                    </span>
                  );
                })}
                {!list.length && (
                  <span className="muted">no prediction recorded</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function Thumb({ matrix, signed }) {
  const n = matrix.length,
    m = matrix[0]?.length || 0;
  const peak = signed
    ? Math.max(...matrix.flat().map((v) => Math.abs(v ?? 0)), 1e-9)
    : 1;
  return (
    <div
      className="compare-thumb"
      style={{
        gridTemplateColumns: `repeat(${m},1fr)`,
        aspectRatio: `${m}/${n}`,
      }}
      aria-hidden="true"
    >
      {matrix.map((row, r) =>
        row.map((v, c) => (
          <i
            key={`${r}:${c}`}
            style={{
              background:
                v === null
                  ? "#2a2342"
                  : signed
                    ? v >= 0
                      ? `rgba(15,159,134,${Math.min(1, v / peak)})`
                      : `rgba(241,109,131,${Math.min(1, -v / peak)})`
                    : mix(Math.max(0, Math.min(1, v))),
            }}
          />
        )),
      )}
    </div>
  );
}
