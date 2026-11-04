import { useEffect, useMemo, useRef, useState } from "react";
import {
  Cube,
  Stack,
  GridFour,
  ChartBar,
  FolderOpen,
  UploadSimple,
  DownloadSimple,
  Play,
  Pause,
  Stop,
  ArrowClockwise,
  ArrowsOut,
  X,
  CaretRight,
  GearSix,
  Eye,
  Trash,
  Info,
  Crosshair,
  Hand,
  MagnifyingGlass,
  GitDiff,
  CaretDown,
  ArrowsClockwise,
} from "@phosphor-icons/react";
import Scene from "./Scene";
import MatrixWorkspace from "./MatrixWorkspace";
import Prediction from "./Prediction";
import Compare from "./Compare";
import { api, displayNodes, samples, eventPath } from "./api";
const num = (n) =>
  Number.isFinite(n)
    ? new Intl.NumberFormat("en", { maximumFractionDigits: 4 }).format(n)
    : "Unavailable";
const short = (s) => s?.split(".").slice(-2).join(".") || "Model output";
function Button({ icon: Icon, children, ...props }) {
  return (
    <button {...props}>
      <Icon size={19} />
      {children}
    </button>
  );
}
function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

export function App() {
  const [state, setState] = useState(null),
    [catalog, setCatalog] = useState({ artifacts: [] });
  const [tree, setTree] = useState(null),
    [trace, setTrace] = useState(null),
    [traceId, setTraceId] = useState("");
  const [selected, setSelected] = useState(""),
    [eventIndex, setEventIndex] = useState(0),
    [view, setView] = useState("matrix"),
    [panel, setPanel] = useState("component"),
    [modal, setModal] = useState(null);
  const [error, setError] = useState(""),
    [prompt, setPrompt] = useState(
      "Explain how a transformer works in a few sentences.",
    );
  const [device, setDevice] = useState("cuda"),
    [inputMode, setInputMode] = useState("raw"),
    [search, setSearch] = useState("");
  const [spread, setSpread] = useState(1),
    [rotate, setRotate] = useState(false),
    [resetKey, setResetKey] = useState(0),
    [playing, setPlaying] = useState(false);
  const [logScale, setLogScale] = useState(true);
  const [cameraMode, setCameraMode] = useState("orbit");
  const [tensor, setTensor] = useState("attention");
  const [headView, setHeadView] = useState(null);
  const native = state?.host === "comfyui";
  const [elements, setElements] = useState(1024),
    [steps, setSteps] = useState(8),
    [size, setSize] = useState(256),
    [seed, setSeed] = useState(0);
  const [layer, setLayer] = useState(0),
    [head, setHead] = useState(0),
    [inspection, setInspection] = useState(null),
    [capturePaths, setCapturePaths] = useState([]),
    [speed, setSpeed] = useState(1);
  const importRef = useRef(),
    lastJob = useRef(),
    lastPublished = useRef(""),
    selectedTrace = useRef("");
  const busy = state?.job.status === "running",
    nodes = useMemo(() => displayNodes(tree, trace), [tree, trace]),
    allNodes = tree?.nodes || nodes;
  const events = useMemo(
    () =>
      trace?.kind === "attention_capture"
        ? [
            {
              module: `Attention L${trace.layer} / H${trace.head}`,
              values: trace.values.flat(),
              shape: trace.shape,
              total_elements: trace.shape[0] * trace.shape[1],
              selection: "full_attention_head",
              dtype: "float32",
              truncated: false,
            },
          ]
        : trace?.events || [],
    [trace],
  );
  const viewTrace = useMemo(() => {
    if (
      !trace ||
      trace.kind !== "attention_capture" ||
      headView === null ||
      headView === trace.head ||
      !Array.isArray(trace.heads?.[headView])
    )
      return trace;
    return { ...trace, head: headView, values: trace.heads[headView] };
  }, [trace, headView]);
  const replayEvent =
    events[Math.min(eventIndex, Math.max(0, events.length - 1))];
  const event =
    !selected || eventPath(replayEvent) === selected ? replayEvent : null;
  const activeNode = allNodes.find((n) => n.path === selected),
    vals = samples(event);
  const expandedComponents = useMemo(() => {
    const parent = nodes.find(
      (n) => n.path === selected || selected.startsWith(n.path + "."),
    );
    if (!parent || tree?.kind === "artifact_tensor_namespace") return [];
    return allNodes
      .filter(
        (n) =>
          n.path.startsWith(parent.path + ".") &&
          n.path.split(".").length === parent.path.split(".").length + 1,
      )
      .map((n) => ({
        ...n,
        children: allNodes.filter(
          (child) =>
            child.path.startsWith(n.path + ".") &&
            child.path.split(".").length === n.path.split(".").length + 1,
        ),
      }))
      .sort((a, b) => b.children.length - a.children.length)
      .slice(0, 4);
  }, [nodes, selected, allNodes, tree]);
  const modelName =
    tree?.name ||
    trace?.model_name ||
    trace?.model_class?.split(".").pop() ||
    state?.loaded?.name ||
    "Your model, revealed";
  const layerIndices = useMemo(
    () =>
      nodes
        .map((n) =>
          /(?:layers|blocks|transformer_blocks|double_blocks|single_blocks|joint_blocks)\.(\d+)$/.exec(
            n.path,
          ),
        )
        .filter(Boolean)
        .map((m) => Number(m[1])),
    [nodes],
  );
  const mlpUnits = useMemo(() => {
    const unit = allNodes.find((n) =>
      /\.mlp\.(?:up_proj|gate_proj|fc1|w1|c_fc)$/.test(n.path),
    );
    const shape = unit?.parameters?.[0]?.shape;
    return Array.isArray(shape) && shape.length ? Math.max(...shape) : null;
  }, [allNodes]);
  const modelFacts = state?.loaded
    ? [
        state.loaded.name,
        layerIndices.length ? `${layerIndices.length} layers` : null,
        mlpUnits ? `${mlpUnits.toLocaleString()} MLP units/layer` : null,
      ]
        .filter(Boolean)
        .join(" · ")
    : state?.host === "comfyui"
      ? "ComfyUI connected"
      : trace
        ? modelName
        : "Local workspace";
  const layerOptions = layerIndices.length
    ? layerIndices
    : Array.from(
        { length: Math.max(1, (trace?.layer ?? 0) + 1, layer + 1) },
        (_, i) => i,
      );
  const tensorOptions = useMemo(() => {
    const modules = [
      ...new Set(
        events
          .map((e) => e.module)
          .filter((m) => m && !/^Attention L\d+/.test(m)),
      ),
    ];
    const atLayer = modules.filter((m) =>
      new RegExp(
        `(?:layers|blocks|transformer_blocks|double_blocks|single_blocks|joint_blocks)\\.${layer}(?:\\.|$)`,
      ).test(m),
    );
    return (atLayer.length ? atLayer : modules).slice(0, 64);
  }, [events, layer]);
  const status = busy
    ? `${state.job.action} · working`
    : state?.job.status === "failed"
      ? "Needs attention"
      : trace
        ? "Captured · ready to explore"
        : state?.loaded
          ? "Ready to capture"
          : "Local observatory";
  async function action(fn) {
    setError("");
    try {
      return await fn();
    } catch (e) {
      setError(e.message);
    }
  }
  function choose(path) {
    setSelected(path);
    const index = events.findIndex((e) => eventPath(e) === path);
    if (index >= 0) setEventIndex(index);
  }
  async function openTrace(id) {
    selectedTrace.current = id;
    const t = await api(`traces/${id}`);
    if (selectedTrace.current !== id) return;
    setTraceId(id);
    setTrace(t);
    setHeadView(null);
    setTree(t.structure || null);
    setEventIndex(0);
    setSelected(eventPath(t.events?.[0]));
    setPlaying(false);
    setInspection(null);
    setPanel(t.kind === "attention_capture" ? "attention" : "component");
    if (t.kind === "attention_capture") {
      if (Number.isInteger(t.layer)) setLayer(t.layer);
      if (Number.isInteger(t.head)) setHead(t.head);
    }
  }
  useEffect(() => {
    let alive = true,
      timer;
    async function poll() {
      try {
        const s = await api("state");
        if (!alive) return;
        setState(s);
        if (s.job.status !== "running" && s.job.job_id !== lastJob.current) {
          lastJob.current = s.job.job_id;
          setCatalog(await api("catalog"));
          if (s.job.action === "load" && s.job.status === "completed") {
            setTree(await api("structure"));
            setTrace(null);
            setTraceId("");
            setSelected("");
            setCapturePaths([]);
          }
          if (s.job.status === "failed") setError(s.job.error);
        }
        const latest = s.traces.at(-1);
        const version = latest && `${latest.id}:${latest.revision || 1}`;
        if (latest && version !== lastPublished.current) {
          const sameRun = lastPublished.current.startsWith(`${latest.id}:`);
          lastPublished.current = version;
          if (!sameRun) {
            await openTrace(latest.id);
          } else if (selectedTrace.current === latest.id) {
            const updated = await api(`traces/${latest.id}`);
            if (alive && selectedTrace.current === latest.id) setTrace(updated);
          }
        }
      } catch (e) {
        if (alive) setError(e.message);
      }
      if (alive) timer = setTimeout(poll, 900);
    }
    poll();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);
  useEffect(() => {
    if (!playing || !events.length) return;
    const timer = setInterval(
      () =>
        setEventIndex((i) => {
          if (i >= events.length - 1) {
            setPlaying(false);
            return i;
          }
          setSelected(eventPath(events[i + 1]));
          return i + 1;
        }),
      850 / speed,
    );
    return () => clearInterval(timer);
  }, [playing, events, speed]);
  useEffect(() => {
    function escape(e) {
      if (e.key === "Escape") setModal(null);
    }
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, []);
  function run(attention = false) {
    setPlaying(false);
    action(() =>
      api("run", {
        prompt,
        input_mode: inputMode,
        modules: capturePaths,
        elements,
        steps,
        size,
        seed,
        attention,
        layer,
        head,
      }),
    );
  }
  function exportTrace() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(trace)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "model-run-trace.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function importTrace(e) {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    await action(async () => {
      if (file.size > 16 * 1024 * 1024) throw new Error("Trace exceeds 16 MiB");
      const r = await api("import", JSON.parse(await file.text()));
      await openTrace(r.id);
    });
  }
  return (
    <main>
      <header className="topbar">
        <div className="brand">
          <Play weight="bold" size={31} />
          <h1>Model Run</h1>
        </div>
        <span className="tagline">
          A model data observatory for tensors, activations, and events
        </span>
        <span className="host-badge">{modelFacts}</span>
      </header>
      <nav className="toolbar" aria-label="Workspace">
        <div className="toolbar-tabs">
          <button
            className={view === "matrix" ? "active" : ""}
            onClick={() => setView("matrix")}
            aria-pressed={view === "matrix"}
          >
            <GridFour size={19} /> Tensor Exploration Table
          </button>
          <span className="toolbar-rule" />
          <button
            className={view !== "matrix" && view !== "compare" ? "active" : ""}
            onClick={() => setView("model")}
            aria-pressed={view !== "matrix" && view !== "compare"}
          >
            <ArrowsClockwise size={19} /> Run View
          </button>
          <span className="toolbar-rule" />
          <button
            className={view === "compare" ? "active" : ""}
            onClick={() => setView("compare")}
            aria-pressed={view === "compare"}
          >
            <GitDiff size={19} /> Compare
          </button>
          <span className="toolbar-rule" />
          <button disabled={!trace} onClick={exportTrace}>
            <DownloadSimple size={19} /> Export
          </button>
          <span className="toolbar-rule" />
          <button onClick={() => importRef.current?.click()}>
            <UploadSimple size={19} /> Import
          </button>
        </div>
        <div className="toolbar-run">
          <span className="toolbar-rule tall" />
          <label className="toolbar-field">
            <span>Model</span>
            <button className="model-picker" onClick={() => setModal("models")}>
              <span>
                {state?.loaded?.name ||
                  (native && trace ? modelName : "Choose a model")}
              </span>
              <CaretDown size={14} weight="bold" />
            </button>
          </label>
          <label className="toolbar-field">
            <span>Layer</span>
            <select
              aria-label="Layer"
              value={layerOptions.includes(layer) ? layer : layerOptions[0]}
              onChange={(e) => setLayer(Number(e.target.value))}
            >
              {layerOptions.map((n) => (
                <option key={n} value={n}>
                  L{n}
                </option>
              ))}
            </select>
          </label>
          <label className="toolbar-field">
            <span>Tensor</span>
            <select
              aria-label="Tensor"
              className="tensor-select"
              value={tensor}
              onChange={(e) => {
                const next = e.target.value;
                setTensor(next);
                if (next !== "attention") {
                  const at = events.findIndex((ev) => ev.module === next);
                  if (at >= 0) {
                    setEventIndex(at);
                    setSelected(eventPath(events[at]));
                  }
                }
              }}
            >
              <option value="attention">Attention</option>
              {tensorOptions.map((m) => (
                <option key={m} value={m}>
                  {m.replace(/^model\./, "")}
                </option>
              ))}
            </select>
          </label>
          <Button
            icon={busy ? Stop : Play}
            className="primary run-button"
            disabled={
              !native &&
              !busy &&
              (!state?.loaded || tree?.kind === "artifact_tensor_namespace")
            }
            onClick={() =>
              native
                ? window.open(
                    new URL("../", window.location.href),
                    "_blank",
                    "noopener",
                  )
                : busy
                  ? action(() => api("cancel", {}))
                  : run(
                      tensor === "attention" && state?.loaded?.task === "text",
                    )
            }
          >
            {native ? "ComfyUI" : busy ? "Cancel" : "Run"}
          </Button>
        </div>
      </nav>
      <section className="input-bar" aria-label="Capture input">
        <label htmlFor="prompt">{native ? "Capture" : "Prompt"}</label>
        <input
          id="prompt"
          value={
            native
              ? "Connect Model Run · Observe to your MODEL, then queue the workflow in ComfyUI."
              : prompt
          }
          readOnly={native}
          onChange={(e) => setPrompt(e.target.value)}
          maxLength={8192}
        />
        <div className="tokens">
          <span className="muted">
            {trace?.tokens
              ? `Tokens (${trace.tokens.length})`
              : native
                ? "Native execution"
                : "Tokens appear after a capture"}
          </span>
          {trace?.tokens?.map((t, i) => (
            <span
              className="token"
              key={i}
              title={`Position ${i} · Token ID ${t.id}`}
            >
              {t.text || "[empty]"}
            </span>
          ))}
        </div>
        <Button
          icon={GearSix}
          aria-label="Capture settings"
          onClick={() => setModal("settings")}
        />
      </section>
      {error && (
        <div className="error" role="alert">
          <Info size={20} />
          <span>{error}</span>
          <button aria-label="Dismiss error" onClick={() => setError("")}>
            <X />
          </button>
        </div>
      )}
      <input
        hidden
        type="file"
        accept=".json,application/json"
        ref={importRef}
        onChange={importTrace}
      />
      {view === "compare" ? (
        <Compare traceA={trace} traces={state?.traces} traceId={traceId} />
      ) : view === "matrix" ? (
        <MatrixWorkspace
          trace={viewTrace}
          event={event}
          events={events}
          eventIndex={eventIndex}
          onPickEvent={(i) => {
            setEventIndex(i);
            setSelected(eventPath(events[i]));
            setPlaying(false);
          }}
          onPickHead={(i) => {
            setHeadView(i);
            setHead(i);
          }}
          layers={nodes}
          busy={busy}
          canAttention={state?.loaded?.task === "text"}
          onCapture={() => run(true)}
          layer={layer}
          setLayer={setLayer}
          head={head}
          setHead={setHead}
        />
      ) : (
        <div className="workspace">
          <aside className="sidebar panel">
            <h2 className="eyebrow">Explore</h2>
            <nav aria-label="Views">
              {[
                ["model", Cube, "3D Model"],
                ["layers", Stack, "Layer Stack"],
                ["volume", GridFour, "Tensor Volume"],
                ["metrics", ChartBar, "Run Metrics"],
              ].map(([id, Icon, label]) => (
                <Button
                  key={id}
                  icon={Icon}
                  className={view === id ? "active" : ""}
                  onClick={() => setView(id)}
                  aria-pressed={view === id}
                >
                  {label}
                </Button>
              ))}
            </nav>
            <hr />
            <h2 className="eyebrow">Workspace</h2>
            <Button icon={FolderOpen} onClick={() => setModal("models")}>
              Model Library
            </Button>
            <Button
              icon={UploadSimple}
              onClick={() => importRef.current.click()}
            >
              Import Trace
            </Button>
            <Button
              icon={DownloadSimple}
              disabled={!trace}
              onClick={exportTrace}
            >
              Export Trace
            </Button>
            <hr />
            <h2 className="eyebrow">Display</h2>
            <label className="check">
              <input
                type="checkbox"
                checked={rotate}
                onChange={(e) => setRotate(e.target.checked)}
              />
              Auto orbit
            </label>
            <Field label="Layer spacing">
              <input
                aria-label="Layer spacing"
                type="range"
                min=".5"
                max="1.8"
                step=".1"
                value={spread}
                onChange={(e) => setSpread(Number(e.target.value))}
              />
            </Field>
            <label className="check scale-toggle">
              <input
                type="checkbox"
                checked={logScale}
                onChange={(e) => setLogScale(e.target.checked)}
              />
              Signed log scale
            </label>
            <p className="micro">
              Display controls change the view, never the model.
            </p>
            <div className="sidebar-bottom">
              <span className="status-dot" />
              {status}
            </div>
          </aside>
          <section className="stage panel" aria-label="Model visualization">
            <div className="stage-header">
              <div>
                <h2>{modelName}</h2>
                <p>
                  {trace
                    ? `${events.length} captured ${trace.kind === "diffusion_step_capture" ? "steps" : "events"}`
                    : tree
                      ? `${allNodes.length} ${tree.kind === "artifact_tensor_namespace" ? "tensor namespace groups" : "runtime components"}`
                      : "Structure. Activity. Understanding."}
                  {selected && ` · ${short(selected)}`}
                </p>
              </div>
              <div className="scene-tools">
                <Button
                  icon={Cube}
                  className={cameraMode === "orbit" ? "camera-selected" : ""}
                  onClick={() => setCameraMode("orbit")}
                  title="Drag to orbit"
                >
                  Orbit
                </Button>
                <Button
                  icon={Hand}
                  className={cameraMode === "pan" ? "camera-selected" : ""}
                  onClick={() => setCameraMode("pan")}
                  title="Drag to pan"
                >
                  Pan
                </Button>
                <Button
                  icon={MagnifyingGlass}
                  className={cameraMode === "zoom" ? "camera-selected" : ""}
                  onClick={() => setCameraMode("zoom")}
                  title="Drag to zoom"
                >
                  Zoom
                </Button>
                {tree?.kind === "artifact_tensor_namespace" &&
                  state?.loaded && (
                    <Button
                      icon={Eye}
                      title={`Return to loaded ${state.loaded.name}`}
                      onClick={() =>
                        action(async () => {
                          setTree(await api("structure"));
                          setSelected("");
                        })
                      }
                    >
                      Runtime
                    </Button>
                  )}
                <Button
                  icon={Crosshair}
                  onClick={() => setResetKey((x) => x + 1)}
                  title="Reset camera"
                >
                  Reset
                </Button>
                <Button
                  icon={ArrowsOut}
                  onClick={() =>
                    action(() =>
                      document.querySelector(".stage").requestFullscreen(),
                    )
                  }
                  aria-label="Expand visualization"
                />
              </div>
            </div>
            {view === "metrics" ? (
              <div className="metrics">
                <h3>Capture report</h3>
                <dl>
                  <dt>Trace kind</dt>
                  <dd>{trace?.kind || "No capture"}</dd>
                  <dt>Runtime</dt>
                  <dd>PyTorch {trace?.torch_version || "—"}</dd>
                  <dt>Host elapsed</dt>
                  <dd>{num(trace?.elapsed_host_seconds)} s</dd>
                  <dt>Recorded events</dt>
                  <dd>{events.length}</dd>
                  <dt>Dropped events</dt>
                  <dd>{trace?.dropped_events ?? "—"}</dd>
                  <dt>Capture status</dt>
                  <dd>{trace?.status || "—"}</dd>
                  <dt>Model class</dt>
                  <dd>{trace?.model_class || "—"}</dd>
                </dl>
                <h3>Measurement context</h3>
                <ul>
                  {trace?.limitations?.map((l, i) => <li key={i}>{l}</li>) || (
                    <li>
                      Run a model or import a trace to inspect measured results.
                    </li>
                  )}
                </ul>
              </div>
            ) : view === "layers" ? (
              <div className="component-list">
                <div className="list-heading">
                  <h3>
                    {tree?.kind === "artifact_tensor_namespace"
                      ? "Stored tensor namespaces"
                      : "Runtime component hierarchy"}
                  </h3>
                  <p>
                    Containment, not execution order. Select components for
                    inspection or capture.
                  </p>
                </div>
                {allNodes.slice(0, 2000).map((n, i) => (
                  <div
                    className={`component-row ${selected === n.path ? "chosen" : ""}`}
                    key={`${n.path}:${i}`}
                  >
                    <button
                      onClick={() => choose(n.path)}
                      style={{
                        paddingLeft:
                          12 + Math.min(n.path.split(".").length, 5) * 12,
                      }}
                    >
                      <CaretRight />
                      <span>{n.path || "(root)"}</span>
                      <small>{n.type?.split(".").pop()}</small>
                    </button>
                    {state?.loaded?.task === "text" &&
                      tree?.kind !== "artifact_tensor_namespace" && (
                        <input
                          aria-label={`Capture ${n.path || "root"}`}
                          type="checkbox"
                          checked={capturePaths.includes(n.path)}
                          onChange={(e) => {
                            if (e.target.checked && capturePaths.length >= 32) {
                              setError("Capture at most 32 components");
                              return;
                            }
                            setCapturePaths((p) =>
                              e.target.checked
                                ? [...p, n.path]
                                : p.filter((x) => x !== n.path),
                            );
                          }}
                        />
                      )}
                  </div>
                ))}
                {allNodes.length > 2000 && (
                  <p>
                    First 2,000 components shown. The 3D overview displays up to
                    64 groups.
                  </p>
                )}
              </div>
            ) : (
              <>
                <Scene
                  components={expandedComponents}
                  cameraMode={cameraMode}
                  events={events}
                  nodes={nodes}
                  selected={selected}
                  event={event}
                  view={view}
                  spread={spread}
                  rotate={rotate}
                  onSelect={choose}
                  resetKey={resetKey}
                  logScale={logScale}
                />
                {nodes.length > 0 && (
                  <div className="layer-rail" aria-label="Model components">
                    {nodes.map((n, i) => (
                      <button
                        key={n.path}
                        title={n.path}
                        className={
                          n.path === selected ||
                          selected.startsWith(n.path + ".")
                            ? "selected"
                            : ""
                        }
                        onClick={() => choose(n.path)}
                      >
                        {/\d+$/.test(n.path)
                          ? `${n.path.includes("double_blocks") ? "D" : n.path.includes("single_blocks") ? "S" : "L"}${n.path.match(/\d+$/)[0]}`
                          : i + 1}
                      </button>
                    ))}
                  </div>
                )}
                {!nodes.length && !event && (
                  <div className="empty-stage">
                    <Cube size={56} weight="light" />
                    <h3>Step inside a model.</h3>
                    <p>
                      Open a local model, observe a ComfyUI run, or import a
                      capture. Explore the structure and see the values behind
                      each result.
                    </p>
                    <Button
                      icon={FolderOpen}
                      className="primary"
                      onClick={() => setModal("models")}
                    >
                      Open model library
                    </Button>
                    <button
                      className="text-button"
                      onClick={() => importRef.current.click()}
                    >
                      Import an existing trace
                    </button>
                  </div>
                )}
                <div className="scene-caption">
                  <span>
                    {event && view === "model" && expandedComponents.length
                      ? "Layer dots: first captured samples · explore values in Tensor Volume"
                      : event
                        ? `${event.selection || "flattened_prefix"} · ${Math.min(event.values?.flat().length || 0, 2048).toLocaleString()} displayed values`
                        : tree?.kind === "artifact_tensor_namespace"
                          ? "Tensor-name grouping · no weights or activations read"
                          : "Runtime containment · no activation captured"}
                  </span>
                  <span>
                    Drag to orbit · right-drag to pan · scroll to zoom
                  </span>
                </div>
                {event &&
                  !(view === "model" && expandedComponents.length > 0) && (
                    <div className="legend">
                      <span className="negative">Negative</span>
                      <span>0</span>
                      <span className="positive">Positive</span>
                      <small>
                        {logScale ? "Signed log1p" : "Linear"} · ±
                        {num(Math.max(...vals.map(Math.abs), 0))} · null = gray
                      </small>
                    </div>
                  )}
              </>
            )}
          </section>
          <aside className="inspector panel">
            {expandedComponents.length > 0 && (
              <nav
                className="component-shortcuts"
                aria-label="Expanded layer components"
              >
                {expandedComponents.map((c) => (
                  <button
                    key={c.path}
                    className={selected === c.path ? "active" : ""}
                    onClick={() => choose(c.path)}
                  >
                    {c.path.split(".").at(-1).replaceAll("_", " ")}
                  </button>
                ))}
              </nav>
            )}
            <h2>
              Component Inspector
              <span className="pill">
                {trace?.kind === "attention_capture"
                  ? `L${trace.layer}`
                  : short(selected)}
              </span>
            </h2>
            <div className="tabs">
              {[
                ["component", "Component"],
                ["values", "Values"],
                ["attention", "Attention"],
              ].map(([id, label]) => (
                <button
                  key={id}
                  className={panel === id ? "active" : ""}
                  onClick={() => setPanel(id)}
                >
                  {label}
                </button>
              ))}
            </div>
            {panel === "component" && (
              <>
                <h3>
                  {activeNode?.type?.split(".").pop() ||
                    event?.module ||
                    "Select a component"}
                </h3>
                <dl>
                  <dt>Path</dt>
                  <dd>{selected || "Root"}</dd>
                  <dt>Shape</dt>
                  <dd>{event?.shape?.join(" × ") || "Not captured"}</dd>
                  <dt>Data type</dt>
                  <dd>{event?.dtype || "Not captured"}</dd>
                  <dt>Selection</dt>
                  <dd>{event?.selection || "Not captured"}</dd>
                  <dt>Recorded / total</dt>
                  <dd>
                    {event
                      ? `${event.values?.flat().length || 0} / ${num(event.total_elements)}`
                      : "—"}
                  </dd>
                  <dt>Truncated</dt>
                  <dd>{event ? String(event.truncated ?? false) : "—"}</dd>
                  {event?.denoiser_call !== undefined && (
                    <>
                      <dt>Denoiser call</dt>
                      <dd>{event.denoiser_call}</dd>
                    </>
                  )}
                </dl>
                {activeNode?.parameters?.length > 0 && (
                  <>
                    <h3>Local parameters</h3>
                    <div className="param-list">
                      {activeNode.parameters.slice(0, 64).map((p) => (
                        <p key={p.name}>
                          <b>{p.name}</b>
                          <span>{p.shape.join(" × ")}</span>
                          <small>{p.dtype}</small>
                        </p>
                      ))}
                    </div>
                  </>
                )}
                <h3>Captured prefix statistics</h3>
                <dl>
                  <dt>Minimum</dt>
                  <dd>{vals.length ? num(Math.min(...vals)) : "—"}</dd>
                  <dt>Maximum</dt>
                  <dd>{vals.length ? num(Math.max(...vals)) : "—"}</dd>
                  <dt>Mean</dt>
                  <dd>
                    {vals.length
                      ? num(vals.reduce((a, b) => a + b, 0) / vals.length)
                      : "—"}
                  </dd>
                </dl>
                <p className="micro">
                  Statistics describe retained finite values only. A flattened
                  prefix is not a representative sample of the entire tensor.
                </p>
              </>
            )}
            {panel === "values" && (
              <>
                <h3>Captured values</h3>
                <p className="micro">
                  Flattened index · float32. Null means non-finite, not zero.
                  First 256 values.
                </p>
                <div className="values-grid">
                  {event?.values
                    ?.flat()
                    .slice(0, 256)
                    .map((v, i) => (
                      <div
                        key={i}
                        title={`Index ${i}: ${v}`}
                        style={{
                          background:
                            v === null
                              ? "#d2ccd3"
                              : v < 0
                                ? "#fbe0da"
                                : "#d8eee5",
                        }}
                      >
                        <small>{i}</small>
                        {v === null ? "null" : num(v)}
                      </div>
                    ))}
                </div>
                {!event && <p>No captured values.</p>}
              </>
            )}
            {panel === "attention" && (
              <>
                <h3>Attention head</h3>
                {trace?.kind === "attention_capture" ? (
                  <>
                    <p className="micro">
                      Layer {trace.layer} · query head {trace.head}
                      <br />
                      Rows: query position · columns: key position
                    </p>
                    <div
                      className="attention-map"
                      style={{
                        gridTemplateColumns: `repeat(${trace.shape[1]},1fr)`,
                      }}
                    >
                      {trace.values.flatMap((row, q) =>
                        row.map((v, k) => (
                          <div
                            key={`${q}:${k}`}
                            title={`Query ${q}, key ${k}: ${v.toFixed(5)}`}
                            style={{
                              background: `rgba(222,83,85,${Math.max(0.025, v)})`,
                            }}
                          />
                        )),
                      )}
                    </div>
                    <p className="micro">
                      0 = pale · 1 = coral. Attention weights do not establish
                      causality.
                    </p>
                  </>
                ) : (
                  <p className="micro">
                    Capture a head from a loaded text model. This separate
                    forward pass may materialize attention from all layers.
                  </p>
                )}
                <div className="two-fields">
                  <Field label="Layer (0-based)">
                    <input
                      type="number"
                      min="0"
                      max="1024"
                      value={layer}
                      onChange={(e) => setLayer(Number(e.target.value))}
                    />
                  </Field>
                  <Field label="Query head">
                    <input
                      type="number"
                      min="0"
                      max="1024"
                      value={head}
                      onChange={(e) => setHead(Number(e.target.value))}
                    />
                  </Field>
                </div>
                <Button
                  icon={Eye}
                  disabled={busy || state?.loaded?.task !== "text"}
                  onClick={() => run(true)}
                >
                  Capture attention
                </Button>
              </>
            )}
          </aside>
        </div>
      )}
      <section className="output panel">
        <div className="output-main">
          {!trace?.output?.top_tokens && (
            <h2>
              {trace?.preview ? "Generated output" : "Capture history"}
              <small>Local session · up to 8 traces / 32 MiB</small>
            </h2>
          )}
          {trace?.output?.top_tokens ? (
            <Prediction trace={trace} />
          ) : trace?.preview?.mime === "image/png" ? (
            <img
              className="output-image"
              alt="Decoded pipeline output"
              src={`data:image/png;base64,${trace.preview.base64}`}
            />
          ) : (
            <p className="muted">
              {state?.host === "comfyui"
                ? "Connect Model Run · Observe between a model loader and sampler. Captures appear here while ComfyUI runs."
                : "Captures stay in memory until exported. Use the history controls to reopen a trace."}
            </p>
          )}
        </div>
        <div className="history">
          {trace?.output?.top_tokens && (
            <div className="top-candidates">
              <h3>
                Top 5 <small>(prob.)</small>
              </h3>
              {trace.output.top_tokens.slice(0, 5).map((t) => (
                <div key={t.token_id}>
                  <span>{t.text ?? t.token_id}</span>
                  <span>{t.probability.toFixed(4)}</span>
                </div>
              ))}
            </div>
          )}
          <details
            open={!trace?.output?.top_tokens}
            className="history-disclosure"
          >
            <summary>Capture history</summary>
            <label htmlFor="history">Retained traces</label>
            <select
              id="history"
              value={traceId}
              onChange={(e) => action(() => openTrace(e.target.value))}
            >
              <option value="" disabled>
                Select a capture
              </option>
              {state?.traces.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name} · {Math.round(t.size_bytes / 1024)} KB
                </option>
              ))}
            </select>
            <div className="history-actions">
              <Button
                icon={DownloadSimple}
                disabled={!trace}
                onClick={exportTrace}
              >
                Export
              </Button>
              <Button
                icon={Trash}
                disabled={!traceId}
                aria-label="Delete selected trace"
                onClick={() =>
                  action(async () => {
                    await api(`delete/${traceId}`, {});
                    setTrace(null);
                    setTraceId("");
                    setTree(null);
                  })
                }
              />
              <Button
                icon={UploadSimple}
                aria-label="Import trace"
                onClick={() => importRef.current.click()}
              />
            </div>
            <p className="micro">
              Exports include captured values and decoded input tokens.
            </p>
          </details>
        </div>
      </section>
      {view !== "matrix" && view !== "compare" && (
        <footer className="timeline panel">
          <Button
            icon={playing ? Pause : Play}
            disabled={!events.length}
            aria-label={playing ? "Pause replay" : "Play replay"}
            onClick={() => {
              if (eventIndex === events.length - 1) setEventIndex(0);
              setPlaying((p) => !p);
            }}
          />
          <span className="timeline-count">
            {events.length ? eventIndex + 1 : 0} / {events.length}
            <small>Recorded event replay</small>
          </span>
          <div className="scrubber">
            <input
              aria-label="Recorded event"
              type="range"
              min="0"
              max={Math.max(0, events.length - 1)}
              value={eventIndex}
              disabled={!events.length}
              onChange={(e) => {
                const i = Number(e.target.value);
                setEventIndex(i);
                setSelected(eventPath(events[i]));
                setPlaying(false);
              }}
            />
            <div>
              <span>{short(events[0]?.module)}</span>
              <span>
                {event?.denoiser_call !== undefined
                  ? `Denoiser call ${event.denoiser_call}`
                  : event?.step !== undefined
                    ? `Diffusion step ${event.step}`
                    : short(event?.module)}
              </span>
              <span>{short(events.at(-1)?.module)}</span>
            </div>
          </div>
          <select
            aria-label="Replay speed"
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
          >
            {[0.5, 1, 2, 4].map((n) => (
              <option key={n} value={n}>
                {n}×
              </option>
            ))}
          </select>
        </footer>
      )}
      {modal && (
        <Modal
          title={modal === "models" ? "Model library" : "Capture settings"}
          close={() => setModal(null)}
        >
          {modal === "models" ? (
            <>
              <p className="muted">
                Models from configured ComfyUI paths. Local packages load in
                standalone mode; native checkpoints run through ComfyUI.
              </p>
              <div className="library-toolbar">
                <input
                  aria-label="Search models"
                  placeholder="Search models or formats…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
                <Button
                  icon={ArrowClockwise}
                  disabled={busy}
                  onClick={() =>
                    action(async () => {
                      await api("scan", {});
                      setInspection(null);
                    })
                  }
                >
                  Refresh
                </Button>
              </div>
              <div className="library-list">
                {catalog.artifacts
                  .filter(
                    (a) =>
                      !a.relative_path?.includes("/.cache/") &&
                      (a.format !== "unknown" ||
                        /\.(gguf|onnx|ckpt|pt|pth|bin)$/i.test(a.name)),
                  )
                  .filter((a) =>
                    `${a.name} ${a.format} ${a.relative_path}`
                      .toLowerCase()
                      .includes(search.toLowerCase()),
                  )
                  .map((a) => (
                    <div className="artifact" key={a.id}>
                      <div>
                        <b>{a.name}</b>
                        <span>
                          {a.format} · {a.status}
                          {a.tensor_count
                            ? ` · ${a.tensor_count.toLocaleString()} tensors`
                            : ""}
                        </span>
                        <small>{a.relative_path}</small>
                      </div>
                      <button
                        onClick={() =>
                          action(async () =>
                            setInspection(await api(`inspect/${a.id}`)),
                          )
                        }
                      >
                        Inspect
                      </button>
                      {["safetensors", "safetensors_index"].includes(
                        a.format,
                      ) &&
                        a.status === "metadata_inspected" && (
                          <button
                            onClick={() =>
                              action(async () => {
                                const structure = await api(
                                  `artifact-structure/${a.id}`,
                                );
                                setTree(structure);
                                setTrace(null);
                                setTraceId("");
                                setSelected("");
                                setCapturePaths([]);
                                setPlaying(false);
                                setView("model");
                                setPanel("component");
                                setModal(null);
                              })
                            }
                          >
                            Explore
                          </button>
                        )}
                      {a.task !== "inventory" &&
                        state?.host === "standalone" && (
                          <button
                            className="primary"
                            disabled={busy}
                            onClick={() =>
                              action(async () => {
                                await api("load", { id: a.id, device });
                                setModal(null);
                              })
                            }
                          >
                            Load
                          </button>
                        )}
                    </div>
                  ))}
                {!catalog.artifacts.length && (
                  <div className="library-empty">
                    <h3>No model paths configured yet</h3>
                    <p>
                      Start with <code>--models-dir</code> pointing to ComfyUI’s
                      models folder, or <code>--extra-paths</code> pointing to
                      its configuration. Inside ComfyUI, Model Run uses the live
                      folder registry.
                    </p>
                    <p>You can still import and explore traces.</p>
                  </div>
                )}
              </div>
              {catalog.errors?.length > 0 && (
                <details>
                  <summary>{catalog.errors.length} discovery notices</summary>
                  <pre>{JSON.stringify(catalog.errors, null, 2)}</pre>
                </details>
              )}
              {inspection && (
                <details open className="inspection">
                  <summary>Artifact metadata</summary>
                  <pre>{JSON.stringify(inspection, null, 2)}</pre>
                </details>
              )}
              <footer>
                <Field label="Load device">
                  <select
                    value={device}
                    onChange={(e) => setDevice(e.target.value)}
                  >
                    <option value="cuda">CUDA GPU</option>
                    <option value="cpu">CPU</option>
                  </select>
                </Field>
                <Button
                  icon={Stop}
                  disabled={busy || !state?.loaded}
                  onClick={() => action(() => api("unload", {}))}
                >
                  Unload model
                </Button>
              </footer>
            </>
          ) : native ? (
            <>
              <h3>Configure capture in the Observe node</h3>
              <p>
                Select module paths, the denoiser-call limit and values per
                event in <b>Model Run · Observe</b>. Use output selector JSON
                for multimodal tuples, such as an LTX audio branch. Queue and
                cancel execution in ComfyUI.
              </p>
              <p className="micro">
                The explorer replays captured calls. It does not start a second
                model or change the workflow's sampler.
              </p>
            </>
          ) : (
            <>
              <p className="muted">
                One model and observation at a time. Cancellation takes effect
                at supported runtime boundaries.
              </p>
              <div className="settings-grid">
                <Field label="Text input">
                  <select
                    value={inputMode}
                    onChange={(e) => setInputMode(e.target.value)}
                  >
                    <option value="raw">Raw prompt</option>
                    <option value="chat">Tokenizer chat template</option>
                  </select>
                </Field>
                <Field label="Values per event">
                  <select
                    value={elements}
                    onChange={(e) => setElements(Number(e.target.value))}
                  >
                    {[256, 1024, 2048, 4096].map((n) => (
                      <option key={n}>{n}</option>
                    ))}
                  </select>
                </Field>
                <Field label="Diffusion steps">
                  <input
                    type="number"
                    min="1"
                    max="32"
                    value={steps}
                    onChange={(e) => setSteps(Number(e.target.value))}
                  />
                </Field>
                <Field label="Image size">
                  <select
                    value={size}
                    onChange={(e) => setSize(Number(e.target.value))}
                  >
                    {[64, 128, 256, 512].map((n) => (
                      <option key={n}>{n}</option>
                    ))}
                  </select>
                </Field>
                <Field label="Seed">
                  <input
                    type="number"
                    min="0"
                    max="2147483647"
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                  />
                </Field>
              </div>
              <p>
                Text captures accept up to 128 tokens.{" "}
                {capturePaths.length
                  ? `${capturePaths.length} components selected.`
                  : "The first 32 numbered components are selected automatically."}{" "}
                Use Layer Stack to choose modules.
              </p>
              <p className="micro">
                3D shows up to 64 component groups and 2,048 values. Export
                preserves all retained values. Downloads and remote model code
                are disabled.
              </p>
            </>
          )}
        </Modal>
      )}
    </main>
  );
}
function Modal({ title, close, children }) {
  const ref = useRef();
  useEffect(() => {
    const previous = document.activeElement,
      el = ref.current;
    el.querySelector("button")?.focus();
    function key(e) {
      if (e.key !== "Tab") return;
      const items = [
          ...el.querySelectorAll(
            "button:not(:disabled),input:not(:disabled),select:not(:disabled),summary",
          ),
        ],
        first = items[0],
        last = items.at(-1);
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last?.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first?.focus();
      }
    }
    el.addEventListener("keydown", key);
    return () => {
      el.removeEventListener("keydown", key);
      previous?.focus();
    };
  }, []);
  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <section
        ref={ref}
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
      >
        <header>
          <h2 id="modal-title">{title}</h2>
          <button aria-label="Close dialog" onClick={close}>
            <X size={22} />
          </button>
        </header>
        {children}
      </section>
    </div>
  );
}
