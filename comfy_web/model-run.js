import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Model Run beside the canvas: a sidebar, an Observe command, and the workflows Model Run asks this tab to show.
const MODEL_RUN = "./model-run/";
const CHANNEL = "model-run";
const tabId = Math.random().toString(36).slice(2);

const openModelRun = () => window.open(MODEL_RUN, "model-run");

async function modelRunApi(route, body) {
  const session = await api.fetchApi("/model-run/api/session");
  const { token } = await session.json();
  const response = await api.fetchApi(`/model-run/api/${route}`, {
    method: body ? "POST" : "GET",
    headers: {
      "X-Model-Run-Token": token,
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  return value;
}

/** Queue the open canvas with Model Run's nodes added, by the server, for this run only. The canvas is not changed. */
async function observeAndQueue() {
  const graph = await app.graphToPrompt();
  const { prompt, changes, observed } = await modelRunApi("comfy/instrument", {
    prompt: graph.output,
  });
  if (!observed)
    throw new Error(
      "This workflow has no sampler, text encoder, VAE decode or Generate Text for Model Run to record.",
    );
  const response = await api.fetchApi("/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      client_id: api.clientId,
      extra_data: { extra_pnginfo: { workflow: graph.workflow } },
    }),
  });
  const result = await response.json();
  if (!response.ok || result.error) {
    const detail = result.error?.message || result.error || response.statusText;
    const nodes = Object.values(result.node_errors || {})
      .flatMap((n) => n.errors || [])
      .map((e) => e.details || e.message);
    throw new Error([detail, ...nodes].filter(Boolean).join(" · "));
  }
  return { changes, promptId: result.prompt_id };
}

// Model Run finds an open ComfyUI tab and shows the workflow it ran on the canvas, to change and run again.
const channel = new BroadcastChannel(CHANNEL);
channel.onmessage = async ({ data }) => {
  if (data?.type === "ping")
    channel.postMessage({ type: "pong", id: data.id, tab: tabId });
  if (data?.type !== "open-workflow" || data.tab !== tabId) return;
  try {
    await app.loadGraphData(data.workflow, true, true, data.name);
    window.focus();
    channel.postMessage({ type: "result", id: data.id, ok: true });
  } catch (error) {
    channel.postMessage({
      type: "result",
      id: data.id,
      ok: false,
      error: String(error?.message || error),
    });
  }
};

function button(label, primary) {
  const element = document.createElement("button");
  element.textContent = label;
  element.style.cssText = `display:block;width:100%;margin:0 0 8px;padding:10px 12px;border-radius:8px;cursor:pointer;font:inherit;border:1px solid ${primary ? "#4a3bb3" : "var(--border-color,#555)"};background:${primary ? "#4a3bb3" : "transparent"};color:${primary ? "#fff" : "inherit"}`;
  return element;
}

function renderSidebar(element) {
  element.innerHTML = "";
  element.style.cssText =
    "padding:16px;display:flex;flex-direction:column;gap:6px;font-size:13px";
  const title = document.createElement("h3");
  title.textContent = "Model Run";
  title.style.margin = "0 0 4px";
  const about = document.createElement("p");
  about.textContent =
    "Look inside the models this canvas runs: each layer, attention, denoising steps and what a language model nearly wrote.";
  about.style.cssText = "margin:0 0 10px;opacity:.8;line-height:1.4";
  const open = button("Open Model Run ↗", true);
  open.onclick = openModelRun;
  const observe = button("Observe this workflow", false);
  const status = document.createElement("p");
  status.style.cssText = "margin:0;opacity:.85;line-height:1.4";
  status.setAttribute("role", "status");
  observe.onclick = async () => {
    observe.disabled = true;
    status.textContent = "Adding Model Run's nodes and queueing…";
    try {
      const { changes } = await observeAndQueue();
      status.textContent = `Queued. ${changes.length} change${changes.length === 1 ? "" : "s"} for this run only; your canvas is unchanged. Open Model Run to watch it.`;
    } catch (error) {
      status.textContent = String(error?.message || error);
    } finally {
      observe.disabled = false;
    }
  };
  const recent = document.createElement("div");
  recent.style.cssText = "margin-top:14px";
  element.append(title, about, open, observe, status, recent);
  modelRunApi("state")
    .then((state) => {
      const traces = (state.traces || []).slice(-6).reverse();
      recent.innerHTML = "";
      const heading = document.createElement("h4");
      heading.textContent = traces.length
        ? "Recent captures"
        : "No captures yet";
      heading.style.margin = "0 0 6px";
      recent.append(heading);
      for (const trace of traces) {
        const row = document.createElement("a");
        row.textContent = `${trace.name} · ${trace.kind.replace(/_/g, " ")}`;
        row.href = MODEL_RUN;
        row.target = "model-run";
        row.style.cssText =
          "display:block;padding:4px 0;color:inherit;opacity:.85";
        recent.append(row);
      }
    })
    .catch(() => {});
}

app.registerExtension({
  name: "ModelRun.Observatory",
  commands: [
    {
      id: "ModelRun.Open",
      label: "Open Model Run",
      icon: "pi pi-chart-bar",
      function: openModelRun,
    },
    {
      id: "ModelRun.Observe",
      label: "Observe this workflow in Model Run",
      icon: "pi pi-eye",
      function: () =>
        observeAndQueue()
          .then(openModelRun)
          .catch((error) => alert(String(error?.message || error))),
    },
  ],
  menuCommands: [
    { path: ["Model Run"], commands: ["ModelRun.Open", "ModelRun.Observe"] },
  ],
  setup() {
    app.extensionManager?.registerSidebarTab?.({
      id: "model-run",
      icon: "pi pi-chart-bar",
      title: "Model Run",
      tooltip: "Model Run: look inside this workflow's models",
      type: "custom",
      render: renderSidebar,
    });
  },
});
