// Running ComfyUI models from Model Run, when Model Run is served by ComfyUI. Model Run's page and ComfyUI's
// share an origin, so this page queues prompts on ComfyUI's own API and asks an open ComfyUI tab to run templates.

/** A small prompt that loads one text model file and records it: Generate Text, or Text Encode for an encoder. */
export function textModelPrompt({ file, type, mode, text, maxLength = 256 }) {
  const load = {
    class_type: "CLIPLoader",
    inputs: { clip_name: file, type, device: "default" },
  };
  const record =
    mode === "generate"
      ? {
          class_type: "ModelRunTextGenerate",
          inputs: {
            clip: ["1", 0],
            prompt: text,
            max_length: maxLength,
            sampling: false,
            temperature: 0.7,
            top_k: 64,
            top_p: 0.95,
            min_p: 0.05,
            repetition_penalty: 1.05,
            seed: 0,
          },
        }
      : { class_type: "ModelRunTextEncode", inputs: { clip: ["1", 0], text } };
  // ComfyUI runs only what reaches an output node; Preview Any shows the result on the canvas too.
  return {
    1: load,
    2: record,
    3: { class_type: "PreviewAny", inputs: { source: ["2", 0] } },
  };
}

/** The reason ComfyUI gives for refusing or failing a prompt, in one line. */
export function comfyError(value) {
  if (!value) return "ComfyUI did not say why";
  const node = Object.values(value.node_errors || {})
    .flatMap((n) => n.errors || [])
    .map((e) => e.details || e.message)
    .filter(Boolean);
  const failed = (value.status?.messages || []).find(
    ([kind]) => kind === "execution_error",
  )?.[1];
  return (
    [
      value.error?.message ||
        (typeof value.error === "string" ? value.error : null),
      ...node,
      failed && `${failed.node_type}: ${failed.exception_message}`.trim(),
    ]
      .filter(Boolean)
      .join(" · ") || "ComfyUI did not say why"
  );
}

const comfyUrl = (path) => new URL(`../${path}`, window.location.href);
// One id for this page, so ComfyUI sends it the progress of the runs it queued.
const clientId = `model-run-${Math.random().toString(36).slice(2)}`;
let socket = null;
const listeners = new Set();

function progressSocket() {
  if (socket && socket.readyState <= 1) return socket;
  const url = comfyUrl(`ws?clientId=${clientId}`);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  try {
    socket = new WebSocket(url);
    socket.onmessage = ({ data }) => {
      if (typeof data !== "string") return; // live preview images
      let message;
      try {
        message = JSON.parse(data);
      } catch {
        return;
      }
      for (const listener of listeners) listener(message);
    };
  } catch {
    socket = null; // progress is a nicety; history still says when the run ends
  }
  return socket;
}

/** Queue a prompt on ComfyUI and return its id. */
export async function queuePrompt(prompt) {
  progressSocket();
  const response = await fetch(comfyUrl("prompt"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, client_id: clientId }),
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok || !value.prompt_id) throw new Error(comfyError(value));
  return value.prompt_id;
}

/** What ComfyUI is doing for one prompt, in words, from its progress messages. */
export function progressText(message, promptId, nodes = {}) {
  const data = message?.data || {};
  if (data.prompt_id && data.prompt_id !== promptId) return null;
  if (message.type === "progress" && data.max)
    return `Step ${data.value} of ${data.max}`;
  if (message.type === "executing" && data.node)
    return `Running ${nodes[data.node] || data.node}`;
  if (message.type === "execution_cached") return "Reusing cached results";
  return null;
}

/**
 * Wait until ComfyUI finishes a prompt, telling onProgress what it is doing; throws with its error when it fails.
 * nodes names the prompt's nodes by id, for the progress text.
 */
export async function waitForPrompt(
  id,
  { onProgress, nodes, timeoutMs = 60 * 60 * 1000 } = {},
) {
  const listener = (message) => {
    const text = progressText(message, id, nodes);
    if (text) onProgress?.(text);
  };
  listeners.add(listener);
  try {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 1200));
      const history = await (await fetch(comfyUrl(`history/${id}`))).json();
      const record = history[id];
      if (!record) continue;
      if (record.status?.status_str !== "success")
        throw new Error(comfyError(record));
      return record;
    }
    throw new Error("ComfyUI is still running this; it continues in ComfyUI");
  } finally {
    listeners.delete(listener);
  }
}

/** Where ComfyUI serves a file from its input folder, by the name a Load node takes ("sub/name.png"). */
export function inputPreviewUrl(name) {
  const cut = name.lastIndexOf("/");
  const params = new URLSearchParams({
    filename: cut >= 0 ? name.slice(cut + 1) : name,
    subfolder: cut >= 0 ? name.slice(0, cut) : "",
    type: "input",
  });
  return comfyUrl(`view?${params}`).href;
}

/** Put a file of yours into ComfyUI's input folder; the name a workflow's Load node takes. */
export async function uploadInput(file) {
  const form = new FormData();
  form.append("image", file, file.name);
  form.append("type", "input");
  form.append("overwrite", "true");
  const response = await fetch(comfyUrl("upload/image"), {
    method: "POST",
    body: form,
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok || !value.name)
    throw new Error(value.error || "ComfyUI did not take the file");
  return value.subfolder ? `${value.subfolder}/${value.name}` : value.name;
}

/** Stop a prompt: take it off the queue, and interrupt it if it is running. */
export async function cancelPrompt(id) {
  const post = (path, body) =>
    fetch(comfyUrl(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  await post("queue", { delete: [id] });
  await post("interrupt", { prompt_id: id });
}

function ping(channel, timeoutMs) {
  return new Promise((resolve) => {
    const id = Math.random().toString(36).slice(2);
    const timer = setTimeout(() => {
      channel.removeEventListener("message", heard);
      resolve(null);
    }, timeoutMs);
    function heard({ data }) {
      if (data?.type === "pong" && data.id === id) {
        clearTimeout(timer);
        channel.removeEventListener("message", heard);
        resolve(data.tab);
      }
    }
    channel.addEventListener("message", heard);
    channel.postMessage({ type: "ping", id });
  });
}

/** Show a workflow template on the ComfyUI canvas: in an open ComfyUI tab, or a new one opened on it. */
export async function openInComfy(workflow, name) {
  const channel = new BroadcastChannel("model-run");
  try {
    const tab = await ping(channel, 700);
    if (!tab) {
      // ComfyUI opens a template named in its address.
      window.open(comfyUrl(`?template=${encodeURIComponent(name)}`), "comfyui");
      return;
    }
    const id = Math.random().toString(36).slice(2);
    await new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error("ComfyUI did not answer")),
        15000,
      );
      channel.onmessage = ({ data }) => {
        if (data?.type !== "result" || data.id !== id) return;
        clearTimeout(timer);
        (data.ok ? resolve : reject)(data.ok ? null : new Error(data.error));
      };
      channel.postMessage({ type: "open-workflow", id, tab, workflow, name });
    });
  } finally {
    channel.close();
  }
}
