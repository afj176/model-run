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

/** Queue a prompt on ComfyUI and return its id. */
export async function queuePrompt(prompt) {
  const response = await fetch(comfyUrl("prompt"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, client_id: "model-run" }),
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok || !value.prompt_id) throw new Error(comfyError(value));
  return value.prompt_id;
}

/** Wait until ComfyUI finishes a prompt; throws with its error when it fails. */
export async function waitForPrompt(id, { timeoutMs = 30 * 60 * 1000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 1500));
    const history = await (await fetch(comfyUrl(`history/${id}`))).json();
    const record = history[id];
    if (!record) continue;
    if (record.status?.status_str !== "success")
      throw new Error(comfyError(record));
    return record;
  }
  throw new Error("ComfyUI is still running this; it continues in ComfyUI");
}

/** Ask an open ComfyUI tab to load a workflow template and queue it with Model Run's nodes added. */
export function runTemplateInComfy(name, { timeoutMs = 2000 } = {}) {
  return new Promise((resolve, reject) => {
    const channel = new BroadcastChannel("model-run");
    const id = Math.random().toString(36).slice(2);
    let tab = null;
    const done = (fn, value) => {
      clearTimeout(timer);
      channel.close();
      fn(value);
    };
    const timer = setTimeout(
      () =>
        done(
          reject,
          new Error(
            "Open ComfyUI in another tab of this browser, then run the workflow again. Model Run hands workflows to ComfyUI's canvas to run.",
          ),
        ),
      timeoutMs,
    );
    channel.onmessage = ({ data }) => {
      if (data?.id !== id) return;
      if (data.type === "pong" && !tab) {
        tab = data.tab;
        clearTimeout(timer);
        channel.postMessage({ type: "run-template", id, tab, name });
      }
      if (data.type === "result")
        done(
          data.ok ? resolve : reject,
          data.ok ? data : new Error(data.error),
        );
    };
    channel.postMessage({ type: "ping", id });
  });
}
