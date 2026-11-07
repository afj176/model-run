// Adds Model Run's recording nodes to a ComfyUI prompt (the API graph ComfyUI queues), leaving the canvas alone.
// Pure, so it is tested outside ComfyUI. Every swap keeps the node's id, so links into it still hold.

// Nodes that run a MODEL through sampling, and so take Model Run · Observe on their model input.
const SAMPLER = /(Sampler|Guider)/;
const SWAPS = {
  CLIPTextEncode: "ModelRunTextEncode",
  VAEDecode: "ModelRunVAEDecode",
  VAEEncode: "ModelRunVAEEncode",
};
const GENERATION = [
  "temperature",
  "top_k",
  "top_p",
  "min_p",
  "repetition_penalty",
  "seed",
];
const isLink = (value) =>
  Array.isArray(value) && value.length === 2 && typeof value[0] === "string";

function freshId(prompt, base) {
  let n = 1;
  while (prompt[`${base}${n}`]) n += 1;
  return `${base}${n}`;
}

/** A value from Generate Text's sampling options, wherever this ComfyUI version keeps it. */
function samplingValue(inputs, key) {
  const mode = inputs.sampling_mode;
  if (mode && typeof mode === "object" && key in mode) return mode[key];
  if (`sampling_mode.${key}` in inputs) return inputs[`sampling_mode.${key}`];
  return inputs[key];
}

/** The text node behind a sampler's positive conditioning, a few steps back at most. */
function positiveSource(prompt, link) {
  let current = link;
  for (let hops = 0; hops < 6 && isLink(current); hops += 1) {
    const node = prompt[current[0]];
    if (!node) return null;
    if (typeof node.inputs?.text === "string") return current[0];
    current = node.inputs?.conditioning ?? node.inputs?.positive;
  }
  return null;
}

/**
 * The prompt with Model Run's nodes in it, and what changed in words.
 * promptText replaces the positive prompt (or Generate Text's prompt) when given.
 */
export function instrument(source, { promptText } = {}) {
  const prompt = JSON.parse(JSON.stringify(source || {}));
  const changes = [];
  if (
    Object.values(prompt).some((n) => n?.class_type?.startsWith("ModelRun"))
  ) {
    return { prompt, changes, observed: true };
  }
  const text =
    typeof promptText === "string" && promptText.trim() ? promptText : null;

  // The positive prompt first, while the text nodes still have their own class.
  if (text) {
    for (const node of Object.values(prompt)) {
      if (!SAMPLER.test(node.class_type || "")) continue;
      const id = positiveSource(prompt, node.inputs?.positive);
      if (id) {
        prompt[id].inputs.text = text;
        changes.push("Set the positive prompt");
        break;
      }
    }
  }

  // Observe between each model and the sampler that uses it; one per model.
  const observers = new Map();
  for (const [id, node] of Object.entries(prompt)) {
    if (!SAMPLER.test(node.class_type || "") || !isLink(node.inputs?.model))
      continue;
    const key = node.inputs.model.join(":");
    if (!observers.has(key)) {
      const observe = freshId(prompt, "model_run_observe_");
      prompt[observe] = {
        class_type: "ModelRunObserve",
        inputs: {
          model: node.inputs.model,
          modules: "",
          max_calls: 8,
          elements: 1024,
          output_selectors: "{}",
          attention_block: "",
          attention_grid: 32,
          prompt: text || "",
          denoising_views: true,
          ablate_tokens: "",
        },
        _meta: { title: "Model Run · Observe" },
      };
      observers.set(key, observe);
      changes.push(`Observe the model of ${node.class_type} (${id})`);
    }
    node.inputs.model = [observers.get(key), 0];
  }

  for (const [id, node] of Object.entries(prompt)) {
    const swap = SWAPS[node.class_type];
    if (swap) {
      node.class_type = swap;
      changes.push(`Record ${id} with ${swap}`);
    }
    if (
      node.class_type === "TextGenerate" &&
      !node.inputs.video &&
      !node.inputs.audio
    ) {
      const inputs = node.inputs;
      const sampled =
        (typeof inputs.sampling_mode === "object"
          ? inputs.sampling_mode.sampling_mode
          : inputs.sampling_mode) === "on";
      if (text) inputs.prompt = text;
      node.class_type = "ModelRunTextGenerate";
      node.inputs = {
        clip: inputs.clip,
        prompt: inputs.prompt,
        max_length: inputs.max_length ?? 256,
        sampling: sampled,
        ...Object.fromEntries(
          GENERATION.map((key) => [key, samplingValue(inputs, key)]).filter(
            ([, value]) => value !== undefined,
          ),
        ),
        ...(inputs.image ? { image: inputs.image } : {}),
        ...(inputs.thinking !== undefined ? { thinking: inputs.thinking } : {}),
        ...(inputs.use_default_template !== undefined
          ? { use_default_template: inputs.use_default_template }
          : {}),
        ...(inputs.system_prompt
          ? { system_prompt: inputs.system_prompt }
          : {}),
      };
      changes.push(`Record ${id} with ModelRunTextGenerate`);
    }
  }

  // Pin the decoded image to the observed run, so Model Run shows what the steps made.
  const observe = observers.values().next().value;
  const decode = Object.keys(prompt).find(
    (id) => prompt[id].class_type === "ModelRunVAEDecode",
  );
  if (observe && decode) {
    prompt[freshId(prompt, "model_run_attach_")] = {
      class_type: "ModelRunAttachOutput",
      inputs: { observed_model: [observe, 0], images: [decode, 0] },
      _meta: { title: "Model Run · Attach Output" },
    };
    changes.push("Attach the decoded image to the run");
  }
  return { prompt, changes, observed: changes.length > 0 };
}
