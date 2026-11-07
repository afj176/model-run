import test from "node:test";
import assert from "node:assert/strict";
import { instrument } from "../../comfy_web/instrument.js";
import { comfyError, textModelPrompt } from "../src/comfyRun.js";

// A text-to-image prompt in ComfyUI's API format.
const graph = () => ({
  4: { class_type: "UNETLoader", inputs: { unet_name: "flux.safetensors" } },
  5: {
    class_type: "CLIPLoader",
    inputs: { clip_name: "t5.safetensors", type: "flux2" },
  },
  6: {
    class_type: "CLIPTextEncode",
    inputs: { clip: ["5", 0], text: "a cat" },
  },
  7: { class_type: "CLIPTextEncode", inputs: { clip: ["5", 0], text: "" } },
  8: {
    class_type: "FluxGuidance",
    inputs: { conditioning: ["6", 0], guidance: 3.5 },
  },
  9: {
    class_type: "KSampler",
    inputs: {
      model: ["4", 0],
      positive: ["8", 0],
      negative: ["7", 0],
      latent_image: ["10", 0],
    },
  },
  10: { class_type: "EmptyLatentImage", inputs: { width: 512, height: 512 } },
  11: {
    class_type: "VAEDecode",
    inputs: { samples: ["9", 0], vae: ["12", 0] },
  },
  12: { class_type: "VAELoader", inputs: { vae_name: "ae.safetensors" } },
  13: { class_type: "SaveImage", inputs: { images: ["11", 0] } },
});

test("a sampler's model goes through Observe, and the text and VAE nodes record", () => {
  const source = graph();
  const { prompt, changes, observed } = instrument(source);
  assert.ok(observed);
  const observe = prompt[9].inputs.model[0];
  assert.equal(prompt[observe].class_type, "ModelRunObserve");
  assert.deepEqual(prompt[observe].inputs.model, ["4", 0]);
  assert.equal(prompt[6].class_type, "ModelRunTextEncode");
  assert.equal(prompt[11].class_type, "ModelRunVAEDecode");
  const attach = Object.values(prompt).find(
    (n) => n.class_type === "ModelRunAttachOutput",
  );
  assert.deepEqual(attach.inputs, {
    observed_model: [observe, 0],
    images: ["11", 0],
  });
  assert.equal(source[9].inputs.model[0], "4"); // the original is untouched
  assert.ok(changes.length >= 4);
});

test("two samplers on one model share one observer", () => {
  const source = graph();
  source[20] = {
    class_type: "KSampler",
    inputs: { model: ["4", 0], positive: ["6", 0] },
  };
  const { prompt } = instrument(source);
  assert.equal(prompt[9].inputs.model[0], prompt[20].inputs.model[0]);
  assert.equal(
    Object.values(prompt).filter((n) => n.class_type === "ModelRunObserve")
      .length,
    1,
  );
});

test("a new prompt replaces the positive text, through conditioning steps", () => {
  const { prompt } = instrument(graph(), { promptText: "a lighthouse" });
  assert.equal(prompt[6].inputs.text, "a lighthouse");
  assert.equal(prompt[7].inputs.text, "");
});

test("Generate Text becomes Model Run's, keeping its settings", () => {
  const source = {
    1: {
      class_type: "CLIPLoader",
      inputs: { clip_name: "gemma.safetensors", type: "stable_diffusion" },
    },
    2: {
      class_type: "TextGenerate",
      inputs: {
        clip: ["1", 0],
        prompt: "hi",
        max_length: 99,
        sampling_mode: "on",
        "sampling_mode.temperature": 0.3,
        "sampling_mode.seed": 7,
      },
    },
    3: { class_type: "PreviewAny", inputs: { source: ["2", 0] } },
  };
  const { prompt } = instrument(source, { promptText: "hello" });
  assert.equal(prompt[2].class_type, "ModelRunTextGenerate");
  assert.equal(prompt[2].inputs.prompt, "hello");
  assert.equal(prompt[2].inputs.sampling, true);
  assert.equal(prompt[2].inputs.temperature, 0.3);
  assert.equal(prompt[2].inputs.seed, 7);
  assert.equal(prompt[2].inputs.max_length, 99);
});

test("a workflow Model Run already observes, or one with nothing to record, is left alone", () => {
  const done = instrument({ 1: { class_type: "ModelRunObserve", inputs: {} } });
  assert.equal(done.changes.length, 0);
  const nothing = instrument({ 1: { class_type: "LoadImage", inputs: {} } });
  assert.equal(nothing.observed, false);
});

test("a text model file runs through a small prompt that ends in an output node", () => {
  const generate = textModelPrompt({
    file: "g.safetensors",
    type: "stable_diffusion",
    mode: "generate",
    text: "hi",
    maxLength: 48,
  });
  assert.equal(generate[1].inputs.clip_name, "g.safetensors");
  assert.equal(generate[2].class_type, "ModelRunTextGenerate");
  assert.equal(generate[2].inputs.max_length, 48);
  assert.equal(generate[3].class_type, "PreviewAny");
  const encode = textModelPrompt({
    file: "q.safetensors",
    type: "qwen_image",
    mode: "encode",
    text: "hi",
  });
  assert.deepEqual(encode[2], {
    class_type: "ModelRunTextEncode",
    inputs: { clip: ["1", 0], text: "hi" },
  });
});

test("ComfyUI's refusals and failures read as one line", () => {
  assert.equal(
    comfyError({
      error: { message: "Prompt outputs failed validation" },
      node_errors: { 2: { errors: [{ details: "clip_name not in list" }] } },
    }),
    "Prompt outputs failed validation · clip_name not in list",
  );
  assert.equal(
    comfyError({
      status: {
        messages: [
          [
            "execution_error",
            { node_type: "CLIPLoader", exception_message: "out of memory" },
          ],
        ],
      },
    }),
    "CLIPLoader: out of memory",
  );
});
