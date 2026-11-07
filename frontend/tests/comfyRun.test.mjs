import test from "node:test";
import assert from "node:assert/strict";
import { comfyError, progressText, textModelPrompt } from "../src/comfyRun.js";

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

test("ComfyUI's progress reads as words, for this run only", () => {
  const nodes = { 9: "KSampler" };
  assert.equal(
    progressText(
      { type: "progress", data: { value: 3, max: 25, prompt_id: "a" } },
      "a",
    ),
    "Step 3 of 25",
  );
  assert.equal(
    progressText(
      { type: "executing", data: { node: "9", prompt_id: "a" } },
      "a",
      nodes,
    ),
    "Running KSampler",
  );
  assert.equal(
    progressText(
      { type: "progress", data: { value: 1, max: 2, prompt_id: "b" } },
      "a",
    ),
    null,
  );
});
