import test from "node:test";
import assert from "node:assert/strict";
import {
  formatCount,
  groupArtifacts,
  groupOf,
  sectionOf,
} from "../src/libraryData.js";

test("files that run here come first, then ComfyUI's folders in a familiar order", () => {
  const artifacts = [
    { name: "b", relative_path: "vae/b.safetensors", task: "inventory" },
    {
      name: "a",
      relative_path: "diffusion_models/a.safetensors",
      task: "inventory",
    },
    { name: "q", relative_path: "text_encoders/qwen", task: "text" },
    {
      name: "x",
      relative_path: "my_custom_folder/x.safetensors",
      task: "inventory",
    },
    {
      name: "c",
      relative_path: "checkpoints/c.safetensors",
      task: "inventory",
    },
  ];
  assert.deepEqual(
    groupArtifacts(artifacts).map(([id, label, list]) => [
      id,
      label,
      list.map((a) => a.name),
    ]),
    [
      ["here", "Runs here", ["q"]],
      ["checkpoints", "Checkpoints", ["c"]],
      ["diffusion_models", "Diffusion models", ["a"]],
      ["vae", "VAEs", ["b"]],
      ["my_custom_folder", "my custom folder", ["x"]],
    ],
  );
  assert.equal(groupOf({ relative_path: "", task: "inventory" }), "other");
});

test("parameter counts read in words", () => {
  assert.equal(formatCount(3875544576), "3.88 billion");
  assert.equal(formatCount(135000000), "135.0 million");
  assert.equal(formatCount(4096), "4,096");
});

test("workflows fall into sections by how close they are to ready", () => {
  assert.equal(sectionOf({ missing: 0, installable: 0 }), "ready");
  assert.equal(sectionOf({ missing: 1, installable: 1 }), "one");
  assert.equal(sectionOf({ missing: 3, installable: 3 }), "more");
  assert.equal(sectionOf({ missing: 2, installable: 1 }), "blocked");
});
