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

import { withoutPackageFiles } from "../src/libraryData.js";

test("a package folder stands for the files inside it", () => {
  const artifacts = [
    {
      name: "Qwen3-0.6B",
      format: "local_package",
      relative_path: "text_encoders/Qwen3-0.6B",
    },
    {
      name: "model.safetensors",
      format: "safetensors",
      relative_path: "text_encoders/Qwen3-0.6B/model.safetensors",
    },
    {
      name: "Qwen3-0.6B-extra.safetensors",
      format: "safetensors",
      relative_path: "text_encoders/Qwen3-0.6B-extra.safetensors",
    },
  ];
  assert.deepEqual(
    withoutPackageFiles(artifacts).map((a) => a.name),
    ["Qwen3-0.6B", "Qwen3-0.6B-extra.safetensors"],
  );
});

import { kindOf, startStage } from "../src/libraryData.js";

test("the start panel follows what the app is doing", () => {
  assert.equal(startStage(null), "choose");
  assert.equal(
    startStage({ host: "standalone", job: { status: "idle" } }),
    "choose",
  );
  assert.equal(
    startStage({
      host: "standalone",
      job: { status: "running", action: "load" },
    }),
    "loading",
  );
  assert.equal(
    startStage({
      host: "standalone",
      job: { status: "running", action: "observe" },
      loaded: { name: "q" },
    }),
    "ready",
  );
  assert.equal(
    startStage({ host: "standalone", loaded: { name: "q" } }),
    "ready",
  );
  assert.equal(startStage({ host: "comfyui", loaded: null }), "comfyui");
  // A package loaded inside ComfyUI runs in Model Run like the standalone app.
  assert.equal(startStage({ host: "comfyui", loaded: { name: "q" } }), "ready");
});

test("packages that run here say what kind of model they are", () => {
  assert.equal(kindOf({ task: "text" }), "Language model");
  assert.equal(kindOf({ task: "audio" }), "Speech recognition");
  assert.equal(kindOf({ task: "inventory" }), null);
});
