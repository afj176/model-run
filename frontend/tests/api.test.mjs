import test from "node:test";
import assert from "node:assert/strict";

test("refresh expired session once without replaying a transport failure", async () => {
  const originalFetch = globalThis.fetch;
  const originalWindow = globalThis.window;
  globalThis.window = { location: { href: "http://127.0.0.1:8766/" } };
  const requests = [];
  const responses = [
    [200, { token: "old" }],
    [403, { error: "Missing application session token" }],
    [200, { token: "new" }],
    [200, { accepted: true }],
  ];
  globalThis.fetch = async (url, options) => {
    requests.push([url.pathname, options]);
    const [status, value] = responses.shift();
    return { status, ok: status === 200, json: async () => value };
  };
  try {
    const { api, eventPath } = await import("../src/api.js");
    assert.equal(eventPath({ step: 2 }), "Step 2");
    assert.equal(eventPath({ module: "model.layers.12" }), "model.layers.12");
    assert.equal(eventPath({ module: "" }), "");
    assert.deepEqual(await api("run", { prompt: "fixture" }), {
      accepted: true,
    });
    assert.equal(requests[3][1].headers["X-Model-Run-Token"], "new");
    assert.equal(requests[1][1].body, requests[3][1].body);
    let attempts = 0;
    globalThis.fetch = async () => {
      attempts++;
      throw new Error("connection interrupted");
    };
    await assert.rejects(api("run", {}), /connection interrupted/);
    assert.equal(attempts, 1);
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.window = originalWindow;
  }
});

test("layers are the outermost stack, not blocks nested inside it", async () => {
  const originalWindow = globalThis.window;
  globalThis.window = { location: { href: "http://127.0.0.1:8766/" } };
  const { displayNodes } = await import("../src/api.js");
  globalThis.window = originalWindow;
  const paths = (list) =>
    displayNodes({ nodes: list.map((path) => ({ path })) }).map((n) => n.path);
  assert.deepEqual(
    paths([
      "input_blocks",
      "input_blocks.4",
      "input_blocks.4.1",
      "input_blocks.4.1.transformer_blocks.0",
      "middle_block.1",
      "output_blocks.0.0.emb_layers.1",
      "output_blocks.0",
    ]),
    ["input_blocks.4", "middle_block.1", "output_blocks.0"],
  );
  assert.deepEqual(
    paths(["double_blocks.0", "double_blocks.0.img_attn", "single_blocks.3"]),
    ["double_blocks.0", "single_blocks.3"],
  );
  assert.deepEqual(
    paths(["model.encoder.layers.2", "model.decoder.layers.0"]),
    ["model.encoder.layers.2", "model.decoder.layers.0"],
  );
});
