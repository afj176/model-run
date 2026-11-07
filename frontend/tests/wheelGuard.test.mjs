import test from "node:test";
import assert from "node:assert/strict";
import { guardWheel, wheelZooms } from "../src/wheelGuard.js";

test("the wheel scrolls the page unless Ctrl or ⌘ is held or the view was clicked", () => {
  assert.equal(wheelZooms({}, false), false);
  assert.equal(wheelZooms({ ctrlKey: true }, false), true);
  assert.equal(wheelZooms({ metaKey: true }, false), true);
  assert.equal(wheelZooms({}, true), true);
});

test("clicking the view lets the wheel zoom until the pointer leaves", () => {
  const listeners = {};
  const element = {
    addEventListener: (name, fn) => (listeners[name] = fn),
    removeEventListener: (name) => delete listeners[name],
  };
  const controls = { enableZoom: true };
  const release = guardWheel(element, controls);
  listeners.wheel({});
  assert.equal(controls.enableZoom, false);
  listeners.pointerdown();
  listeners.wheel({});
  assert.equal(controls.enableZoom, true);
  listeners.pointerleave();
  listeners.wheel({});
  assert.equal(controls.enableZoom, false);
  release();
  assert.deepEqual(Object.keys(listeners), []);
});
