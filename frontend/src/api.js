import { unpackValues } from "./packedValues.js";
let token;
export const eventPath = (event) =>
  event?.module ?? (event?.step !== undefined ? `Step ${event.step}` : "");
const base = new URL("./api/", window.location.href);
export async function api(route, body, refreshed = false) {
  if (!token) {
    const r = await fetch(new URL("session", base));
    if (!r.ok)
      throw new Error(
        "Cannot connect to Model Run. Open the Python service URL.",
      );
    token = (await r.json()).token;
  }
  const r = await fetch(new URL(route, base), {
    method: body === undefined ? "GET" : "POST",
    headers: {
      "X-Model-Run-Token": token,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value = await r.json();
  if (
    r.status === 403 &&
    value.error === "Missing application session token" &&
    !refreshed
  ) {
    token = undefined;
    return api(route, body, true);
  }
  if (!r.ok) throw new Error(value.error || `Request failed (${r.status})`);
  // Long value lists arrive packed; every view reads plain numbers.
  return route.startsWith("traces/") ? unpackValues(value) : value;
}
export function displayNodes(tree, trace) {
  const nodes = tree?.nodes || [];
  // Outermost only: a UNet's input_blocks.4 holds transformer_blocks.0, which is not a layer of the model.
  const stack = /(?:^|\.)[a-z_]*(?:blocks?|layers?)\.\d+(?=\.|$)/;
  const blocks = nodes.filter((n) => {
    const match = stack.exec(n.path);
    return match && match.index + match[0].length === n.path.length;
  });
  // A text prompt runs the language model's stack, so it comes before any image or audio tower.
  const primary = tree?.primary_prefix;
  const own = primary
    ? blocks.filter((n) => n.path.startsWith(`${primary}.`))
    : [];
  if (own.length) return own.slice(0, 64);
  if (blocks.length) return blocks.slice(0, 64);
  if (nodes.length) {
    const roots = nodes.filter((n) => n.path && n.path.split(".").length <= 2);
    return (roots.length ? roots : nodes).slice(0, 64);
  }
  return [
    ...new Set((trace?.events || []).map((e) => e.module ?? `Step ${e.step}`)),
  ]
    .slice(0, 64)
    .map((path) => ({ path, type: "Captured output" }));
}
export function samples(event) {
  return (event?.values || [])
    .flat()
    .filter((v) => v !== null && Number.isFinite(v));
}
