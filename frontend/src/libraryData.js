// Pure helpers for the model library, kept apart from the components so they can be tested.

// Folders in the order a person looks for them; anything else follows by name.
const GROUPS = [
  ["here", "Runs here"],
  ["checkpoints", "Checkpoints"],
  ["diffusion_models", "Diffusion models"],
  ["text_encoders", "Text encoders"],
  ["clip_vision", "Vision encoders"],
  ["vae", "VAEs"],
  ["loras", "LoRAs"],
  ["controlnet", "ControlNets"],
  ["model_patches", "Model patches"],
  ["upscale_models", "Upscalers"],
];
const LABELS = Object.fromEntries(GROUPS);

/** Which group an artifact is listed under: packages this app loads first, then ComfyUI's folders. */
export const groupOf = (a) =>
  a.task && a.task !== "inventory"
    ? "here"
    : (a.relative_path || "").split("/")[0] || "other";

/** Artifacts in display order: a list of [group, label, artifacts]. */
export function groupArtifacts(artifacts) {
  const found = new Map();
  for (const a of artifacts) {
    const group = groupOf(a);
    if (!found.has(group)) found.set(group, []);
    found.get(group).push(a);
  }
  const known = GROUPS.map(([id]) => id).filter((id) => found.has(id));
  const rest = [...found.keys()].filter((id) => !LABELS[id]).sort();
  return [...known, ...rest].map((id) => [
    id,
    LABELS[id] || id.replace(/_/g, " "),
    found.get(id),
  ]);
}

export function formatCount(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} billion`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} million`;
  return n.toLocaleString("en-US");
}

// Sections of the workflow list, nearest to ready first; the server sorts within each.
export const SECTIONS = [
  ["ready", "Ready with your models"],
  ["one", "One file to get"],
  ["more", "More files to get"],
  ["blocked", "Needs a file this app cannot install"],
];

export const sectionOf = (w) =>
  !w.missing
    ? "ready"
    : w.installable < w.missing
      ? "blocked"
      : w.missing === 1
        ? "one"
        : "more";

/** Leave out files inside a listed package folder: the package row stands for them. */
export function withoutPackageFiles(artifacts) {
  const packages = artifacts
    .filter((a) => a.format === "local_package" && a.relative_path)
    .map((a) => `${a.relative_path}/`);
  return artifacts.filter(
    (a) =>
      a.format === "local_package" ||
      !packages.some((p) => (a.relative_path || "").startsWith(p)),
  );
}

/** Which start panel to show before a capture: comfyui, loading, ready or choose. */
export function startStage(state) {
  if (state?.host === "comfyui") return "comfyui";
  if (state?.job?.status === "running" && state.job.action === "load")
    return "loading";
  if (state?.loaded) return "ready";
  return "choose";
}

const KINDS = {
  text: "Language model",
  embedding: "Embedding model",
  image: "Image classifier",
  audio: "Speech recognition",
  diffusion: "Diffusion pipeline",
};

/** A plain name for what a package that runs here is, from the task the app detected. */
export const kindOf = (artifact) => KINDS[artifact.task] || null;
