// The head census: every head of an attention capture, with the habit Model Run reads off its
// matrix, its focus, and its ablation effect where a sweep ran. Pure functions; HeadCensus.jsx draws them.
import { attentionHeadProfile, classifyHead } from "./insights.js";

/** Short names for the habits classifyHead reports. */
export const ROLES = {
  "Attention sink": { key: "sink", label: "sink (first token)" },
  "Previous-token head": { key: "prev", label: "previous token" },
  "Self-attending head": { key: "self", label: "self" },
  "Focused head": { key: "focused", label: "focused" },
  "Diffuse head": { key: "diffuse", label: "diffuse" },
  "Mixed head": { key: "mixed", label: "mixed" },
};

/** The layers an attention capture kept, each with its heads' profiles, or only its mean when
 * per-head matrices were not kept. Ablation effects come from the sweeps attached to the trace. */
export function headCensus(trace) {
  if (!trace || trace.kind !== "attention_capture") return null;
  const stacks = trace.layer_stacks || [];
  const indices = [...new Set([trace.layer, ...stacks.map((s) => s.index)])]
    .filter(Number.isInteger)
    .sort((a, b) => a - b);
  const ablations = [
    ...(trace.ablations || []),
    ...(trace.ablation ? [trace.ablation] : []),
  ];
  const headCount =
    trace.head_count ?? (Array.isArray(trace.heads) ? trace.heads.length : 0);
  const layers = indices.map((index) => {
    const probed = index === trace.layer;
    const stack = stacks.find((s) => s.index === index);
    const matrices = probed ? trace.heads : stack?.heads;
    const sweep = ablations.find((a) => a?.layer === index);
    const effect = (head) =>
      sweep?.heads?.find((h) => h.head === head)?.kl_bits ?? null;
    if (Array.isArray(matrices))
      return {
        index,
        probed,
        heads: matrices.map((matrix, head) => {
          const profile = attentionHeadProfile(matrix);
          const role = profile ? classifyHead(profile).label : "Unknown";
          return {
            head,
            profile,
            role: ROLES[role]?.key ?? "unknown",
            roleLabel: role,
            focus: profile?.focus ?? 0,
            effect: effect(head),
          };
        }),
        swept: Boolean(sweep),
      };
    const matrix = probed ? trace.values : stack?.mean;
    const profile = matrix ? attentionHeadProfile(matrix) : null;
    const role = profile ? classifyHead(profile).label : "Unknown";
    return {
      index,
      probed,
      mean: {
        profile,
        role: ROLES[role]?.key ?? "unknown",
        roleLabel: role,
        focus: profile?.focus ?? 0,
      },
      heads: null,
      swept: Boolean(sweep),
      sweep: sweep
        ? sweep.heads.map((h) => ({ head: h.head, effect: h.kl_bits }))
        : null,
    };
  });
  return {
    layers,
    headCount,
    layerCount: trace.layer_count ?? null,
    strongest: layers
      .flatMap((l) => (l.heads || []).map((h) => ({ layer: l.index, ...h })))
      .filter((h) => h.effect !== null)
      .sort((a, b) => b.effect - a.effect)
      .slice(0, 3),
  };
}
