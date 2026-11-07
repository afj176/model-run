import { useState } from "react";
import {
  FileAudio,
  FileImage,
  FileVideo,
  GearSix,
  X,
} from "@phosphor-icons/react";

const ICONS = { image: FileImage, video: FileVideo, audio: FileAudio };
const short = (name) =>
  name.length > 28 ? `${name.slice(0, 12)}…${name.slice(-12)}` : name;

/** One file a workflow reads: a chip that picks it, shows it, and clears it. */
function FileChip({ input, chosen, onPick, onClear }) {
  const [uploading, setUploading] = useState(false);
  const Icon = ICONS[input.kind] || FileImage;
  const label = input.kind[0].toUpperCase() + input.kind.slice(1);
  const state = chosen
    ? short(chosen.split("/").pop())
    : uploading
      ? "uploading…"
      : input.needed && !input.present
        ? "needed"
        : !input.present
          ? "optional"
          : short(input.file);
  return (
    <span
      className={`file-chip ${chosen ? "chosen" : ""} ${input.needed && !input.present && !chosen ? "needed" : ""}`}
    >
      <label title={`${input.title} (${input.node}) reads this ${input.kind}`}>
        <Icon size={16} aria-hidden="true" />
        <b>{label}</b>
        <span>{state}</span>
        <input
          type="file"
          className="sr-only"
          accept={`${input.kind}/*`}
          onChange={async (e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (!file) return;
            setUploading(true);
            try {
              await onPick(file);
            } finally {
              setUploading(false);
            }
          }}
        />
      </label>
      {chosen && (
        <button
          className="chip-clear"
          aria-label={`Clear the ${input.kind}`}
          onClick={onClear}
        >
          <X size={12} weight="bold" />
        </button>
      )}
    </span>
  );
}

/** A chosen workflow's files and settings, beside the prompt, and the way to its canvas. */
export default function WorkflowBar({
  model,
  settings,
  onPick,
  onClear,
  onSettings,
  onOpen,
}) {
  const facts = [
    settings.seed !== "" && settings.seed !== undefined
      ? `Seed ${String(settings.seed).length > 8 ? `${String(settings.seed).slice(0, 6)}…` : settings.seed}`
      : null,
    settings.steps ? `${settings.steps} steps` : null,
    settings.width && settings.height
      ? `${settings.width} × ${settings.height}`
      : null,
  ].filter(Boolean);
  return (
    <section className="workflow-bar" aria-label="Workflow files and settings">
      {model.inputs.length > 0 && (
        <div className="workflow-files" role="group" aria-label="Files">
          {model.inputs.map((input) => (
            <FileChip
              key={input.node}
              input={input}
              chosen={settings.files?.[input.node]}
              onPick={(file) => onPick(input.node, file)}
              onClear={() => onClear(input.node)}
            />
          ))}
        </div>
      )}
      {facts.length > 0 && (
        <button
          className="settings-chip"
          onClick={onSettings}
          title="Change the seed, steps and size"
        >
          <GearSix size={15} aria-hidden="true" /> {facts.join(" · ")}
        </button>
      )}
      <button
        className="host-link"
        title="Show this workflow on the ComfyUI canvas, to change it there"
        onClick={onOpen}
      >
        Open in ComfyUI ↗
      </button>
    </section>
  );
}
