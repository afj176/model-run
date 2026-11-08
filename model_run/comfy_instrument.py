"""Model Run's recording nodes added to a ComfyUI prompt (the graph ComfyUI queues), and a run's settings applied.

One copy serves both the canvas (Observe this workflow) and Model Run's own Run. Swaps keep each node's id,
so links into it still hold. Pure, so it is tested without ComfyUI.
"""
from copy import deepcopy
import re

from . import capture_limits

SAMPLER = re.compile(r"(Sampler|Guider)")
SWAPS = {"CLIPTextEncode": "ModelRunTextEncode", "VAEDecode": "ModelRunVAEDecode", "VAEEncode": "ModelRunVAEEncode"}
GENERATION = ("temperature", "top_k", "top_p", "min_p", "repetition_penalty", "seed")
# Nodes that make the empty latent a run starts from, and so set its size.
LATENT = re.compile(r"^Empty.*(Latent|Image)")
SEED_INPUTS = ("seed", "noise_seed")
SETTINGS = ("prompt", "seed", "steps", "width", "height", "files")
# Nodes that read a file you give a workflow: the input that names it, and what kind of file it is.
MEDIA = {"LoadImage": ("image", "image"), "LoadImageMask": ("image", "image"), "LoadVideo": ("file", "video"), "LoadAudio": ("audio", "audio")}


def media_inputs(prompt):
    """The files a workflow reads: [{node, kind, file, title}]."""
    found = []
    for node_id, node in prompt.items():
        spec = MEDIA.get(node.get("class_type"))
        if spec and isinstance(node.get("inputs", {}).get(spec[0]), str):
            found.append({"node": node_id, "kind": spec[1], "file": node["inputs"][spec[0]], "title": (node.get("_meta") or {}).get("title") or node["class_type"]})
    return found


# Nodes a run cannot lose: leaving out a file must not take one of these with it.
ESSENTIAL = re.compile(r"(Sampler|Guider|TextGenerate|Encode|Decode|Save)")
# Where a file is truly optional: a language model's image, video or audio. Elsewhere an optional
# image input (an edit model's reference, say) is what the workflow is for.
OPTIONAL_MEDIA = {"TextGenerate", "ModelRunTextGenerate"}


def drop_optional(prompt, node_ids, definitions):
    """Leave out file nodes whose file is missing, with the nodes that only exist to process that file;
    the ids of the files a run still needs.

    Leaving a file out removes every node with a required input fed by it, and so on downstream. That is
    allowed only when none of those is a sampler, generator, encoder, decoder or saver, and when what is
    left reads them only as a language model's optional media."""
    def required(node_id, name):
        spec = (definitions.get(prompt[node_id]["class_type"]) or {}).get("input", {})
        return name in (spec.get("required") or {}) or not spec

    needed = []
    for start in node_ids:
        if start not in prompt:
            continue
        removed, frontier = {start}, [start]
        while frontier:
            current = frontier.pop()
            for other, node in prompt.items():
                if other in removed:
                    continue
                if any(_is_link(v) and v[0] == current and required(other, k) for k, v in node.get("inputs", {}).items()):
                    removed.add(other)
                    frontier.append(other)
        boundary = {other for other, node in prompt.items() if other not in removed
                    for v in node.get("inputs", {}).values() if _is_link(v) and v[0] in removed}
        if any(ESSENTIAL.search(prompt[n]["class_type"]) for n in removed if n != start) or                 any(prompt[n]["class_type"] not in OPTIONAL_MEDIA for n in boundary):
            needed.append(start)
            continue
        for node_id in removed:
            del prompt[node_id]
        for node in prompt.values():
            for name in [k for k, v in node.get("inputs", {}).items() if _is_link(v) and v[0] in removed]:
                del node["inputs"][name]
    return needed


def _is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)


def _fresh(prompt, base):
    n = 1
    while f"{base}{n}" in prompt:
        n += 1
    return f"{base}{n}"


# Nodes that hand a string on unchanged, and the input it arrives on.
PASS_THROUGH = {"PreviewAny": "source", "Reroute": "", "PrimitiveStringMultiline": "value", "PrimitiveString": "value"}
TEXT_INPUTS = ("text", "prompt", "string", "value")


def _text_source(prompt, value, depth=8):
    """Where the text that reaches an input is written: (node id, input name) of the literal string,
    following links through pass-through nodes and the branch a switch has chosen. None when the
    text is made by a node (a language model, say) or the chain cannot be read."""
    for _ in range(depth):
        if not _is_link(value) or value[0] not in prompt:
            return None
        node_id = value[0]
        inputs = prompt[node_id].get("inputs", {})
        kind = prompt[node_id].get("class_type", "")
        if kind == "ComfySwitchNode":
            branch = "on_true" if inputs.get("switch") else "on_false"
            value = inputs.get(branch)
            if isinstance(value, str):
                return node_id, branch
            continue
        name = PASS_THROUGH.get(kind)
        if name is None:
            links = [v for v in inputs.values() if _is_link(v)]
            name = next((k for k, v in inputs.items() if _is_link(v)), "") if len(links) == 1 and len(inputs) == 1 else None
        if name is None:
            return None
        if not name:
            links = [v for v in inputs.values() if _is_link(v)]
            if len(links) != 1:
                return None
            value = links[0]
            continue
        value = inputs.get(name)
        if isinstance(value, str):
            return node_id, name
    return None


def _positive_text(prompt, link):
    """The node holding the text behind a sampler's positive conditioning, a few steps back at most.

    When the encoder's text comes down a link, the literal it leads to is the place: through a
    preview or a primitive, and down the branch a switch is set to, so the prompt reaches the
    sampler the way the workflow is wired today."""
    current = link
    for _ in range(8):
        if not _is_link(current) or current[0] not in prompt:
            return None
        node = prompt[current[0]]
        inputs = node.get("inputs", {})
        for key in ("text", "prompt"):
            if isinstance(inputs.get(key), str):
                return current[0], key
            if _is_link(inputs.get(key)) and _is_link(inputs.get("clip")):
                return _text_source(prompt, inputs[key])
        current = inputs.get("conditioning") or inputs.get("positive")
    return None


def _encoder(prompt, link):
    """The text encoder behind a sampler's positive conditioning: (node id, the input its text arrives on), or None."""
    current = link
    for _ in range(8):
        if not _is_link(current) or current[0] not in prompt:
            return None
        inputs = prompt[current[0]].get("inputs", {})
        if _is_link(inputs.get("clip")):
            key = next((k for k in ("text", "prompt") if isinstance(inputs.get(k), str) or _is_link(inputs.get(k))), None)
            return (current[0], key) if key else None
        current = inputs.get("conditioning") or inputs.get("positive")
    return None


def _samplers(prompt):
    return [(node_id, node) for node_id, node in prompt.items() if SAMPLER.search(node.get("class_type", ""))]


def read_settings(prompt):
    """What a run will use, as far as it can be read: the positive prompt, seed, steps and size."""
    found = {}
    for _, node in _samplers(prompt):
        inputs = node.get("inputs", {})
        located = _positive_text(prompt, inputs.get("positive"))
        if located and "prompt" not in found:
            found["prompt"] = prompt[located[0]]["inputs"][located[1]]
        for key in SEED_INPUTS:
            if isinstance(inputs.get(key), int) and "seed" not in found:
                found["seed"] = inputs[key]
        if isinstance(inputs.get("steps"), int) and "steps" not in found:
            found["steps"] = inputs["steps"]
    for node in prompt.values():
        inputs = node.get("inputs", {})
        if "prompt" not in found and node.get("class_type") == "TextGenerate" and isinstance(inputs.get("prompt"), str):
            found["prompt"] = inputs["prompt"]
        if "seed" not in found and node.get("class_type") == "RandomNoise" and isinstance(inputs.get("noise_seed"), int):
            found["seed"] = inputs["noise_seed"]
        if "steps" not in found and "Scheduler" in node.get("class_type", "") and isinstance(inputs.get("steps"), int):
            found["steps"] = inputs["steps"]
        if LATENT.search(node.get("class_type", "")) and isinstance(inputs.get("width"), int) and "width" not in found:
            found["width"], found["height"] = inputs["width"], inputs.get("height")
    return found


def apply_settings(prompt, settings):
    """Set the positive prompt, seed, steps and size where the graph holds them as plain values; what changed, in words."""
    changes = []
    text = settings.get("prompt")
    if isinstance(text, str) and text.strip():
        for _, node in _samplers(prompt):
            located = _positive_text(prompt, node.get("inputs", {}).get("positive"))
            if located:
                prompt[located[0]]["inputs"][located[1]] = text
                changes.append("Set the positive prompt")
                break
        else:
            for node in prompt.values():
                if node.get("class_type") == "TextGenerate" and isinstance(node["inputs"].get("prompt"), str):
                    node["inputs"]["prompt"] = text
                    changes.append("Set the prompt")
    for key, value in (("seed", settings.get("seed")), ("steps", settings.get("steps"))):
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if key == "seed" else 1):
            continue
        targets = [node["inputs"] for _, node in _samplers(prompt)]
        if key == "seed":
            targets += [n["inputs"] for n in prompt.values() if n.get("class_type") == "RandomNoise"]
        else:
            targets += [n["inputs"] for n in prompt.values() if "Scheduler" in n.get("class_type", "")]
        names = SEED_INPUTS if key == "seed" else ("steps",)
        touched = 0
        for inputs in targets:
            for name in names:
                if isinstance(inputs.get(name), int) and not isinstance(inputs.get(name), bool):
                    inputs[name] = value
                    touched += 1
        if touched:
            changes.append(f"Set {key} to {value}")
    files = settings.get("files")
    if isinstance(files, dict):
        for node_id, name in files.items():
            node = prompt.get(str(node_id))
            spec = MEDIA.get((node or {}).get("class_type"))
            # A file in ComfyUI's input folder, as its upload names it ("photo.png" or "subfolder/photo.png").
            if spec and isinstance(name, str) and 0 < len(name) <= 512 and ".." not in name.replace("\\", "/").split("/"):
                node["inputs"][spec[0]] = name
                changes.append(f"Use {name} in {node_id}")
    width, height = settings.get("width"), settings.get("height")
    if all(isinstance(v, int) and not isinstance(v, bool) and 16 <= v <= 16384 for v in (width, height)):
        for node in prompt.values():
            inputs = node.get("inputs", {})
            if LATENT.search(node.get("class_type", "")) and isinstance(inputs.get("width"), int) and isinstance(inputs.get("height"), int):
                inputs["width"], inputs["height"] = width, height
                changes.append(f"Set the size to {width} × {height}")
    return changes


def instrument(source, settings=None):
    """The prompt with Model Run's nodes added and the settings applied: {prompt, changes, observed}."""
    prompt = deepcopy(source if isinstance(source, dict) else {})
    if any(str(n.get("class_type", "")).startswith("ModelRun") for n in prompt.values() if isinstance(n, dict)):
        changes = apply_settings(prompt, settings or {})
        return {"prompt": prompt, "changes": changes, "observed": True}
    changes = apply_settings(prompt, settings or {})
    text = (settings or {}).get("prompt") or read_settings(prompt).get("prompt") or ""

    # Observe between each model and the samplers that use it: one observer per model, attention found automatically.
    observers = {}
    for node_id, node in list(prompt.items()):
        model = node.get("inputs", {}).get("model")
        if not SAMPLER.search(node.get("class_type", "")) or not _is_link(model):
            continue
        key = tuple(model)
        if key not in observers:
            observe = _fresh(prompt, "model_run_observe_")
            prompt[observe] = {"class_type": "ModelRunObserve", "_meta": {"title": "Model Run · Observe"},
                               "inputs": {"model": model, "modules": "", "max_calls": capture_limits.get("layer_calls"), "elements": 1024, "output_selectors": "{}",
                                          "attention_block": "auto", "attention_grid": 32, "prompt": text if isinstance(text, str) else "",
                                          "denoising_views": True, "ablate_tokens": ""}}
            # Word labels for the attention maps come from the text encoder that read the prompt, and
            # from the text it was given: a literal, or the link that brings it (a rewriter's output, say).
            encoder = _encoder(prompt, node.get("inputs", {}).get("positive"))
            if encoder:
                inputs = prompt[encoder[0]]["inputs"]
                prompt[observe]["inputs"]["clip"] = inputs["clip"]
                prompt[observe]["inputs"]["prompt"] = inputs[encoder[1]]
            observers[key] = observe
            changes.append(f"Observe the model of {node['class_type']} ({node_id})")
        node["inputs"]["model"] = [observers[key], 0]

    for node_id, node in prompt.items():
        swap = SWAPS.get(node.get("class_type"))
        if swap:
            node["class_type"] = swap
            changes.append(f"Record {node_id} with {swap}")
        if node.get("class_type") == "TextGenerate" and not node["inputs"].get("video") and not node["inputs"].get("audio"):
            inputs = node["inputs"]
            mode = inputs.get("sampling_mode")
            sampled = (mode.get("sampling_mode") if isinstance(mode, dict) else mode) == "on"
            def value(key):
                if isinstance(mode, dict) and key in mode:
                    return mode[key]
                return inputs.get(f"sampling_mode.{key}", inputs.get(key))
            node["class_type"] = "ModelRunTextGenerate"
            node["inputs"] = {"clip": inputs.get("clip"), "prompt": inputs.get("prompt", ""), "max_length": inputs.get("max_length", 256), "sampling": sampled,
                              **{k: value(k) for k in GENERATION if value(k) is not None},
                              **{k: inputs[k] for k in ("image", "thinking", "use_default_template", "system_prompt") if inputs.get(k) not in (None, "")}}
            changes.append(f"Record {node_id} with ModelRunTextGenerate")

    # Pin the decoded image to the observed run, so Model Run shows what the steps made.
    observe = next(iter(observers.values()), None)
    decode = next((i for i, n in prompt.items() if n.get("class_type") == "ModelRunVAEDecode"), None)
    if observe and decode:
        prompt[_fresh(prompt, "model_run_attach_")] = {"class_type": "ModelRunAttachOutput", "_meta": {"title": "Model Run · Attach Output"},
                                                       "inputs": {"observed_model": [observe, 0], "images": [decode, 0]}}
        changes.append("Attach the decoded image to the run")
    return {"prompt": prompt, "changes": changes, "observed": bool(observers) or any(n.get("class_type", "").startswith("ModelRun") for n in prompt.values())}
