"""Drop-in ComfyUI nodes for the text encoder and VAE: they do the node's usual work and record it."""
import logging
import re

from .capture import capture_forward
from .structure import inspect_module_tree
from .tensor_slices import SliceRecorder

MAX_LAYERS = 65
MAX_TOKENS = 64
SPECIAL_TOKEN = re.compile(r"<[^<>]+>")
MAX_UNITS = 64
MAX_ELEMENTS = 256
SLICE_VALUES = 524288
SLICE_PER_LAYER = 8192
ENCODER_LIMITATIONS = [
    "Activation is the root mean square of a token's vector at the output of each layer. It shows how strongly a layer responded, not what a word means.",
    "Units are each word's strongest in that layer. Which tokens move a unit says nothing about what it does on other text.",
    "The map is a principal component projection fitted to this prompt's words only, leaving out start and end tokens, so positions cannot be compared between captures. The variance shown says how much it leaves out.",
    "Padding tokens are left out of these views, though the model still receives them.",
]
VAE_LIMITATIONS = [
    "If the VAE decodes in tiles, because memory ran short, a layer is recorded for its first tile only.",
    "Feature maps show magnitude at the output of each block. They do not show what a channel detects.",
]


def stage_paths(root, prefix=""):
    """Modules that really run: containers such as a UNet's or VAE's plain lists never fire a hook."""
    import torch
    found = []
    base = torch.nn.Module.forward
    for name, child in root.named_children():
        path = f"{prefix}{name}"
        # A model may loop over a Sequential's children itself, so its own hook might never fire.
        if getattr(type(child), "forward", base) is not base and not isinstance(child, (torch.nn.ModuleList, torch.nn.Sequential)):
            if any(True for _ in child.parameters()):
                found.append(path)
        else:
            found.extend(stage_paths(child, f"{path}."))
    return found


def _thin(paths, limit=MAX_LAYERS):
    if len(paths) <= limit:
        return paths
    return [paths[round(i * (len(paths) - 1) / (limit - 1))] for i in range(limit)]


class _Run:
    """Lets capture_forward watch a call that is not a module's own forward."""
    @staticmethod
    def module(attribute, inner, call):
        import torch
        class Invocation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                setattr(self, attribute, inner)
                self.training = False  # do not call eval(): module state belongs to ComfyUI
            def train(self, mode=True):
                return self
            def forward(self):
                return call()
        return Invocation()


def _publish(name, trace, note):
    from .comfy_bridge import application
    trace["model_name"] = name
    try:
        application().publish(trace, name, note=note)
    except ValueError as exc:
        for event in trace.get("events", []):
            event.pop("slices", None)
        trace["limitations"].append("Frame and channel views were dropped from this trace so that it could be kept.")
        try:
            application().publish(trace, name, note=note)
        except Exception as retry:  # noqa: BLE001 - a workflow must not fail because a capture is too large
            logging.getLogger(__name__).warning("Model Run could not keep this capture: %s (%s)", retry, exc)
    except Exception as exc:  # noqa: BLE001 - see above
        logging.getLogger(__name__).warning("Model Run could not keep this capture: %s", exc)


def encoder_groups(stack):
    """{encoder attribute: [layer paths]}, with the embedding first where one can be found."""
    from .comfy_bridge import all_layer_paths
    groups = {}
    for path in [p for p in all_layer_paths(stack) if p]:
        groups.setdefault(path.split(".")[0], []).append(path)
    names = dict(stack.named_modules())
    for encoder, paths in groups.items():
        # Encoders differ in which of these they call, so watch them all; one that never runs records nothing.
        paths[:0] = [n for n in names if n.startswith(f"{encoder}.") and re.search(r"(^|\.)(embeddings|token_embedding|embed_tokens|shared)$", n)]
    return groups


def readable_piece(word):
    """Tokenizers mark spaces and line ends with their own symbols."""
    text = str(word).replace("</w>", " ").replace("\u0120", " ").replace("\u2581", " ").replace("\u010a", "\n")
    return text.strip() or ("\u23ce" if "\n" in text else text or str(word))


def encoder_tokens(clip, tokens, encoder):
    """Token ids and readable text for one encoder; None when its tokens cannot be told apart."""
    from .cross_attention import collapse_padding
    if not isinstance(tokens, dict):
        return None
    key = encoder[len("clip_"):] if encoder.startswith("clip_") else encoder
    candidates = {k: v for k, v in tokens.items() if isinstance(v, list)}
    value = candidates.get(key) or candidates.get(encoder) or (next(iter(candidates.values())) if len(candidates) == 1 else None)
    if not value:
        return None
    flat = [pair for chunk in value if isinstance(chunk, (list, tuple)) for pair in chunk if isinstance(pair, (list, tuple)) and len(pair) >= 2]
    ids = [pair[0] if isinstance(pair[0], int) else None for pair in flat]
    weights = [float(pair[1]) if isinstance(pair[1], (int, float)) else 1.0 for pair in flat]
    tokenizer = getattr(getattr(clip, "tokenizer", None), encoder, None) or getattr(clip, "tokenizer", None)
    labels = None
    try:
        if hasattr(tokenizer, "untokenize"):
            labels = [readable_piece(word) for _, word in tokenizer.untokenize([(p[0], p[1]) for p in flat])]
    except Exception:  # noqa: BLE001 - numbered tokens are still honest
        labels = None
    if labels is None or len(labels) != len(flat):
        labels = [f"#{i}" for i in range(len(flat))]
    return {"ids": ids, "weights": weights, "labels": collapse_padding(labels)}


def real_tokens(labels):
    """Positions that are not padding. Padding can be on either side; the first end token stays."""
    padding = re.compile(r"<\|?(pad|padding)\|?>", re.I)
    ending = re.compile(r"<\|?endoftext\|?>|</s>|<eos>", re.I)
    kept, ended = [], False
    for index, label in enumerate(labels):
        text = str(label).strip()
        if padding.fullmatch(text) or (ended and ending.fullmatch(text)):
            continue
        ended = ended or bool(ending.fullmatch(text))
        kept.append(index)
    return kept or list(range(len(labels)))


def encoder_section(encoder, states, tokens):
    """Per-token activity, strongest units and a principal component map for one encoder."""
    import torch
    if not states:
        return None
    width = next(iter(states.values())).shape[-1]
    states = {path: state for path, state in states.items() if state.shape[-1] == width}
    token_total = min(state.shape[0] for state in states.values())
    if tokens and len(tokens["labels"]) == token_total:
        labels, ids, weights = tokens["labels"], tokens["ids"], tokens["weights"]
        numbered = False
    else:
        labels, ids, weights = [f"#{i}" for i in range(token_total)], [None] * token_total, [1.0] * token_total
        numbered = True
    positions = real_tokens(labels)[:MAX_TOKENS]
    keep = len(positions)
    # Start and end tokens can be far stronger than any word, so they do not get to choose what is shown.
    words = [i for i, at in enumerate(positions) if not SPECIAL_TOKEN.fullmatch(str(labels[at]).strip())] or list(range(keep))
    per_token = max(1, MAX_UNITS // max(1, len(words)))
    layers = []
    for path, state in states.items():
        kept = state[positions].float()
        chosen = sorted({int(u) for i in words for u in torch.topk(kept[i].abs(), min(per_token, width)).indices})[:MAX_UNITS]
        units = torch.tensor(chosen)
        layers.append({"path": path, "token_rms": [round(float(v), 5) for v in kept.pow(2).mean(dim=-1).sqrt()],
                       "units": chosen, "unit_values": [[round(v, 4) for v in row] for row in kept[:, units].tolist()]})
    stacked = torch.stack([state[positions].float() for state in states.values()]).double()
    fitted = stacked[:, words].reshape(-1, width)
    mean = fitted.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(fitted - mean, full_matrices=False)
    points = (stacked.reshape(-1, width) - mean) @ vh[:2].T
    # A component's sign is arbitrary, so fix it for the same prompt to draw the same picture.
    points = points * torch.sign(points[points.abs().argmax(dim=0), torch.arange(points.shape[1])])
    variance = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
    grid = points.reshape(len(states), keep, -1)
    return {"encoder": encoder, "token_total": token_total, "hidden_size": int(width), "numbered": numbered,
            "tokens": [{"text": str(labels[at])[:48], "id": ids[at], "weight": weights[at], "special": i not in words} for i, at in enumerate(positions)],
            "layers": layers, "map": {"method": "pca", "explained": [round(float(v), 4) for v in variance[:2]],
                                      "points": [[[round(c, 3) for c in point] for point in layer] for layer in grid.tolist()]}}


class ModelRunTextEncode:
    CATEGORY = "Model Run"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    DESCRIPTION = "Works like CLIP Text Encode and records the text encoder's layers. Open Model Run to see tokens, activity and units."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def encode(self, clip, text):
        import torch
        if clip is None:
            raise RuntimeError("clip input is invalid: None")
        tokens = clip.tokenize(text)
        stack = clip.cond_stage_model
        try:
            groups = encoder_groups(stack)
            paths = _thin([path for group in groups.values() for path in group])
        except Exception as exc:  # noqa: BLE001 - this node must still encode
            logging.getLogger(__name__).warning("Model Run could not read this text encoder: %s", exc)
            return (clip.encode_from_tokens_scheduled(tokens),)
        if not paths:
            # Some models keep their text layers in the denoiser, as HiDream-O1 does.
            return (clip.encode_from_tokens_scheduled(tokens),)
        states, handles, names = {}, [], dict(stack.named_modules())
        def hook_for(path):
            def hook(_module, _args, output):
                value = output[0] if isinstance(output, (tuple, list)) and output else output
                if path not in states and isinstance(value, torch.Tensor) and value.ndim == 3:
                    states[path] = value[0].detach().float().cpu()
            return hook
        for path in paths:
            handles.append(names[path].register_forward_hook(hook_for(path)))
        result = {}
        def call():
            result["started"] = True
            result["value"] = clip.encode_from_tokens_scheduled(tokens)
        invocation = _Run.module("encoder", stack, call)
        try:
            _, trace = capture_forward(invocation, {}, [f"encoder.{p}" for p in paths], max_tensor_elements=MAX_ELEMENTS, max_events=len(paths),
                                       slices=SliceRecorder(max_values=SLICE_VALUES, per_event=SLICE_PER_LAYER), require_eval=False)
        except Exception as exc:
            if result.get("started") and "value" not in result:
                raise  # the encoder itself failed, as it would have without this node
            logging.getLogger(__name__).warning("Model Run could not record this text encoder: %s", exc)
            return (result["value"] if "value" in result else clip.encode_from_tokens_scheduled(tokens),)
        finally:
            for handle in handles:
                handle.remove()
        try:
            for event in trace["events"]:
                event["module"] = event["module"][len("encoder."):]
            sections = []
            for encoder, group in groups.items():
                section = encoder_section(encoder, {p: states[p] for p in group if p in states}, encoder_tokens(clip, tokens, encoder))
                if section:
                    sections.append(section)
            trace.update(model_class=f"{type(stack).__module__}.{type(stack).__qualname__}", source="comfyui_native_text_encoder", structure=inspect_module_tree(stack))
            trace["capture"]["modules"] = paths
            trace["text_encoder"] = {"encoders": sections, "prompt_characters": len(text), "limitations": list(ENCODER_LIMITATIONS)}
            _publish(type(stack).__name__, trace, text if isinstance(text, str) else None)
        except Exception as exc:  # noqa: BLE001 - the conditioning is what the workflow came for
            logging.getLogger(__name__).warning("Model Run could not record this text encoder: %s", exc)
        return (result["value"],)


def _vae_run(vae, attribute, call, note, *, latent_shape=None, images=None):
    from .comfy_bridge import encode_images
    model = vae.first_stage_model
    inner = getattr(model, attribute, None)
    prefix = f"{attribute}." if inner is not None else ""
    paths = _thin([f"{prefix}{p}" for p in stage_paths(inner if inner is not None else model)])
    result = {}
    def run():
        result["started"] = True
        result["value"] = call()
    if not paths:
        return call()
    invocation = _Run.module("vae", model, run)
    try:
        _, trace = capture_forward(invocation, {}, [f"vae.{p}" for p in paths], max_tensor_elements=MAX_ELEMENTS, max_events=len(paths),
                                   slices=SliceRecorder(max_values=SLICE_VALUES, per_event=SLICE_PER_LAYER, convolutional=True), latent_shape=latent_shape, require_eval=False)
    except Exception as exc:
        if result.get("started") and "value" not in result:
            raise  # the VAE itself failed, as it would have without this node
        logging.getLogger(__name__).warning("Model Run could not record this VAE: %s", exc)
        return result["value"] if "value" in result else call()
    try:
        for event in trace["events"]:
            event["module"] = event["module"][len("vae."):]
        trace.update(model_class=f"{type(model).__module__}.{type(model).__qualname__}", source="comfyui_native_vae", structure=inspect_module_tree(model))
        trace["capture"]["modules"] = paths
        trace["limitations"].extend(VAE_LIMITATIONS)
        pictures = result["value"] if images is None else images
        if hasattr(pictures, "ndim") and pictures.ndim == 4 and pictures.shape[-1] in (1, 3, 4):
            preview, frames = encode_images(pictures)
            trace["preview"] = preview
            if frames:
                trace["preview_frames"] = frames
        _publish(type(model).__name__, trace, note)
    except Exception as exc:  # noqa: BLE001 - the images are what the workflow came for
        logging.getLogger(__name__).warning("Model Run could not record this VAE: %s", exc)
    return result["value"]


class ModelRunVAEDecode:
    CATEGORY = "Model Run"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    DESCRIPTION = "Works like VAE Decode and records what each VAE block produced. Open Model Run to see the feature maps."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def decode(self, vae, samples):
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        images = _vae_run(vae, "decoder", lambda: vae.decode(latent), "VAE decode", latent_shape=list(latent.shape))
        if images.ndim == 5:  # same as ComfyUI's own node: a video batch becomes frames
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)


class ModelRunVAEEncode:
    CATEGORY = "Model Run"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    DESCRIPTION = "Works like VAE Encode and records what each VAE block produced. Open Model Run to see the feature maps."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pixels": ("IMAGE",), "vae": ("VAE",)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def encode(self, vae, pixels):
        shape = [int(pixels.shape[0]), int(pixels.shape[-1]), int(pixels.shape[1]), int(pixels.shape[2])]
        latent = _vae_run(vae, "encoder", lambda: vae.encode(pixels), "VAE encode", latent_shape=shape, images=pixels)
        return ({"samples": latent},)
