"""Measured image-to-text attention maps for diffusion denoisers.

The probabilities are computed from the exact query and key tensors the attention kernel
receives (after any normalisation and rotary embedding), with the same scale and mask, so
they are the attention the model applied, averaged over heads and area-pooled to a small
grid. Nothing is inferred from architecture names; unsupported geometries are reported.
"""
import math
import re

MAX_MAP_VALUES = 262144
MAX_TEXT_TOKENS = 4096
MAX_GRID = 64
ROUND_DIGITS = 5
LIMITATIONS = ["Maps are head-averaged attention probabilities from the exact query/key tensors given to the attention kernel; per-head and per-layer differences are not shown.",
               "Image positions are area-averaged to the recorded grid; the source grid is the denoiser's token grid, not pixels.",
               "Attention weights show where text is read, not what caused a pixel; they are not a causal attribution."]


MAX_FRAMES = 16


def image_grid(token_count, latent_shape):
    """Infer (rows, cols, factor, frames) of the token grid from a 4-D image or 5-D video latent shape."""
    if not isinstance(latent_shape, (list, tuple)) or len(latent_shape) not in (4, 5):
        return None
    height, width = int(latent_shape[-2]), int(latent_shape[-1])
    frames_in = int(latent_shape[2]) if len(latent_shape) == 5 else 1
    temporal = (1, 2, 4) if len(latent_shape) == 5 else (1,)
    for ft in temporal:
        frames = -(-frames_in // ft)
        for factor in (1, 2, 4, 8, 16, 32):
            rows, cols = -(-height // factor), -(-width // factor)
            if frames * rows * cols == token_count:
                return rows, cols, factor, frames
    return None


def pooled_grid(rows, cols, text_tokens, max_grid):
    """Largest grid of at most ``max_grid`` per side whose maps fit the value bound."""
    scale = max(1, math.ceil(max(rows, cols) / max_grid))
    while True:
        out_rows, out_cols = max(1, math.ceil(rows / scale)), max(1, math.ceil(cols / scale))
        if text_tokens * out_rows * out_cols <= MAX_MAP_VALUES or (out_rows == 1 and out_cols == 1):
            return out_rows, out_cols
        scale += 1


def measure(q, k, *, heads, text_tokens, latent_shape, scale=None, mask=None, batch_index=0, max_grid=32, stream=None, text_start=0, reference_tokens=0,
            image_first=False, known_image_tokens=None, query_lead=0, image_at=None):
    """Return a bounded map record, or ``{"unavailable": reason}`` when the geometry is not understood.

    ``q`` and ``k`` are either ``[B, heads, L, D]`` or ``[B, L, heads*D]``. Text keys occupy
    the first ``text_tokens`` key positions unless ``text_start`` says otherwise: a count of tokens
    before the text, or -1 when the text comes last, because some models place it after the image. Image queries are the whole query sequence
    (classic cross-attention, keys are text only) or the positions after the text block
    (joint attention over the concatenated text+image sequence).

    ``image_first`` is for keys that run image, text, then other conditioning, so the text is neither
    first nor last. ``known_image_tokens`` says how many tokens are the image when the sequence's length
    cannot, and ``query_lead`` counts queries before the image that are not part of it. ``image_at``
    is where the image starts in joint attention when padding follows the text.
    """
    import torch
    if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor) or q.ndim not in (3, 4) or k.ndim != q.ndim:
        return {"unavailable": "Attention inputs are not tensors of a known layout"}
    if type(heads) is not int or heads < 1 or type(text_tokens) is not int or not 1 <= text_tokens <= MAX_TEXT_TOKENS:
        return {"unavailable": "Head count or text token count is out of range"}
    if type(max_grid) is not int or not 1 <= max_grid <= MAX_GRID:
        raise ValueError(f"max_grid must be between 1 and {MAX_GRID}")
    if q.ndim == 3:
        if q.shape[-1] % heads:
            return {"unavailable": "Query width is not divisible by the head count"}
        head_dim = q.shape[-1] // heads
        if k.shape[-1] % head_dim:
            return {"unavailable": "Key width is not a multiple of the head width"}
        q = q.reshape(q.shape[0], q.shape[1], heads, head_dim).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], k.shape[-1] // head_dim, head_dim).transpose(1, 2)
    if q.shape[1] != heads or q.shape[0] != k.shape[0] or not 0 <= batch_index < q.shape[0]:
        return {"unavailable": "Query/key head layout does not match the head count"}
    if k.shape[1] != heads:
        # Grouped-query attention: each key head serves several query heads.
        if k.shape[1] < 1 or heads % k.shape[1]:
            return {"unavailable": f"{k.shape[1]} key heads cannot serve {heads} query heads"}
        k = k.repeat_interleave(heads // k.shape[1], dim=1)
    if q.shape[-1] != k.shape[-1]:
        return {"unavailable": "Query and key head widths differ"}
    key_count, query_count, dim = int(k.shape[2]), int(q.shape[2]), int(q.shape[3])
    if query_lead and query_count <= query_lead:
        return {"unavailable": "These queries are the model's own summary tokens, not the image"}
    if query_lead and key_count == query_count or image_at == 0:
        return {"unavailable": "Self-attention in this block has no text keys"}
    if image_at is not None and image_at < 0:
        return {"unavailable": "These queries hold no image"}
    if stream == "audio" and known_image_tokens and query_count == key_count:
        return {"unavailable": "Self-attention in this block has no text keys"}
    if stream == "audio" and known_image_tokens:
        # An audio model's queries end with its time steps, and its keys can hold more than the prompt.
        kind, first_image_query = "cross", query_count - known_image_tokens
    elif known_image_tokens and not image_first and query_count == known_image_tokens and key_count not in (text_tokens, query_count):
        # Queries are the image alone and keys hold more than this prompt, as Hunyuan DiT's second encoder.
        kind, first_image_query = "cross", 0
    elif image_first and key_count >= (known_image_tokens or query_count) + text_tokens:
        kind, first_image_query = "joint", 0
    elif key_count == text_tokens and query_count != text_tokens:
        kind, first_image_query = "cross", query_lead
    elif key_count == query_count and key_count > text_tokens:
        kind, first_image_query = "joint", 0 if text_start == -1 else text_start + text_tokens
    elif key_count > text_tokens and query_count >= key_count - text_tokens:
        # Keys are [text, image]; queries end with the image, after any text tokens that attend this way too.
        # Qwen-Image's queries are image only; HiDream-O1's also start with the last text token.
        kind, first_image_query = "joint", query_count - (key_count - text_tokens)
    else:
        return {"unavailable": f"Sequence lengths (queries {query_count}, keys {key_count}, text {text_tokens}) do not form cross or joint attention"}
    image_tokens = query_count - first_image_query - (text_tokens if text_start == -1 and kind == "joint" and key_count == query_count else 0)
    first_text_key = key_count - text_tokens if text_start == -1 else text_start
    if image_at is not None:
        first_image_query = image_at
    if known_image_tokens:
        image_tokens = known_image_tokens
    if image_first and kind == "joint":
        image_tokens = known_image_tokens or query_count
        first_text_key = image_tokens
    # An edit model's reference images follow the generated image, and only the image gets a map.
    if reference_tokens and image_tokens > reference_tokens and stream != "audio" and image_grid(image_tokens, latent_shape) is None:
        image_tokens -= reference_tokens
    # One query is a pooled summary, not an image, unless the image really is one token.
    if image_tokens == 1 and max(int(latent_shape[-2]), int(latent_shape[-1])) > 1:
        return {"unavailable": "A single query token is a pooled summary, not the image"}
    # An audio stream runs along one axis, so its map is a strip over time.
    grid = (1, image_tokens, 1, 1) if stream == "audio" else image_grid(image_tokens, latent_shape)
    if grid is None:
        return {"unavailable": f"{image_tokens} image tokens do not map onto the latent grid {list(latent_shape)}"}
    rows, cols, factor, frames = grid
    scale = float(scale) if scale is not None else dim ** -0.5
    qb = q[batch_index, :, first_image_query:first_image_query + image_tokens].detach()
    kb = k[batch_index].detach()
    row_mask = None
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            return {"unavailable": "Attention mask is not a tensor"}
        m = mask
        if m.ndim in (3, 4):
            m = m[batch_index] if m.shape[0] > 1 else m[0]
        if m.ndim == 3 and m.shape[0] == heads:
            m = m[0]  # a mask repeated for each head
        while m.ndim > 1 and m.shape[0] == 1:
            m = m[0]
        if m.ndim not in (1, 2) or m.shape[-1] != key_count:
            return {"unavailable": "Attention mask layout is not understood"}
        if m.ndim == 2:
            if m.shape[0] != query_count:
                return {"unavailable": "Attention mask layout is not understood"}
            m = m[first_image_query:first_image_query + image_tokens]
        row_mask = m.to(device=qb.device)
    with torch.inference_mode():
        total = None
        for head in range(heads):
            scores = (qb[head].float() @ kb[head].float().transpose(0, 1)) * scale
            if row_mask is not None:
                if row_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~row_mask, float("-inf"))
                else:
                    scores = scores + row_mask.float()
            probabilities = torch.softmax(scores, dim=-1)[:, first_text_key:first_text_key + text_tokens]
            total = probabilities if total is None else total + probabilities
            del scores
        mean = (total / heads).cpu()  # [image_tokens, text_tokens]
        if not torch.isfinite(mean).all().item():
            return {"unavailable": "Attention probabilities were not finite"}
        maps = mean.transpose(0, 1).reshape(text_tokens, frames, rows, cols)
        frame_indices = list(range(frames))
        if frames > MAX_FRAMES:
            # Keep an even subset of frames so the record stays bounded.
            frame_indices = [round(i * (frames - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)]
            maps = maps[:, frame_indices]
        kept = len(frame_indices)
        out_rows, out_cols = pooled_grid(rows, cols, text_tokens * kept, max_grid)
        if (out_rows, out_cols) != (rows, cols):
            maps = torch.nn.functional.adaptive_avg_pool2d(maps.reshape(text_tokens * kept, 1, rows, cols), (out_rows, out_cols)).reshape(text_tokens, kept, out_rows, out_cols)
        maps = maps.reshape(text_tokens, kept * out_rows * out_cols)
        text_mass = maps.sum(0)
        token_mass = maps.mean(1)
    return {**({"stream": stream} if stream else {}), **({"text_spreads": True} if kind == "joint" and key_count != query_count and not image_first else {}),
            "attention_kind": kind, "heads": heads, "head_dim": dim, "text_tokens": text_tokens, "image_tokens": image_tokens,
            "source_grid": [rows, cols], "grid_factor": factor, "grid": [out_rows, out_cols], "scale": scale,
            "frames": kept, "frame_indices": frame_indices, "source_frames": frames,
            "mask_applied": row_mask is not None,
            "maps": [[round(v, ROUND_DIGITS) for v in row] for row in maps.tolist()],
            "text_mass": [round(v, ROUND_DIGITS) for v in text_mass.tolist()],
            "token_mass": [round(v, ROUND_DIGITS) for v in token_mass.tolist()]}


def block_record(selection, *, denoiser_call, batch_label, measurement, block_label):
    """Wrap one measurement as a trace event."""
    return {"denoiser_call": denoiser_call, "block": block_label, "selection": selection, "batch": batch_label, **measurement}


def new_section(selection, source, *, token_labels=None, max_grid=32):
    section = {"method": "measured_attention_probabilities", "source": source, "selection": selection,
               "grid_limit": max_grid, "events": [], "limitations": list(LIMITATIONS)}
    if token_labels is not None:
        section["text_labels"] = [str(label)[:48] for label in token_labels][:MAX_TEXT_TOKENS]
    return section


TEXT_MIXING_LIMITATION = ("In this model a prompt's meaning spreads across its text tokens, so other tokens can carry a word's meaning. "
                          "A low value on a word's map does not mean the word had no effect, and masking one word may not remove it.")
UNCHECKED_LABEL_LIMITATION = ("Word positions on this model family have not been checked against a real run. "
                              "They assume the prompt's tokens line up with the model's text keys.")
MASK_BIAS = -10000.0
ABLATION_LIMITATIONS = ["A small effect may only be sensitivity, because any change can shift the result. Compare it with masking an unrelated token.",
                        "Ablated text tokens are masked out of every attention call that reads text keys for the whole run; remaining attention renormalises over the other keys.",
                        "Masking a token's keys is an intervention on attention only; the token still passed through the text encoder and may influence other tokens' embeddings.",
                        "Compare against a run with the same prompt, seed, steps and size and no ablation.",
                        "applied_attention_calls counts attention calls inside the recorded denoiser calls; the mask also applies to later, unrecorded calls."]


def parse_token_list(value, *, limit=4096, bound=4096):
    """Distinct token indices from an int list or a comma/space separated string."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p for p in value.replace(",", " ").split() if p]
        try:
            value = [int(p) for p in parts]
        except ValueError as exc:
            raise ValueError("Token indices must be integers") from exc
    if not isinstance(value, list) or len(value) > limit or any(type(v) is not int or not 0 <= v < bound for v in value):
        raise ValueError(f"Ablate at most {limit} token indices below {bound}")
    return sorted(set(value))


def joint_by_geometry(query_count, key_count, text_tokens, latent_shape, extra=0):
    """Whether an unlabelled call must be joint attention: its count is the text plus the image, and alone fits no image."""
    if query_count != key_count or query_count <= text_tokens + extra:
        return False
    return image_grid(query_count - text_tokens - extra, latent_shape) is not None and image_grid(query_count, latent_shape) is None


def text_key_range(query_count, key_count, text_tokens, allow_joint=False, text_start=0):
    """Where text keys sit in an attention call: (start, stop) or None when keys carry no text.

    Joint attention (text and image concatenated, keys == queries) is only recognised when the
    caller knows the block is a joint-attention block; a UNet self-attention has the same shape
    and no text keys.
    """
    if key_count == text_tokens and query_count != text_tokens:
        return 0, text_tokens
    if allow_joint and key_count > text_tokens and query_count >= key_count - text_tokens:
        start = key_count - text_tokens if text_start == -1 else text_start
        return start, start + text_tokens
    return None


REGISTER_LABEL = "<register>"
REGISTER_LIMITATION = ("This model appends learned registers after the prompt's tokens. They receive most of the attention to text, but attention share "
                       "is not importance: in testing, masking the real tokens removed nearly all of the prompt's influence while masking the registers kept most of it.")
PADDING_LABEL = re.compile(r"^<\|?(?:pad|endoftext)\|?>$", re.IGNORECASE)


def collapse_padding(labels):
    """Some encoders pad with an ordinary token, as SDXL's does with '!', which must not read as a word."""
    end = next((i for i, label in enumerate(labels) if str(label).strip() in ("<|endoftext|>", "</s>", "<eos>")), None)
    if end is None or end + 1 >= len(labels) or len(set(labels[end + 1:])) != 1:
        return labels
    return [*labels[:end + 1], *[str(labels[end]).strip()] * (len(labels) - end - 1)]


def place_registers(labels, text_tokens):
    """Labels for a model that keeps only real tokens and fills the rest with learned registers.

    The tokenizer pads to a fixed length, but the encoder drops the padding and a connector appends
    registers after the real tokens, so a word's position is its place among the real tokens.
    """
    if not labels or type(text_tokens) is not int or text_tokens < 1:
        return None
    # Some tokenizers decode padding to nothing rather than to a pad token.
    real = [label for label in labels if str(label).strip() and not PADDING_LABEL.match(str(label).strip())]
    if not real or len(real) > text_tokens:
        return None
    return real + [REGISTER_LABEL] * (text_tokens - len(real))


def align_labels(labels, text_tokens):
    """Token labels for the text keys, or None when the keys cannot be matched to labels.

    Labels come from re-tokenising the prompt, so they only name the keys when the counts agree.
    Encoders that drop a leading system turn keep a suffix that starts at a role marker; that case
    is recognised. Anything else is refused rather than naming the wrong token.
    """
    if not labels or type(text_tokens) is not int or text_tokens < 1:
        return None
    if len(labels) == text_tokens:
        return list(labels)
    if len(labels) > text_tokens and labels[-text_tokens] == "<|im_start|>":
        return list(labels[-text_tokens:])
    return None


def stable_audio_labels(labels, text_tokens):
    """Stable Audio adds two number embeddings after the prompt's tokens, or None if the counts say otherwise."""
    if not labels or len(labels) + 2 != text_tokens:
        return None
    return [*labels, "<seconds start>", "<seconds total>"]


def key_mask(ablated, key_count, *, start=0, existing=None, device=None, dtype=None):
    """Additive attention mask of shape [1, 1, 1, key_count] that removes the ablated text keys.

    ``existing`` (bool or additive float, any broadcastable shape) is folded in so the model's own
    mask still applies. Returns None when nothing is ablated and no existing mask is given.
    """
    import torch
    positions = [start + t for t in ablated if 0 <= start + t < key_count]
    if not positions and existing is None:
        return None
    dtype = dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    bias = torch.zeros(1, 1, 1, key_count, device=device, dtype=dtype)
    if positions:
        bias[..., positions] = MASK_BIAS
    if existing is not None:
        current = existing
        if current.dtype == torch.bool:
            current = torch.zeros(current.shape, device=current.device, dtype=dtype).masked_fill(~current, MASK_BIAS)
        while current.ndim < 4:
            current = current.unsqueeze(0)
        bias = bias + current.to(device=bias.device, dtype=dtype)
    return bias
