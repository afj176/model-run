"""Measured image-to-text attention maps for diffusion denoisers.

The probabilities are computed from the exact query and key tensors the attention kernel
receives (after any normalisation and rotary embedding), with the same scale and mask, so
they are the attention the model applied, averaged over heads and area-pooled to a small
grid. Nothing is inferred from architecture names; unsupported geometries are reported.
"""
import math

MAX_MAP_VALUES = 262144
MAX_TEXT_TOKENS = 1024
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


def measure(q, k, *, heads, text_tokens, latent_shape, scale=None, mask=None, batch_index=0, max_grid=32, text_first=True):
    """Return a bounded map record, or ``{"unavailable": reason}`` when the geometry is not understood.

    ``q`` and ``k`` are either ``[B, heads, L, D]`` or ``[B, L, heads*D]``. Text keys occupy
    the first ``text_tokens`` key positions. Image queries are the whole query sequence
    (classic cross-attention, keys are text only) or the positions after the text block
    (joint attention over the concatenated text+image sequence).
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
    if key_count == text_tokens and query_count != text_tokens:
        kind, first_image_query, text_start = "cross", 0, 0
    elif key_count == query_count and key_count > text_tokens:
        kind = "joint"
        first_image_query = text_tokens if text_first else 0
        text_start = 0 if text_first else key_count - text_tokens
    else:
        return {"unavailable": f"Sequence lengths (queries {query_count}, keys {key_count}, text {text_tokens}) do not form cross or joint attention"}
    image_tokens = query_count - text_tokens if kind == "joint" else query_count
    grid = image_grid(image_tokens, latent_shape)
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
            probabilities = torch.softmax(scores, dim=-1)[:, text_start:text_start + text_tokens]
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
    return {"attention_kind": kind, "heads": heads, "head_dim": dim, "text_tokens": text_tokens, "image_tokens": image_tokens,
            "source_grid": [rows, cols], "grid_factor": factor, "grid": [out_rows, out_cols], "scale": scale,
            "frames": kept, "frame_indices": frame_indices, "source_frames": frames, "text_first": text_first,
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


MASK_BIAS = -10000.0
ABLATION_LIMITATIONS = ["Ablated text tokens are masked out of every attention call that reads text keys for the whole run; remaining attention renormalises over the other keys.",
                        "Masking a token's keys is an intervention on attention only; the token still passed through the text encoder and may influence other tokens' embeddings.",
                        "Compare against a run with the same prompt, seed, steps and size and no ablation.",
                        "applied_attention_calls counts attention calls inside the recorded denoiser calls; the mask also applies to later, unrecorded calls."]


def parse_token_list(value, *, limit=64, bound=1024):
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


def text_key_range(query_count, key_count, text_tokens, allow_joint=False):
    """Where text keys sit in an attention call: (start, stop) or None when keys carry no text.

    Joint attention (text and image concatenated, keys == queries) is only recognised when the
    caller knows the block is a joint-attention block; a UNet self-attention has the same shape
    and no text keys.
    """
    if key_count == text_tokens and query_count != text_tokens:
        return 0, text_tokens
    if allow_joint and key_count == query_count and key_count > text_tokens:
        return 0, text_tokens
    return None


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
