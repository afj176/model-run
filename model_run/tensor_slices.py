"""Bounded per-frame and per-channel records of a tensor, so navigation shows measured values.

A flattened prefix of a ``[B, C, F, H, W]`` latent lies entirely in the first channel and frame.
This keeps real values for each slice instead: exact when the slice fits ``SIDE``, area-averaged
when it does not, and every record says which. Axis names appear only when the tensor's layout is
known from the denoiser's latent; otherwise axes are reported by position.
"""
from __future__ import annotations

import math

from . import capture_limits

MAX_SLICES = 64
MAX_FRAMES = 16
SIDE = 16
SEQUENCE_COLS = 64
MAX_SEQUENCE_BATCH = 8
TRACE_VALUES = 262144
# Token grids: every token of a text layer (up to this many) by as many of its features as fit.
# The Standard capture size; the active one comes from capture_limits (token_rows, token_values).
MAX_TOKEN_ROWS = 128
TOKEN_VALUES = 8388608  # per trace, across every layer
ALL_FEATURES = 1 << 20
# Image patches: a ViT's class token and every patch (a 32 x 32 grid and a few registers fit).
MAX_PATCH_ROWS = 1040
REFERENCE_ASSUMPTION = "Reference image tokens are assumed to follow the generated image and are not shown; the count matches the references this call received."
PATCH_ASSUMPTION = "Patches are assumed to follow the class token in row order. The class token itself is not shown here."
JOINT_ASSUMPTION = "This block carries text and image tokens together. Text is assumed to come first and is not shown; the count matches, and ComfyUI's Flux code uses this order."
PACKED_ASSUMPTION = "Tokens are assumed to be packed in frame, row, column order, the same assumption the attention maps use."


def _spaced(size, count):
    """``count`` evenly spaced indices of ``range(size)``, always including the first."""
    if count >= size:
        return list(range(size))
    return sorted({int(i * size / count) for i in range(count)})


def _plan_counts(sizes, names, max_slices=MAX_SLICES, max_frames=MAX_FRAMES):
    """How many indices each axis keeps. Frames are kept first; other axes share what is left."""
    counts = list(sizes)
    frame = names.index("frame") if "frame" in names else None
    if frame is not None:
        counts[frame] = min(counts[frame], max_frames)
    while math.prod(counts) > max_slices:
        others = [i for i in range(len(counts)) if i != frame and counts[i] > 1]
        pool = others or [i for i in range(len(counts)) if counts[i] > 1]
        widest = max(pool, key=lambda i: counts[i])
        over = math.prod(counts) / max_slices
        counts[widest] = max(1, min(counts[widest] - 1, math.floor(counts[widest] / over)))
    return counts


def _same_image_shape(shape, latent_shape):
    """Whether height and width are the latent's scaled by one whole factor."""
    height, width, latent_height, latent_width = int(shape[-2]), int(shape[-1]), int(latent_shape[-2]), int(latent_shape[-1])
    if not all((height, width, latent_height, latent_width)) or height * latent_width != width * latent_height:
        return False
    return latent_height % height == 0 or height % latent_height == 0


def _layout(tensor, latent_shape, token_grid=None, text_tokens=None, reference_tokens=0):
    """Return (view, names, layout, assumption), or None when no navigation applies."""
    shape = list(tensor.shape)
    if token_grid is not None and len(shape) == 3:
        prefix, rows, cols = token_grid
        if shape[1] == prefix + rows * cols:
            view = tensor[:, prefix:].reshape(shape[0], rows, cols, shape[2]).permute(0, 3, 1, 2)
            return view, ["batch", "feature", "row", "column"], "patch_tokens", PATCH_ASSUMPTION
    # Guidance doubles the batch, so the batch size is not part of the match.
    if latent_shape is not None and shape[1:] == list(latent_shape)[1:] and len(shape) in (4, 5):
        names = ["batch", "channel", "frame", "height", "width"] if len(shape) == 5 else ["batch", "channel", "height", "width"]
        return tensor, names, "latent", None
    if latent_shape is not None and len(shape) == 3:
        from .cross_attention import image_grid
        grid = image_grid(shape[1], latent_shape)
        if grid is not None and grid[0] * grid[1] > 1:
            rows, cols, _, frames = grid
            view = tensor.reshape(shape[0], frames, rows, cols, shape[2]).permute(0, 4, 1, 2, 3)
            return view, ["batch", "feature", "frame", "row", "column"], "packed_tokens", PACKED_ASSUMPTION
        if reference_tokens and shape[1] > reference_tokens:
            # An edit model's reference images follow the image being generated.
            grid = image_grid(shape[1] - reference_tokens, latent_shape)
            if grid is not None and grid[0] * grid[1] > 1:
                rows, cols, _, frames = grid
                view = tensor[:, :shape[1] - reference_tokens].reshape(shape[0], frames, rows, cols, shape[2]).permute(0, 4, 1, 2, 3)
                return view, ["batch", "feature", "frame", "row", "column"], "packed_tokens", REFERENCE_ASSUMPTION
        if text_tokens and shape[1] > text_tokens + reference_tokens:
            # Single-stream blocks carry text and image together, with text first.
            grid = image_grid(shape[1] - text_tokens - reference_tokens, latent_shape)
            if grid is not None and grid[0] * grid[1] > 1:
                rows, cols, _, frames = grid
                view = tensor[:, text_tokens:shape[1] - reference_tokens].reshape(shape[0], frames, rows, cols, shape[2]).permute(0, 4, 1, 2, 3)
                return view, ["batch", "feature", "frame", "row", "column"], "packed_tokens", JOINT_ASSUMPTION
    if latent_shape is not None and len(shape) == len(latent_shape) and len(shape) in (4, 5) and _same_image_shape(shape, latent_shape):
        # A convolutional block: the latent's layout at another resolution, such as inside a UNet.
        names = ["batch", "channel", "frame", "height", "width"] if len(shape) == 5 else ["batch", "channel", "height", "width"]
        return tensor, names, "features", None
    if len(shape) >= 4:
        return tensor, [None] * len(shape), "axes", None
    return None


def _clean(values):
    return [float(f"{v:.5g}") if math.isfinite(v) else None for v in values]


class SliceRecorder:
    """Records slices for one trace, spending at most ``max_values`` numbers across all events."""

    def __init__(self, max_values=TRACE_VALUES, token_grid=None, sequence=None, per_event=None, convolutional=False, token_features=None,
                 parts=(), part_features=None, patch_features=None):
        # Set when every tensor is known to be a convolutional block, such as inside a VAE.
        self.convolutional = convolutional
        self.remaining = max_values
        self.token_grid = token_grid
        self.text_tokens = None
        self.reference_tokens = 0
        # With guidance a batch holds both prompts, and the first entry is usually the negative one.
        self.conditional_batch = None
        # When many layers are recorded, each event's share, so the last layer gets as much as the first.
        self.per_event = per_event
        # sequence(length) -> (kept steps, axis name, note) when a [1, T, D] tensor runs along a known axis.
        self.sequence = sequence
        # Text layers: how many features of every token to keep (None leaves text layers to the prefix).
        self.token_features = token_features
        # A layer's attention and MLP outputs keep fewer features; their per-token RMS stays exact.
        self.parts, self.part_features = set(parts), part_features
        self._path = None
        # Image patch layers also keep every patch by its features, from a budget of their own, so the
        # patch maps they already record are unchanged.
        self.patch_features = patch_features
        self.patch_remaining = capture_limits.get("token_values")
        self.token_rows = capture_limits.get("token_rows")
        # What text layers held and what their grids kept, for Run Metrics.
        self.token_stats = {"layers": 0, "length": 0, "rows": 0, "depth": 0, "cols": None, "short": 0}

    def _placed(self, tensor, latent_shape):
        # Some VAEs keep channels last in places; only name the axes when the shape really is the image's.
        if self.convolutional and tensor.ndim in (4, 5) and (latent_shape is None or _same_image_shape(list(tensor.shape), latent_shape)):
            names = ["batch", "channel", "frame", "height", "width"] if tensor.ndim == 5 else ["batch", "channel", "height", "width"]
            return tensor, names, "features", None
        return _layout(tensor, latent_shape, self.token_grid, self.text_tokens, self.reference_tokens)

    def record(self, tensor, latent_shape=None, path=None):
        """A slice record; None when the tensor has nothing to navigate; ``unavailable`` on failure."""
        self._path = path
        try:
            return self._record(tensor, latent_shape)
        except Exception as exc:  # Observation must never break the model being observed.
            return {"unavailable": f"Slices could not be recorded: {type(exc).__name__}"}

    def shows_image(self, tensor, latent_shape):
        """Whether this tensor is the image stream, so a block returning several can be read by its image."""
        placed = self._placed(tensor, latent_shape)
        return placed is not None and placed[2] != "axes"

    def _sequence_record(self, tensor, kept, name, note):
        """A feature-by-time map of a [B, T, D] tensor per batch entry, so all of its length can be seen."""
        import torch
        batch, depth = int(tensor.shape[0]), int(tensor.shape[2])
        rows, cols = min(depth, SIDE), min(kept, SEQUENCE_COLS)
        indices = _spaced(batch, MAX_SEQUENCE_BATCH)
        if len(indices) * rows * cols > self.remaining:
            return {"unavailable": "This trace's slice budget was already spent on earlier events."}
        self.remaining -= len(indices) * rows * cols
        chosen = tensor.detach()[indices, :kept].to(dtype=torch.float32)
        view = chosen.transpose(1, 2).unsqueeze(1)
        pooled = (rows, cols) != (depth, kept)
        if pooled:
            view = torch.nn.functional.adaptive_avg_pool2d(view, (rows, cols))
        axes = []
        if batch > 1:
            magnitude = chosen.abs().to(device="cpu")
            axes = [{"dim": 0, "name": "batch", "size": batch, "indices": indices,
                     "mean_abs": _clean(magnitude.mean(dim=(1, 2)).tolist()), "max_abs": _clean(magnitude.amax(dim=(1, 2)).tolist())}]
            if self.conditional_batch in indices:
                axes[0]["conditional"] = self.conditional_batch
        return {"layout": "sequence", "source_shape": list(tensor.shape), "axes": axes, "row_axis": "feature", "col_axis": name,
                "rows": rows, "cols": cols, "source_rows": depth, "source_cols": kept, "pooled": pooled, "assumption": note,
                "values": _clean(view.reshape(-1).to(device="cpu").tolist()), "aggregate_axis": None,
                "aggregate_measure": None, "aggregate": []}

    def _token_record(self, tensor, max_rows=None, features=None, patch_grid=None):
        """Every token of a [B, T, D] (or [T, D]) text layer by the same evenly spaced features, exact, not averaged.

        Rows are the first tokens (up to the capture size's token rows): text comes first and padding last. Each row also
        keeps the token's RMS and peak over all of its features, so a token's strength is never a sample."""
        import torch
        state = tensor.detach()
        if state.ndim == 2:
            state = state.unsqueeze(0)
        batch, length, depth = (int(n) for n in state.shape)
        chosen_batch = self.conditional_batch if self.conditional_batch is not None and self.conditional_batch < batch else 0
        rows = min(length, max_rows or self.token_rows)
        pool = "patch_remaining" if patch_grid is not None else "remaining"
        budget = getattr(self, pool) if self.per_event is None else min(getattr(self, pool), self.per_event)
        limit = features or (self.part_features if self.part_features and self._path in self.parts else self.token_features)
        cols = min(depth, limit, budget // rows)
        # A layer's own grid; its attention and MLP parts keep fewer features by design.
        stats = self.token_stats if patch_grid is None and self._path not in self.parts else None
        if cols < 1:
            if stats is not None:
                stats["short"] += 1
            return {"unavailable": "This trace's value budget was already spent on earlier layers."}
        features = _spaced(depth, cols)
        cols = len(features)
        if stats is not None:
            stats.update(layers=stats["layers"] + 1, length=max(stats["length"], length), rows=max(stats["rows"], rows),
                         depth=max(stats["depth"], depth), cols=cols if stats["cols"] is None else min(stats["cols"], cols))
        setattr(self, pool, getattr(self, pool) - rows * cols)
        tokens = state[chosen_batch, :rows].to(dtype=torch.float32)
        values = tokens[:, torch.tensor(features, device=tokens.device)]
        magnitude = tokens.abs()
        return {"layout": "tokens", "source_shape": list(tensor.shape), "axes": [], "row_axis": "token", "col_axis": "feature",
                "rows": rows, "cols": cols, "source_rows": length, "source_cols": depth, "pooled": False, "assumption": None,
                "batch_index": chosen_batch, "token_indices": list(range(rows)), "feature_indices": features,
                "values": _clean(values.reshape(-1).to(device="cpu").tolist()),
                "token_rms": _clean(tokens.pow(2).mean(dim=-1).sqrt().to(device="cpu").tolist()),
                "token_max_abs": _clean(magnitude.amax(dim=-1).to(device="cpu").tolist()),
                "aggregate_axis": None, "aggregate_measure": None, "aggregate": [],
                **({"patch_grid": list(patch_grid), "row_labels": _patch_labels(rows, patch_grid)} if patch_grid else {})}

    def _record(self, tensor, latent_shape):
        import torch
        grid = self._patch_grid(tensor)
        # A part of a patch layer (its attention, say): every patch, few features, each patch's RMS exact.
        if grid is not None and self.part_features and self._path in self.parts:
            return self._token_record(tensor, MAX_PATCH_ROWS, self.part_features, grid)
        if grid is not None and self.patch_features:
            record = self._record_placed(tensor, latent_shape)
            if record is not None and "unavailable" not in record:
                record["patches"] = self._token_record(tensor, MAX_PATCH_ROWS, self.patch_features, grid)
            return record
        return self._record_placed(tensor, latent_shape)

    def _patch_grid(self, tensor):
        """(prefix tokens, rows, cols) when this [B, T, D] tensor is a class token or two followed by image patches."""
        if self.token_grid is None or tensor.ndim != 3:
            return None
        prefix, rows, cols = self.token_grid
        return self.token_grid if int(tensor.shape[1]) == prefix + rows * cols and int(tensor.shape[1]) <= MAX_PATCH_ROWS else None

    def _record_placed(self, tensor, latent_shape):
        import torch
        if self.token_features and (tensor.ndim == 2 or (tensor.ndim == 3 and self.sequence is None and self._placed(tensor, latent_shape) is None)):
            return self._token_record(tensor)
        if self.sequence is not None and tensor.ndim == 3:
            plan = self.sequence(int(tensor.shape[1]))
            if plan is not None:
                return self._sequence_record(tensor, *plan)
        placed = self._placed(tensor, latent_shape)
        if placed is None:
            return None
        view, names, layout, assumption = placed
        leading = list(view.shape[:-2])
        height, width = int(view.shape[-2]), int(view.shape[-1])
        navigable = [dim for dim, size in enumerate(leading) if size > 1]
        if not navigable or height < 1 or width < 1:
            return None
        rows, cols = min(height, SIDE), min(width, SIDE)
        axis_names = [names[dim] for dim in navigable]
        limits = {}
        if self.per_event is not None:
            frames = leading[navigable[axis_names.index("frame")]] if "frame" in axis_names else 1
            if frames > max(1, self.per_event // (rows * cols)) // 2:
                # Many frames: a coarser grid keeps more of them.
                rows, cols = min(rows, SIDE // 2), min(cols, SIDE // 2)
            maps = max(2, self.per_event // (rows * cols))
            kept_frames = min(frames, MAX_FRAMES, max(1, maps // 2))
            limits = {"max_frames": kept_frames, "max_slices": min(MAX_SLICES, max(kept_frames, kept_frames * (maps // kept_frames - 1)))}
        counts = _plan_counts([leading[dim] for dim in navigable], axis_names, **limits)
        if self.per_event is not None:
            # The all-channels maps count too, so a late layer is not left with nothing.
            aggregate_at = next((i for i, name in enumerate(axis_names) if name in ("channel", "feature")), 0)
            while (math.prod(counts) + math.prod(counts) // counts[aggregate_at]) * rows * cols > self.per_event and counts[aggregate_at] > 1:
                counts[aggregate_at] -= 1
        kept = [_spaced(leading[dim], count) for dim, count in zip(navigable, counts)]
        aggregate = next((i for i, name in enumerate(axis_names) if name in ("channel", "feature")), 0)
        slice_count = math.prod(len(indices) for indices in kept)
        map_count = slice_count // len(kept[aggregate])
        cost = (slice_count + map_count) * rows * cols
        if cost > self.remaining:
            return {"unavailable": "This trace's slice budget was already spent on earlier events."}
        self.remaining -= cost

        view = view.detach().to(dtype=torch.float32)
        magnitude = view.abs()

        def pooled(selected):
            flat = selected.reshape(-1, 1, height, width)
            if (rows, cols) != (height, width):
                flat = torch.nn.functional.adaptive_avg_pool2d(flat, (rows, cols))
            return _clean(flat.reshape(-1).to(device="cpu").tolist())

        def select(selected, skip=None):
            for position, dim in enumerate(navigable):
                if position != skip:
                    index = torch.tensor(kept[position], device=selected.device)
                    selected = selected.index_select(dim, index)
            return selected

        axes = []
        for position, dim in enumerate(navigable):
            others = [d for d in range(view.ndim) if d != dim]
            index = torch.tensor(kept[position], device=view.device)
            means = magnitude.mean(dim=others).index_select(0, index)
            peaks = magnitude.amax(dim=others).index_select(0, index)
            axes.append({"dim": dim, "name": names[dim], "size": leading[dim], "indices": kept[position],
                         "mean_abs": _clean(means.to(device="cpu").tolist()), "max_abs": _clean(peaks.to(device="cpu").tolist())})
            if names[dim] == "batch" and self.conditional_batch in kept[position]:
                axes[-1]["conditional"] = self.conditional_batch
        return {"layout": layout, "source_shape": list(view.shape), "axes": axes,
                "row_axis": names[-2], "col_axis": names[-1],
                "rows": rows, "cols": cols, "source_rows": height, "source_cols": width,
                "pooled": (rows, cols) != (height, width), "assumption": assumption,
                "values": pooled(select(view)), "aggregate_axis": aggregate,
                "aggregate_measure": "mean_abs",
                "aggregate": pooled(select(magnitude.mean(dim=navigable[aggregate], keepdim=True), skip=aggregate))}


def validate_slices(record):
    """Raise ValueError unless ``record`` is a consistent slice record."""
    if not isinstance(record, dict):
        raise ValueError("Invalid slice record")
    if "unavailable" in record:
        if not isinstance(record["unavailable"], str) or len(record) != 1:
            raise ValueError("Invalid slice record")
        return
    if "patches" in record:
        # A patch layer's every-patch grid, beside its patch maps.
        patches = record["patches"]
        if not isinstance(patches, dict):
            raise ValueError("Invalid patch grid")
        if "unavailable" in patches:
            validate_slices(patches)
        elif patches.get("layout") != "tokens" or "patch_grid" not in patches:
            raise ValueError("Invalid patch grid")
        else:
            validate_slices(patches)
    for field in ("row_axis", "col_axis", "assumption"):
        if record.get(field) is not None and not isinstance(record[field], str):
            raise ValueError("Invalid slice description")
    for field in ("source_rows", "source_cols"):
        if type(record.get(field)) is not int or record[field] < 1:
            raise ValueError("Invalid slice source size")
    if type(record.get("pooled")) is not bool:
        raise ValueError("Invalid slice description")
    axes, rows, cols = record.get("axes"), record.get("rows"), record.get("cols")
    if record.get("layout") == "tokens":
        _validate_tokens(record)
        return
    if record.get("layout") == "sequence":
        values = record.get("values")
        if not isinstance(axes, list) or len(axes) > 1:
            raise ValueError("Invalid sequence record")
        batches = 1
        for axis in axes:
            indices = axis.get("indices") if isinstance(axis, dict) else None
            if (not isinstance(indices, list) or not 1 <= len(indices) <= MAX_SEQUENCE_BATCH or type(axis.get("size")) is not int
                    or any(type(i) is not int or not 0 <= i < axis["size"] for i in indices)
                    or any(not isinstance(axis.get(field), list) or len(axis[field]) != len(indices) for field in ("mean_abs", "max_abs"))):
                raise ValueError("Invalid sequence axis")
            batches = len(indices)
        if (type(rows) is not int or type(cols) is not int or not 1 <= rows <= SIDE or not 1 <= cols <= SEQUENCE_COLS
                or not isinstance(values, list) or len(values) != batches * rows * cols or record.get("aggregate") != []):
            raise ValueError("Invalid sequence record")
        for value in values:
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("Invalid slice value")
        return
    if not isinstance(axes, list) or not 1 <= len(axes) <= 16 or type(rows) is not int or type(cols) is not int or not 1 <= rows <= SIDE or not 1 <= cols <= SIDE:
        raise ValueError("Invalid slice record")
    for axis in axes:
        indices = axis.get("indices") if isinstance(axis, dict) else None
        if (not isinstance(indices, list) or not indices or len(indices) > MAX_SLICES or type(axis.get("size")) is not int
                or any(type(i) is not int or not 0 <= i < axis["size"] for i in indices)
                or (axis.get("name") is not None and not isinstance(axis["name"], str))):
            raise ValueError("Invalid slice axis")
        for field in ("mean_abs", "max_abs"):
            if not isinstance(axis.get(field), list) or len(axis[field]) != len(indices):
                raise ValueError("Invalid slice axis statistics")
    for axis in axes:
        for field in ("mean_abs", "max_abs"):
            if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in axis[field]):
                raise ValueError("Invalid slice axis statistics")
    aggregate = record.get("aggregate_axis")
    if type(aggregate) is not int or not 0 <= aggregate < len(axes):
        raise ValueError("Invalid slice aggregate axis")
    slices = math.prod(len(axis["indices"]) for axis in axes)
    if slices > MAX_SLICES:
        raise ValueError("Too many slices")
    values, summary = record.get("values"), record.get("aggregate")
    if not isinstance(values, list) or len(values) != slices * rows * cols:
        raise ValueError("Slice values do not match their axes")
    if not isinstance(summary, list) or len(summary) != slices // len(axes[aggregate]["indices"]) * rows * cols:
        raise ValueError("Slice aggregate does not match its axes")
    for value in (*values, *summary):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("Invalid slice value")


def _validate_tokens(record):
    """A token grid: every listed token by the same listed features, with each token's RMS and peak."""
    rows, cols, values = record.get("rows"), record.get("cols"), record.get("values")
    tokens, features = record.get("token_indices"), record.get("feature_indices")
    length, depth = record["source_rows"], record["source_cols"]
    grid = record.get("patch_grid")
    if grid is not None and (not isinstance(grid, list) or len(grid) != 3 or any(type(n) is not int or n < 0 for n in grid)
                             or grid[1] < 1 or grid[2] < 1 or grid[0] + grid[1] * grid[2] != length):
        raise ValueError("Invalid patch grid")
    if (record.get("axes") != [] or record.get("aggregate") != [] or type(rows) is not int or type(cols) is not int
            or not 1 <= rows <= (MAX_PATCH_ROWS if grid else capture_limits.CEILING["token_rows"]) or not 1 <= cols <= depth
            or rows * cols > capture_limits.CEILING["token_values"]
            or type(record.get("batch_index")) is not int or record["batch_index"] < 0):
        raise ValueError("Invalid token grid")
    if (not isinstance(tokens, list) or len(tokens) != rows or any(type(i) is not int or not 0 <= i < length for i in tokens)
            or not isinstance(features, list) or len(features) != cols or any(type(i) is not int or not 0 <= i < depth for i in features)
            or features != sorted(set(features))):
        raise ValueError("Invalid token grid indices")
    if not isinstance(values, list) or len(values) != rows * cols:
        raise ValueError("Token grid values do not match its size")
    for field in ("token_rms", "token_max_abs"):
        if not isinstance(record.get(field), list) or len(record[field]) != rows:
            raise ValueError("Invalid token grid statistics")
    labels = record.get("row_labels")
    if labels is not None and (not isinstance(labels, list) or len(labels) != rows or any(not isinstance(x, str) or len(x) > 64 for x in labels)):
        raise ValueError("Invalid token labels")
    for value in (*values, *record["token_rms"], *record["token_max_abs"]):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("Invalid slice value")


def _patch_labels(rows, patch_grid):
    """"CLS" for a class token (or "token N" for others), then "r3 c7" for each patch."""
    prefix, _height, width = patch_grid
    labels = []
    for index in range(rows):
        if index < prefix:
            labels.append("CLS" if index == 0 else f"token {index}")
        else:
            row, col = divmod(index - prefix, width)
            labels.append(f"r{row} c{col}")
    return labels
