"""Bounded per-frame and per-channel records of a tensor, so navigation shows measured values.

A flattened prefix of a ``[B, C, F, H, W]`` latent lies entirely in the first channel and frame.
This keeps real values for each slice instead: exact when the slice fits ``SIDE``, area-averaged
when it does not, and every record says which. Axis names appear only when the tensor's layout is
known from the denoiser's latent; otherwise axes are reported by position.
"""
from __future__ import annotations

import math

MAX_SLICES = 64
MAX_FRAMES = 16
SIDE = 16
TRACE_VALUES = 262144
PACKED_ASSUMPTION = "Tokens are assumed to be packed in frame, row, column order, the same assumption the attention maps use."


def _spaced(size, count):
    """``count`` evenly spaced indices of ``range(size)``, always including the first."""
    if count >= size:
        return list(range(size))
    return sorted({int(i * size / count) for i in range(count)})


def _plan_counts(sizes, names):
    """How many indices each axis keeps. Frames are kept first; other axes share what is left."""
    counts = list(sizes)
    frame = names.index("frame") if "frame" in names else None
    if frame is not None:
        counts[frame] = min(counts[frame], MAX_FRAMES)
    while math.prod(counts) > MAX_SLICES:
        others = [i for i in range(len(counts)) if i != frame and counts[i] > 1]
        pool = others or [i for i in range(len(counts)) if counts[i] > 1]
        widest = max(pool, key=lambda i: counts[i])
        over = math.prod(counts) / MAX_SLICES
        counts[widest] = max(1, min(counts[widest] - 1, math.floor(counts[widest] / over)))
    return counts


def _layout(tensor, latent_shape):
    """Return (view, names, layout, assumption), or None when no navigation applies."""
    shape = list(tensor.shape)
    if latent_shape is not None and shape == list(latent_shape) and len(shape) in (4, 5):
        names = ["batch", "channel", "frame", "height", "width"] if len(shape) == 5 else ["batch", "channel", "height", "width"]
        return tensor, names, "latent", None
    if latent_shape is not None and len(shape) == 3:
        from .cross_attention import image_grid
        grid = image_grid(shape[1], latent_shape)
        if grid is not None and grid[0] * grid[1] > 1:
            rows, cols, _, frames = grid
            view = tensor.reshape(shape[0], frames, rows, cols, shape[2]).permute(0, 4, 1, 2, 3)
            return view, ["batch", "feature", "frame", "row", "column"], "packed_tokens", PACKED_ASSUMPTION
    if len(shape) >= 4:
        return tensor, [None] * len(shape), "axes", None
    return None


def _clean(values):
    return [float(f"{v:.5g}") if math.isfinite(v) else None for v in values]


class SliceRecorder:
    """Records slices for one trace, spending at most ``max_values`` numbers across all events."""

    def __init__(self, max_values=TRACE_VALUES):
        self.remaining = max_values

    def record(self, tensor, latent_shape=None):
        """A slice record; None when the tensor has nothing to navigate; ``unavailable`` on failure."""
        try:
            return self._record(tensor, latent_shape)
        except Exception as exc:  # Observation must never break the model being observed.
            return {"unavailable": f"Slices could not be recorded: {type(exc).__name__}"}

    def _record(self, tensor, latent_shape):
        import torch
        placed = _layout(tensor, latent_shape)
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
        counts = _plan_counts([leading[dim] for dim in navigable], axis_names)
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
    axes, rows, cols = record.get("axes"), record.get("rows"), record.get("cols")
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
