"""How much a run keeps: one setting for every capture limit.

Standard holds the numbers Model Run was tested with. Large and Maximum keep more for computers with room to
spare, and Custom sets each limit. Every capture reads the active limits when it starts, and every stored trace
says what it kept against what the run held (``trace["kept"]``). The trace checker accepts anything up to the
Maximum preset, so a trace saved on a larger setting still opens on a smaller one.
"""
import math
import os
import sys
from contextlib import contextmanager
from threading import RLock

MiB = 1024 * 1024
GiB = 1024 * MiB

# key: (label, what it limits, standard, large, maximum)
FIELDS = {
    "prompt_tokens": ("Prompt tokens", "Longest prompt a text run accepts", 1024, 8192, 32768),
    "token_rows": ("Token rows per layer", "Tokens of each layer kept for the barcode and token views", 128, 512, 2048),
    "token_values": ("Token values per run", "Values every layer's token grids may hold together", 8388608, 33554432, 134217728),
    "attention_side": ("Attention side", "Tokens an attention matrix keeps per side; longer prompts are shrunk to blocks", 128, 512, 2048),
    "head_values": ("Per-head attention", "Values kept for every head of the probed layer", 262144, 4194304, 16777216),
    "stack_values": ("Every layer's heads", "Values kept for every head of every layer", 1048576, 16777216, 67108864),
    "response_tokens": ("Response tokens", "Longest answer a text model writes", 256, 1024, 4096),
    "capture_components": ("Picked components", "Components you can pick to capture by hand", 32, 128, 512),
    "layer_paths": ("ComfyUI layers", "Layers one ComfyUI run records", 65, 257, 1025),
    "layer_events": ("ComfyUI layer events", "Layer events one ComfyUI run records, over every denoiser call", 1000, 4000, 16000),
    "text_layers": ("Text encoder layers", "Text encoder layers one ComfyUI run records", 160, 640, 2560),
    "layer_calls": ("Denoiser calls with layers", "Denoiser calls whose layers a ComfyUI run records", 8, 32, 128),
    "attention_calls": ("Cross-attention calls", "Denoiser calls whose cross-attention is mapped", 4, 16, 64),
    "cross_attention_values": ("Cross-attention values", "Cross-attention map values per run", 524288, 4194304, 16777216),
    "denoising_steps": ("Denoising steps", "Denoising previews kept per run", 32, 128, 512),
    "trace_bytes": ("Size of one run", "Largest saved run, in bytes", 64 * MiB, 256 * MiB, 1 * GiB),
    "history_bytes": ("History size", "Memory all kept runs may use together, in bytes", 1 * GiB, 4 * GiB, 16 * GiB),
    "history_runs": ("Runs kept", "Runs kept in history", 16, 32, 64),
}
PROFILES = {name: {key: field[2 + i] for key, field in FIELDS.items()} for i, name in enumerate(("standard", "large", "maximum"))}
CEILING = PROFILES["maximum"]
FLOOR = {key: max(1, value // 4) for key, value in PROFILES["standard"].items()}
LABELS = {"standard": "Standard", "large": "Large", "maximum": "Maximum", "custom": "Custom"}

_lock = RLock()
_active = {"profile": "standard", "values": dict(PROFILES["standard"])}


def machine_memory():
    """Bytes of RAM and graphics memory, total and free; a value is None where it cannot be read."""
    found = {"ram_total": None, "ram_free": None, "gpu_total": None, "gpu_free": None, "gpu_name": None}
    try:
        import psutil
        memory = psutil.virtual_memory()
        found.update(ram_total=int(memory.total), ram_free=int(memory.available))
    except Exception:  # noqa: BLE001 - psutil is optional
        try:
            if sys.platform == "win32":
                import ctypes

                class Status(ctypes.Structure):
                    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                                ("page_total", ctypes.c_ulonglong), ("page_free", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                                ("virtual_free", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
                status = Status()
                status.length = ctypes.sizeof(Status)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    found.update(ram_total=int(status.total), ram_free=int(status.free))
            elif os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo", encoding="ascii") as stream:
                    info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in stream if line.split()[1:2]}
                found.update(ram_total=info.get("MemTotal"), ram_free=info.get("MemAvailable", info.get("MemFree")))
            else:
                pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
                found["ram_total"] = int(pages * size)
        except (OSError, ValueError, AttributeError):
            pass
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            found.update(gpu_total=int(total), gpu_free=int(free), gpu_name=str(torch.cuda.get_device_name())[:80])
    except Exception:  # noqa: BLE001 - a busy or missing device reads as unknown
        pass
    return found


def _clamped(values):
    """Each value within [FLOOR, CEILING]; history never more than half this computer's RAM."""
    out = {}
    for key, value in values.items():
        if key not in FIELDS or type(value) is not int:
            raise ValueError(f"Unknown or invalid capture limit {key!r}")
        out[key] = max(FLOOR[key], min(CEILING[key], value))
    total = machine_memory()["ram_total"]
    if total:
        out["history_bytes"] = min(out["history_bytes"], max(PROFILES["standard"]["history_bytes"], total // 2))
    return out


def current():
    """The active limits, as a plain dict of numbers."""
    with _lock:
        return dict(_active["values"])


def profile():
    with _lock:
        return _active["profile"]


def get(key):
    with _lock:
        return _active["values"][key]


@contextmanager
def override(**values):
    """Set limits as given, past the floor, for the length of a block: for tests."""
    with _lock:
        saved = dict(_active["values"])
        _active["values"].update(values)
    try:
        yield
    finally:
        with _lock:
            _active["values"] = saved


def configure(name, values=None):
    """Make a preset, or Custom with the given values (others from Standard), the active limits."""
    if name not in LABELS:
        raise ValueError("Capture size must be standard, large, maximum or custom")
    if name == "custom":
        if not isinstance(values, dict):
            raise ValueError("Custom capture size needs a value for each limit to change")
        chosen = _clamped({**PROFILES["standard"], **values})
    else:
        chosen = _clamped(PROFILES[name])
    with _lock:
        _active.update(profile=name, values=chosen)
    return describe()


def describe():
    """What the settings dialog shows: the active limits, every preset and this computer's memory."""
    with _lock:
        active = {"profile": _active["profile"], "values": dict(_active["values"])}
    try:
        import torch  # noqa: F401 - loaded here so the graphics card shows before any model runs
    except ImportError:
        pass
    return {**active, "profiles": PROFILES, "floor": FLOOR, "ceiling": CEILING,
            "fields": [{"key": key, "label": field[0], "help": field[1]} for key, field in FIELDS.items()],
            "machine": machine_memory()}


def stamp(trace):
    """Record the limits a capture ran under; they show in Run Metrics beside what was kept."""
    trace["capture_limits"] = {"profile": profile(), "values": current()}
    trace.setdefault("kept", [])
    return trace


def kept(trace, what, needed, kept_count, note=None):
    """One row of Run Metrics' "What was kept": what the run held, and how much of it was stored."""
    rows = trace.setdefault("kept", [])
    rows[:] = [row for row in rows if row["what"] != what]
    row = {"what": str(what)[:80], "needed": str(needed)[:80], "kept": str(kept_count)[:80]}
    if note:
        row["note"] = str(note)[:240]
    rows.append(row)


def validate_kept(trace):
    """The capture_limits and kept blocks of a stored trace."""
    if "capture_limits" in trace:
        block = trace["capture_limits"]
        if (not isinstance(block, dict) or block.get("profile") not in LABELS or not isinstance(block.get("values"), dict)
                or any(key not in FIELDS or type(value) is not int or value < 1 for key, value in block["values"].items())):
            raise ValueError("Invalid capture limits")
    if "kept" in trace:
        rows = trace["kept"]
        if (not isinstance(rows, list) or len(rows) > 64
                or any(not isinstance(row, dict) or set(row) - {"what", "needed", "kept", "note"}
                       or any(not isinstance(row.get(k), str) or len(row[k]) > 80 for k in ("what", "needed", "kept"))
                       or ("note" in row and (not isinstance(row["note"], str) or len(row["note"]) > 240)) for row in rows)):
            raise ValueError("Invalid kept rows")


def check_memory(needed_bytes, what, *, device=None):
    """Stop before a capture that would hold more at once than this computer has free.

    ``device`` is the torch device the bytes live on; graphics memory is checked for a CUDA device, RAM otherwise.
    Unknown free memory never stops a run.
    """
    memory = machine_memory()
    on_gpu = device is not None and str(device).startswith("cuda")
    free = memory["gpu_free"] if on_gpu else memory["ram_free"]
    if not free or needed_bytes <= free * 0.85:
        return
    where = "graphics memory" if on_gpu else "RAM"
    raise ValueError(f"{what} would hold about {_size(needed_bytes)} at once, more than the {_size(free)} of {where} free now. "
                     "Shorten the prompt, or close other programs and run again.")


def _size(count):
    return f"{count / GiB:.1f} GB" if count >= GiB else f"{math.ceil(count / MiB)} MB"


def pool_attention(weights, side):
    """Attention weights [..., T, T] shrunk to at most side × side blocks, or unchanged when T fits.

    Each block's value is the attention its queries gave its keys: summed over the block's keys and averaged over
    its queries, so every row still sums to 1. Returns (weights, ranges) where ranges lists each block's
    [first, last + 1] token, or None when nothing was shrunk.
    """
    import torch
    length = int(weights.shape[-1])
    if length <= side:
        return weights, None
    block = math.ceil(length / side)
    count = math.ceil(length / block)
    padded = count * block - length
    if padded:
        weights = torch.nn.functional.pad(weights, (0, padded, 0, padded))
    shape = weights.shape[:-2]
    grouped = weights.reshape(*shape, count, block, count, block).sum(-1).sum(-2)
    rows = torch.full((count,), float(block), dtype=grouped.dtype, device=grouped.device)
    rows[-1] = block - padded
    ranges = [[i * block, min(length, (i + 1) * block)] for i in range(count)]
    return grouped / rows.view(*([1] * len(shape)), count, 1), ranges


def pooled_tokens(tokens, ranges):
    """One label per block: the block's decoded text, shortened, with its first token's id."""
    out = []
    for first, last in ranges:
        text = "".join(str(t.get("text", "")) for t in tokens[first:last]).strip() or f"{first}–{last - 1}"
        out.append({"id": int(tokens[first]["id"]) if type(tokens[first].get("id")) is int else first,
                    "text": text if len(text) <= 24 else text[:23] + "…"})
    return out
