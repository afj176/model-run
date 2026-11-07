"""Bounded, non-executing inspection of local safetensors artifacts."""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

MAX_HEADER = 16 * 1024 * 1024
MAX_INDEX = 16 * 1024 * 1024
MAX_SHARDS = 4096
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
               "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
               "F8_E4M3": 1, "F8_E5M2": 1, "F8_E4M3FN": 1, "F8_E5M2FNUZ": 1,
               "F8_E4M3FNUZ": 1, "F8_E8M0": 1}


class InventoryError(ValueError):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InventoryError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data):
    try:
        return json.loads(data, object_pairs_hook=_pairs, parse_constant=lambda x: (_ for _ in ()).throw(InventoryError(f"Invalid constant: {x}")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InventoryError(f"Invalid JSON: {exc}") from exc


MAX_GGUF_COUNT = 1_000_000  # far above any real model's tensors or metadata entries


def check_gguf(path: Path) -> dict:
    """Read a GGUF file's fixed 24-byte header, never its tensors; raise InventoryError if it is not GGUF."""
    path = Path(path)
    with path.open("rb") as stream:
        head = stream.read(24)
    if len(head) < 24 or head[:4] != b"GGUF":
        raise InventoryError("It does not start with the GGUF signature")
    version, tensors, entries = struct.unpack("<IQQ", head[4:])
    if version not in (2, 3):
        raise InventoryError(f"GGUF version {version} is not one ComfyUI-GGUF reads")
    if not 1 <= tensors <= MAX_GGUF_COUNT or entries > MAX_GGUF_COUNT:
        raise InventoryError(f"Its header counts are not plausible ({tensors} tensors, {entries} metadata entries)")
    return {"format": "gguf", "version": version, "tensor_count": tensors, "metadata_count": entries, "size_bytes": path.stat().st_size}


def inspect_file(path: Path) -> dict:
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise InventoryError("File is too short for a safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if length == 0 or length > MAX_HEADER or length > size - 8:
            raise InventoryError("Invalid or oversized safetensors header")
        raw_header = stream.read(length)
        if not raw_header.startswith(b"{"):
            raise InventoryError("Safetensors header must start with '{'")
        header = _json(raw_header)
    if not isinstance(header, dict):
        raise InventoryError("Header must be an object")
    data_size = size - 8 - length
    tensors, intervals, unknown_dtypes = [], [], set()
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise InventoryError("Metadata must contain string values")
    for name, spec in header.items():
        if not isinstance(spec, dict):
            raise InventoryError(f"Invalid tensor descriptor: {name}")
        dtype, shape, offsets = spec.get("dtype"), spec.get("shape"), spec.get("data_offsets")
        if not isinstance(dtype, str) or not dtype or not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
            raise InventoryError(f"Invalid dtype or shape: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise InventoryError(f"Invalid offsets: {name}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise InventoryError(f"Offsets out of bounds: {name}")
        if dtype in DTYPE_BYTES:
            expected = math.prod(shape) * DTYPE_BYTES[dtype]
            if expected != end - start:
                raise InventoryError(f"Tensor byte length does not match dtype and shape: {name}")
        else:
            unknown_dtypes.add(dtype)
        if start != end:
            intervals.append((start, end, name))
        tensors.append({"name": name, "dtype": dtype, "shape": shape, "data_offsets": offsets})
    intervals.sort()
    cursor = 0
    for start, end, name in intervals:
        if start != cursor:
            raise InventoryError(f"Tensor data has a gap or overlap at: {name}")
        cursor = end
    if cursor != data_size:
        raise InventoryError("Tensor data contains unindexed bytes")
    return {"path": str(path), "size_bytes": size, "format": "safetensors", "metadata": metadata, "tensors": tensors,
            "limitations": ([f"Byte lengths not validated for unknown or sub-byte dtypes: {', '.join(sorted(unknown_dtypes))}"] if unknown_dtypes else []) + ["Tensor values are not read; no architecture or execution is inferred."]}


def inspect_index(path: Path) -> dict:
    path = Path(path).resolve()
    if path.stat().st_size > MAX_INDEX:
        raise InventoryError("Index exceeds size limit")
    index = _json(path.read_bytes())
    mapping = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise InventoryError("Index requires a nonempty weight_map")
    if any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in mapping.items()):
        raise InventoryError("Invalid weight_map")
    names = set(mapping.values())
    if len(names) > MAX_SHARDS:
        raise InventoryError("Too many shards")
    root = path.parent
    shards = []
    actual = {}
    for name in sorted(names):
        shard = (root / name).resolve()
        if not shard.is_relative_to(root) or shard == root or Path(name).is_absolute():
            raise InventoryError(f"Shard escapes index directory: {name}")
        if not shard.is_file():
            raise InventoryError(f"Missing shard: {name}")
        report = inspect_file(shard)
        for tensor in report["tensors"]:
            key = tensor["name"]
            if key in actual:
                raise InventoryError(f"Duplicate tensor across shards: {key}")
            actual[key] = name
        shards.append(report)
    if actual != mapping:
        missing = sorted(set(mapping) - set(actual))
        unexpected = sorted(set(actual) - set(mapping))
        wrong = sorted(k for k in actual.keys() & mapping.keys() if actual[k] != mapping[k])
        raise InventoryError(f"Index mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}, wrong_shard={wrong[:10]}")
    return {"path": str(path), "format": "safetensors_index", "shards": shards, "tensor_count": len(actual)}


def inspect(path: Path) -> dict:
    path = Path(path)
    if path.name.endswith(".safetensors.index.json"):
        result = inspect_index(path)
    elif path.suffix.lower() == ".safetensors":
        result = inspect_file(path)
    else:
        raise InventoryError("Unsupported artifact; select a safetensors file or index")
    return {"schema_version": "1.0", "artifact": result, "capabilities": {
        "inventory": {"status": "partial", "reason": "Header metadata only; tensor values are not read."},
        "structure": {"status": "unsupported", "reason": "Weights do not define an executable graph."},
        "execution": {"status": "unsupported", "reason": "No runtime adapter is installed."},
        "observation": {"status": "unsupported", "reason": "No instrumented runtime is installed."}}}
