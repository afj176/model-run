"""Bounded discovery of model artifacts under explicit configured roots."""
import os
from pathlib import Path
import re
from .inventory import InventoryError, inspect_file, inspect_index

MAX_FILES = 10000
MAX_DEPTH = 12
PLACEHOLDER = re.compile(r"put_.+_here")


def discover_models(entries, *, max_files=MAX_FILES, max_depth=MAX_DEPTH):
    """Scan explicit roots without following symlinks or opening tensor values.

    Each entry is a path/category mapping; duplicate physical files retain all origins.
    Unsupported extensions remain visible but are never deserialized.
    """
    if not isinstance(entries, list) or type(max_files) is not int or not 1 <= max_files <= MAX_FILES or type(max_depth) is not int or not 0 <= max_depth <= MAX_DEPTH:
        raise ValueError("Invalid discovery configuration")
    artifacts, errors, seen = [], [], {}
    visited_files, truncated, depth_limited = 0, False, False
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("category"), str):
            raise ValueError("Each root requires path and category strings")
        root = Path(entry["path"])
        # ComfyUI lists folders it creates only when it first saves a model there (output/checkpoints
        # and the like), so a folder that is not there yet is expected, not a problem.
        if not root.exists():
            continue
        if not root.is_dir():
            errors.append({"path": str(root), "error": "This is a file, not a folder"})
            continue
        root = root.resolve()
        origin = {"root": str(root), "category": entry["category"], "section": entry.get("section")}
        def onerror(error):
            reason = getattr(error, "strerror", None) or str(error)
            errors.append({"path": str(error.filename), "error": f"Can't be read: {reason[:1].lower()}{reason[1:]}"})
        for directory, folders, files in os.walk(root, followlinks=False, onerror=onerror):
            depth = len(Path(directory).relative_to(root).parts)
            eligible = sorted(d for d in folders if not (Path(directory) / d).is_symlink())
            if depth >= max_depth and eligible:
                depth_limited = True
            folders[:] = eligible if depth < max_depth else []
            for name in sorted(files):
                candidate = Path(directory) / name
                # ComfyUI ships an empty put_checkpoints_here in each of its folders.
                if candidate.is_symlink() or PLACEHOLDER.fullmatch(name):
                    continue
                key = os.path.normcase(str(candidate.resolve()))
                if key in seen:
                    if origin not in seen[key]["origins"]:
                        seen[key]["origins"].append(origin)
                    continue
                if visited_files >= max_files:
                    truncated = True
                    break
                visited_files += 1
                record = {"path": str(candidate), "origins": [origin], "format": "unknown", "status": "uninspected"}
                seen[key] = record
                artifacts.append(record)
                if candidate.suffix.lower() == ".safetensors":
                    try:
                        info = inspect_file(candidate)
                        dtype_counts = {}
                        for tensor in info["tensors"]:
                            dtype_counts[tensor["dtype"]] = dtype_counts.get(tensor["dtype"], 0) + 1
                        record.update({"format": "safetensors", "status": "metadata_inspected",
                                       "size_bytes": info["size_bytes"], "tensor_count": len(info["tensors"]),
                                       "tensor_dtypes": dtype_counts,
                                       "limitations": info["limitations"]})
                    except (OSError, InventoryError) as exc:
                        record.update({"format": "safetensors", "status": "invalid_or_unreadable", "error": str(exc)})
                elif candidate.name.endswith(".safetensors.index.json"):
                    try:
                        info = inspect_index(candidate)
                        record.update({"format": "safetensors_index", "status": "metadata_inspected",
                                       "tensor_count": info["tensor_count"],
                                       "shard_paths": [shard["path"] for shard in info["shards"]],
                                       "limitations": ["Shard membership is declared by this index, not inferred from names.",
                                                       "Architecture and execution have not been verified."]})
                    except (OSError, InventoryError) as exc:
                        record.update({"format": "safetensors_index", "status": "invalid_or_unreadable", "error": str(exc)})
            if truncated:
                break
        if truncated:
            break
    for record in artifacts:
        record["configured_categories"] = sorted({origin["category"] for origin in record["origins"]})
        record["capabilities"] = {
            "inventory": {"status": "partial" if record["status"] == "metadata_inspected" else "unavailable",
                          "reason": "Header metadata only" if record["status"] == "metadata_inspected" else "No validated metadata reader result"},
            "structure": {"status": "unavailable", "reason": "Catalog does not infer an executable graph from files"},
            "execution": {"status": "unverified", "reason": "No runtime probe performed"},
            "observation": {"status": "unverified", "reason": "No capture probe performed"},
        }
    return {"schema_version": "1.0", "kind": "artifact_catalog", "artifacts": artifacts,
            "errors": errors, "truncated": truncated, "depth_limited": depth_limited, "limits": {"max_files": max_files, "max_depth": max_depth},
            "limitations": ["Only supplied roots are scanned; symbolic links are skipped.",
                            "File extensions select readers; unsupported files are not deserialized.",
                            "Categories are configured locations, not proof of artifact type or compatibility.",
                            "Metadata inventory does not establish architecture or execution support."]}
