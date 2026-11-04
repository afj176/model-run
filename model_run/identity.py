"""Explicit content identity for local model artifacts."""
import hashlib
from pathlib import Path

CHUNK_SIZE = 1024 * 1024
MAX_FILES = 4096


def fingerprint_files(root, names):
    """Hash explicit files without executing them; returns no inferred model identity."""
    root = Path(root).resolve(strict=True)
    if not root.is_dir() or not isinstance(names, (list, tuple)) or not names or len(names) > MAX_FILES:
        raise ValueError("A model directory and a bounded nonempty file list are required")
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("File names must be unique nonempty strings")
    files = []
    for name in sorted(names):
        if Path(name).is_absolute():
            raise ValueError("Absolute file paths are not allowed")
        target = (root / name).resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError("File must remain within the model directory")
        hasher = hashlib.sha256()
        with target.open("rb") as stream:
            before = stream.seek(0, 2)
            stream.seek(0)
            while chunk := stream.read(CHUNK_SIZE):
                hasher.update(chunk)
            after = stream.seek(0, 2)
        if before != after:
            raise ValueError(f"File size changed while hashing: {name}")
        files.append({"path": name, "size_bytes": before, "sha256": hasher.hexdigest()})
    return {"algorithm": "sha256", "files": files,
            "limitations": ["Only explicitly listed files are included.",
                            "A hash identifies bytes, not model provenance or trustworthiness.",
                            "Files modified in place while hashing may produce an inconsistent digest."]}
