"""Explicit local-only loader for supported Transformers causal language models."""
from pathlib import Path


def load_local_text_model(directory, *, device="cpu", attention_implementation=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(directory).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Model path must be a local directory")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if attention_implementation not in (None, "eager"):
        raise ValueError("attention_implementation must be None or eager")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    dtype = fitting_dtype(path, device)
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
                                                 trust_remote_code=False, use_safetensors=True,
                                                 dtype=dtype, attn_implementation=attention_implementation)
    model.to(device)
    model.eval()
    return model, tokenizer


def fitting_dtype(path, device, free_bytes=None):
    """float32, unless a 16-bit checkpoint would not fit the GPU at that size: then its own 16-bit type."""
    import json
    import torch
    if device != "cuda":
        return torch.float32
    try:
        config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return torch.float32
    stored = config.get("torch_dtype") or config.get("dtype") or (config.get("text_config") or {}).get("torch_dtype") or (config.get("text_config") or {}).get("dtype")
    if stored not in ("bfloat16", "float16"):
        return torch.float32
    weights = sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))
    if free_bytes is None:
        free_bytes = torch.cuda.mem_get_info()[0]
    # Weights take twice their stored size in float32; a tenth is left for activations.
    return getattr(torch, stored) if weights * 2 > free_bytes * 0.9 else torch.float32
