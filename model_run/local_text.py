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
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
                                                 trust_remote_code=False, use_safetensors=True,
                                                 dtype=torch.float32, attn_implementation=attention_implementation)
    model.to(device)
    model.eval()
    return model, tokenizer
