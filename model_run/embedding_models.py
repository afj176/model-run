"""Sentence embedding models saved in the Sentence Transformers layout, run without that library.

The package's `modules.json` lists the steps after the Transformers encoder: pooling, optional dense
projections and normalisation. Each step is read from the package itself, so the vector is the one
the package defines, and the encoder's layers are recorded on the way.
"""
import json
from pathlib import Path

from .adapters import TorchModuleAdapter
from .capture import capture_forward

MAX_VALUES = 4096
POOLING_MODES = ("cls_token", "mean_tokens", "max_tokens", "lasttoken")


def is_embedding_package(directory):
    """A Sentence Transformers package: a modules.json whose first step is a Transformers encoder."""
    try:
        steps = json.loads((Path(directory) / "modules.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(steps, list) and bool(steps) and isinstance(steps[0], dict) and steps[0].get("type", "").endswith("models.Transformer") and steps[0].get("path", "") in ("", ".")


def read_head(directory):
    """The steps after the encoder, as plain data: pooling mode, dense layers, normalise."""
    import torch
    from safetensors.torch import load_file
    root = Path(directory)
    head = {"pooling": None, "dense": [], "normalize": False, "prompts": {}}
    for step in json.loads((root / "modules.json").read_text(encoding="utf-8"))[1:]:
        kind, folder = step.get("type", ""), root / step.get("path", "")
        if kind.endswith("models.Pooling"):
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            modes = [mode for mode in POOLING_MODES if config.get(f"pooling_mode_{mode}")]
            if len(modes) != 1:
                raise ValueError(f"Pooling combines {modes or 'no'} modes; only one mode is supported")
            head["pooling"] = modes[0]
        elif kind.endswith("models.Dense"):
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            activation = config.get("activation_function", "torch.nn.modules.linear.Identity")
            if not activation.endswith("Identity"):
                raise ValueError(f"Dense activation {activation} is not supported")
            weights = load_file(str(folder / "model.safetensors"))
            head["dense"].append({"weight": weights["linear.weight"].to(torch.float32), "bias": weights.get("linear.bias")})
        elif kind.endswith("models.Normalize"):
            head["normalize"] = True
        else:
            raise ValueError(f"Embedding step {kind or 'unknown'} is not supported")
    if head["pooling"] is None:
        raise ValueError("The package has no pooling step")
    try:
        head["prompts"] = json.loads((root / "config_sentence_transformers.json").read_text(encoding="utf-8")).get("prompts") or {}
    except (OSError, ValueError):
        pass
    return head


def load_local_embedding_model(directory, *, device="cpu"):
    from transformers import AutoModel, AutoTokenizer
    from .local_text import fitting_dtype
    path = Path(directory).resolve(strict=True)
    if not is_embedding_package(path):
        raise ValueError("Not a Sentence Transformers package")
    head = read_head(path)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(str(path), local_files_only=True, trust_remote_code=False, use_safetensors=True,
                                      dtype=fitting_dtype(path, device), attn_implementation="eager")
    model.to(device)
    model.eval()
    return model, tokenizer, head


class EmbeddingAdapter(TorchModuleAdapter):
    adapter_id = "sentence_transformers.embedding"

    def __init__(self, model, head):
        super().__init__(model)
        self.head = head

    def embed(self, inputs, module_paths, *, max_tensor_elements=1024, control=None, token_features=None):
        import torch
        from .transformers_capture import token_slices
        self._require("module_outputs")
        def forward():
            return capture_forward(self.model, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                                   module_paths, control=control, max_tensor_elements=max_tensor_elements,
                                   slices=token_slices(module_paths, token_features))
        output, trace = self._run(forward)
        states = output.last_hidden_state[0].detach().to(device="cpu", dtype=torch.float32)
        mask = inputs["attention_mask"][0].to("cpu").bool()
        kept = states[mask]
        mode = self.head["pooling"]
        pooled = {"cls_token": lambda: kept[0], "mean_tokens": lambda: kept.mean(0), "max_tokens": lambda: kept.max(0).values,
                  "lasttoken": lambda: kept[-1]}[mode]()
        # How much each token's final state points the way the pooled vector does, before any projection.
        alignment = torch.nn.functional.cosine_similarity(kept, pooled.unsqueeze(0), dim=-1)
        vector = pooled
        for layer in self.head["dense"]:
            vector = layer["weight"] @ vector + (layer["bias"].to(torch.float32) if layer["bias"] is not None else 0)
        norm = float(vector.norm())
        if self.head["normalize"]:
            vector = vector / max(norm, 1e-12)
        trace["embedding"] = {
            "pooling": mode, "dense_layers": [list(layer["weight"].shape) for layer in self.head["dense"]], "normalized": self.head["normalize"],
            "dimension": int(vector.numel()), "norm_before_normalize": norm, "values": [round(float(v), 6) for v in vector[:MAX_VALUES]],
            "token_alignment": [round(float(a), 4) for a in alignment], "token_count": int(mask.sum()),
            "prompts": dict(self.head["prompts"]),
            "limitations": ["Token alignment is each token's final hidden state compared with the pooled vector by cosine similarity, before dense layers. It shows which tokens point the way the sentence does, not which caused it.",
                            "Some embedding models expect a task prefix such as the package's prompts; the text is embedded exactly as given."]}
        trace["adapter_id"] = self.adapter_id
        return output, trace


def validate_embedding(record):
    """Raise ValueError unless ``record`` is a consistent embedding section."""
    import math
    finite = lambda v: type(v) in (int, float) and math.isfinite(v)
    if not isinstance(record, dict) or record.get("pooling") not in POOLING_MODES or type(record.get("normalized")) is not bool:
        raise ValueError("Invalid embedding")
    dimension, values, alignment = record.get("dimension"), record.get("values"), record.get("token_alignment")
    if type(dimension) is not int or not 1 <= dimension <= 65536 or not isinstance(values, list) or len(values) != min(dimension, MAX_VALUES):
        raise ValueError("Invalid embedding values")
    if not all(finite(v) for v in values) or not finite(record.get("norm_before_normalize")):
        raise ValueError("Invalid embedding values")
    count = record.get("token_count")
    if type(count) is not int or not 1 <= count <= 4096 or not isinstance(alignment, list) or len(alignment) != count or not all(finite(a) and -1.0001 <= a <= 1.0001 for a in alignment):
        raise ValueError("Invalid embedding token alignment")
    dense = record.get("dense_layers")
    if not isinstance(dense, list) or len(dense) > 8 or any(not isinstance(d, list) or len(d) != 2 or any(type(v) is not int or v < 1 for v in d) for d in dense):
        raise ValueError("Invalid embedding dense layers")
    prompts = record.get("prompts", {})
    if not isinstance(prompts, dict) or len(prompts) > 32 or any(not isinstance(k, str) or not isinstance(v, str) or len(k) > 128 or len(v) > 1024 for k, v in prompts.items()):
        raise ValueError("Invalid embedding prompts")
    limitations = record.get("limitations", [])
    if not isinstance(limitations, list) or any(not isinstance(t, str) for t in limitations):
        raise ValueError("Invalid embedding limitations")
