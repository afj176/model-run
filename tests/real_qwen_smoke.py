"""Opt-in integration smoke: python tests/real_qwen_smoke.py MODEL_DIR [cpu|cuda]."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model_run.local_text import load_local_text_model
from model_run.adapters import default_registry
from model_run.control import RunControl, ObservationCancelled
from model_run.host import ModelHost

path = Path(sys.argv[1])
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
started = time.perf_counter()
model, tokenizer = load_local_text_model(path, device=device, attention_implementation="eager")
input_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"].to(device)
with torch.inference_mode():
    baseline = model(input_ids=input_ids, use_cache=False).logits
adapter = default_registry().bind("transformers.causal_lm", model)
observed, trace = adapter.next_token(input_ids, ["model.layers.0", "model.layers.27"], top_k=5)
torch.testing.assert_close(observed.logits, baseline)
reference_probabilities = torch.softmax(baseline[0, -1].float(), dim=-1)
reference_top = torch.topk(reference_probabilities, 5)
assert [item["token_id"] for item in trace["output"]["top_tokens"]] == reference_top.indices.tolist()
torch.testing.assert_close(torch.tensor([item["probability"] for item in trace["output"]["top_tokens"]]), reference_top.values.cpu())
assert trace["events"][0]["shape"][-1] == model.config.hidden_size == 1024
assert len(model.model.layers) == 28
attention = adapter.probe_attention(input_ids, 0, 0)
with torch.inference_mode():
    reference_attention = model(input_ids=input_ids, use_cache=False, output_attentions=True).attentions[0][0, 0].float().cpu()
torch.testing.assert_close(torch.tensor(attention["values"]), reference_attention)
torch.testing.assert_close(reference_attention.sum(-1), torch.ones(input_ids.shape[1]), atol=1e-5, rtol=1e-5)
assert torch.allclose(reference_attention.triu(1), torch.zeros_like(reference_attention), atol=1e-6)
control = RunControl(on_progress=lambda event: control.cancel() if event["stage"] == "module:model.layers.0" else None)
existing_hooks = set(model.model.layers[0]._forward_hooks)
existing_pre_hooks = set(model.model.layers[0]._forward_pre_hooks)
try:
    adapter.next_token(input_ids, ["model.layers.0", "model.layers.27"], control=control)
    raise AssertionError("Expected cancellation")
except ObservationCancelled as exc:
    assert exc.trace["status"] == "cancelled" and len(exc.trace["events"]) == 1
assert set(model.model.layers[0]._forward_hooks) == existing_hooks
assert set(model.model.layers[0]._forward_pre_hooks) == existing_pre_hooks
recovered, _ = adapter.next_token(input_ids, ["model.layers.0"], top_k=5)
torch.testing.assert_close(recovered.logits, baseline)
with ModelHost() as host:
    host.register("qwen-smoke", adapter, release=lambda: model.to("cpu"))
    host.submit("qwen-smoke", "next_token", inputs={"input_ids": input_ids}, module_paths=["model.layers.0"], options={"top_k": 5})
    assert host.wait(60)["run"]["status"] == "completed"
    hosted = host.take_trace()
    assert hosted["host_model_id"] == "qwen-smoke"
    assert hosted["output"]["top_tokens"] == trace["output"]["top_tokens"]
assert next(model.parameters()).device.type == "cpu"
print(json.dumps({"model_type": model.config.model_type, "device": device,
                  "torch": str(torch.__version__), "layers": len(model.model.layers),
                  "tokens": input_ids.shape[1], "captured_modules": len(trace["events"]),
                  "top_token": tokenizer.decode([trace["output"]["top_tokens"][0]["token_id"]]),
                  "elapsed_seconds": round(time.perf_counter() - started, 2)}))
