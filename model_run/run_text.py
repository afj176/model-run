"""Explicit local text-model capture CLI. Prompt is read from stdin."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from .local_text import load_local_text_model
from .trace_store import save_trace
from .adapters import default_registry


def main():
    parser = argparse.ArgumentParser(description="Run one local causal model forward pass; read prompt from stdin.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--module", required=True, help="Exact PyTorch module path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--attention", nargs=2, type=int, metavar=("LAYER", "HEAD"))
    parser.add_argument("--trace", type=Path, help="Explicit destination for module capture JSON")
    parser.add_argument("--attention-trace", type=Path, help="Explicit destination for attention JSON")
    parser.add_argument("--fingerprint-file", action="append", default=[], help="Relative model file to hash; may be repeated")
    args = parser.parse_args()
    if args.attention_trace and not args.attention:
        parser.error("--attention-trace requires --attention")
    if args.trace and args.attention_trace and args.trace.resolve() == args.attention_trace.resolve():
        parser.error("trace destinations must be different")
    try:
        prompt = sys.stdin.read(65537)
        if not prompt or len(prompt) > 65536:
            raise ValueError("Prompt must contain 1 to 65536 characters")
        model, tokenizer = load_local_text_model(args.model_dir, device=args.device,
            attention_implementation="eager" if args.attention else None)
        ids = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"].to(args.device)
        adapter = default_registry().bind("transformers.causal_lm", model)
        _, trace = adapter.next_token(ids, [args.module])
        config = args.model_dir / "config.json"
        trace["provenance"] = {"model_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                               "model_type": getattr(model.config, "model_type", None), "device": args.device,
                               "input_mode": "raw_tokenizer", "checkpoint_identity": "not verified"}
        if args.fingerprint_file:
            from .identity import fingerprint_files
            trace["provenance"]["artifact_fingerprint"] = fingerprint_files(args.model_dir, args.fingerprint_file)
        attention = adapter.probe_attention(ids, *args.attention) if args.attention else None
        if args.trace:
            save_trace(args.trace, trace)
        if args.attention_trace:
            attention["provenance"] = trace["provenance"]
            save_trace(args.attention_trace, attention)
        print(json.dumps({"model_type": trace["provenance"]["model_type"], "token_count": int(ids.shape[1]),
                          "top_tokens": trace["output"]["top_tokens"], "events": len(trace["events"]),
                          "attention_shape": attention["shape"] if attention else None,
                          "trace_saved": bool(args.trace), "attention_trace_saved": bool(args.attention_trace)}))
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
