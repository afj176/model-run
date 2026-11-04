import unittest

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, GPT2Config, GPT2LMHeadModel
except ImportError:
    torch = None

from model_run.application import Application


@unittest.skipIf(torch is None, "optional torch/transformers dependencies unavailable")
class ActivationPatchingTests(unittest.TestCase):
    def test_patching_recovers_source_answer_and_restores_model(self):
        from model_run.transformers_capture import capture_activation_patching
        from model_run.trace_store import encode_trace
        cases = [
            Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)),
            GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=32, n_layer=3, n_head=4, n_positions=32, bos_token_id=1, eos_token_id=2)),
        ]
        source = torch.tensor([[1, 2, 3, 4]])
        target = torch.tensor([[1, 2, 3, 5]])
        for model in cases:
            with self.subTest(model=type(model).__name__):
                model.eval()
                with torch.inference_mode():
                    reference = torch.softmax(model(input_ids=target, use_cache=False).logits[0, -1].float(), dim=-1)
                    source_reference = torch.softmax(model(input_ids=source, use_cache=False).logits[0, -1].float(), dim=-1)
                result = capture_activation_patching(model, source, target, top_k=5)
                self.assertEqual(result["layer_count"], 3)
                self.assertEqual([e["layer"] for e in result["layers"]], [0, 1, 2])
                self.assertEqual(result["target_run"]["top_tokens"][0]["token_id"], int(torch.argmax(reference)))
                self.assertEqual(result["source_run"]["top_tokens"][0]["token_id"], int(torch.argmax(source_reference)))
                # Patching the last layer's output at the last position reproduces the source distribution exactly
                # (only the final norm and head follow), so recovery is 1 within rounding.
                last = result["layers"][-1]
                self.assertAlmostEqual(last["source_answer_probability"], float(source_reference.max()), places=4)
                self.assertAlmostEqual(last["recovery"], 1.0, places=3)
                self.assertTrue(last["top_is_source"])
                self.assertTrue(all(e["kl_bits"] >= 0 for e in result["layers"]))
                with torch.inference_mode():
                    again = torch.softmax(model(input_ids=target, use_cache=False).logits[0, -1].float(), dim=-1)
                torch.testing.assert_close(again, reference)
                every = capture_activation_patching(model, source, target, positions="all", layers=[1], top_k=5)
                self.assertEqual(len(every["layers"]), 1)
                encode_trace(every)
                with self.assertRaises(ValueError): capture_activation_patching(model, source, torch.tensor([[1, 2]]), positions="all")
                with self.assertRaises(ValueError): capture_activation_patching(model, source, target, layers=[7])
                with self.assertRaises(ValueError): capture_activation_patching(model, source, target, positions="middle")

    def test_attach_checks_target_prompt(self):
        app = Application()
        try:
            base = {"schema_version": "1.0", "kind": "pytorch_module_capture", "model_class": "fixture", "torch_version": "fixture",
                    "capture": {}, "events": [], "limitations": [], "input": {"token_count": 4, "token_ids_included": False}}
            identifier = app.publish(base, "fixture")["id"]
            dist = {"top_tokens": [{"token_id": 3, "probability": 0.5}], "entropy_bits": 1.0}
            result = {"kind": "activation_patching", "method": "residual_stream_patch", "positions": "last", "layer_count": 2,
                      "source_run": {**dist, "token_count": 4}, "target_run": {**dist, "token_count": 4, "source_answer_probability": 0.1, "target_answer_probability": 0.5},
                      "layers": [{"layer": 0, **dist, "kl_bits": 0.1, "source_answer_probability": 0.2, "target_answer_probability": 0.4, "recovery": 0.25, "top_is_source": False}],
                      "limitations": []}
            app._attach_patching(identifier, result)
            self.assertEqual(app.trace(identifier)["patching"]["layers"][0]["recovery"], 0.25)
            self.assertEqual(app.snapshot()["traces"][-1]["revision"], 2)
            wrong = dict(result, target_run={**result["target_run"], "token_count": 9})
            with self.assertRaises(ValueError): app._attach_patching(identifier, wrong)
            with self.assertRaises(ValueError): app.run({"prompt": "hi", "patch": "last"})
            with self.assertRaises(ValueError): app.run({"prompt": "hi", "patch": "sideways", "source_prompt": "x", "attach_to": identifier})
        finally:
            app.close()
