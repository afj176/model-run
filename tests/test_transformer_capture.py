import unittest

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, GPT2Config, GPT2LMHeadModel
except ImportError:
    torch = None

from model_run.capture import capture_forward
from model_run.transformers_capture import capture_next_token, capture_attention
from model_run.adapters import CausalLMAdapter


@unittest.skipIf(torch is None, "optional torch/transformers dependencies unavailable")
class TransformerCaptureTests(unittest.TestCase):
    def test_two_transformer_architectures(self):
        cases = [
            (Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)), "model.layers.0"),
            (GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=4, n_positions=32, bos_token_id=1, eos_token_id=2)), "transformer.h.0"),
        ]
        for model, path in cases:
            with self.subTest(model=type(model).__name__):
                model.eval()
                inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "use_cache": False}
                with torch.inference_mode(): reference = model(**inputs).logits
                output, trace = capture_forward(model, inputs, [path], max_tensor_elements=16)
                torch.testing.assert_close(output.logits, reference)
                self.assertEqual(trace["events"][0]["shape"], [1, 3, 32])
                self.assertTrue(trace["events"][0]["truncated"])
                _, token_trace = capture_next_token(model, inputs["input_ids"], [path], top_k=5)
                probs = torch.softmax(reference[0, -1].float(), dim=-1)
                expected = torch.topk(probs, 5)
                self.assertEqual([v["token_id"] for v in token_trace["output"]["top_tokens"]], expected.indices.tolist())
                lens = token_trace["logit_lens"]
                self.assertEqual(len(lens["layers"]), 3)
                self.assertEqual(lens["final_token_id"], expected.indices[0].item())
                # The last lens layer is the model's own final state, so it must reproduce the output distribution.
                self.assertEqual(lens["layers"][-1]["top_tokens"][0]["token_id"], expected.indices[0].item())
                self.assertAlmostEqual(lens["layers"][-1]["top_tokens"][0]["probability"], expected.values[0].item(), places=4)
                self.assertGreaterEqual(token_trace["output"]["entropy_bits"], 0)
                from model_run.trace_store import encode_trace
                encode_trace(token_trace)
                torch.testing.assert_close(torch.tensor([v["probability"] for v in token_trace["output"]["top_tokens"]]), expected.values)
                adapter = CausalLMAdapter(model)
                self.assertTrue(adapter.negotiate(["next_token"])["accepted"])
                self.assertFalse(adapter.negotiate(["attention"])["accepted"])
                bound_output, bound_trace = adapter.next_token(inputs["input_ids"], [path], top_k=5)
                torch.testing.assert_close(bound_output.logits, reference)
                self.assertEqual(bound_trace["adapter_id"], "transformers.causal_lm")

    def test_attention_and_bounds(self):
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, attn_implementation="eager")).eval()
        ids = torch.tensor([[1, 2, 3]])
        result = capture_attention(model, ids, 0, 0)
        self.assertEqual(result["shape"], [3, 3])
        torch.testing.assert_close(torch.tensor(result["values"]).sum(-1), torch.ones(3))
        self.assertEqual(len(result["heads"]), 4)
        self.assertEqual(result["head_count"], 4)
        self.assertEqual(result["layer_count"], 2)
        torch.testing.assert_close(torch.tensor(result["heads"][0]), torch.tensor(result["values"]), atol=1e-5, rtol=0)
        self.assertEqual(len(result["attribution"]["values"]), 3)
        self.assertAlmostEqual(sum(result["attribution"]["values"]), 1.0, places=5)
        self.assertAlmostEqual(sum(result["attribution"]["last_position_attention"]), 1.0, places=5)
        self.assertEqual([e["index"] for e in result["layers"]], [0, 1])
        self.assertTrue(all(0 <= e["sink_mass"] <= 1 and e["entropy_bits"] >= 0 for e in result["layers"]))
        from model_run.trace_store import encode_trace
        encode_trace(result)
        self.assertNotIn("heads", capture_attention(model, ids, 0, 0, all_heads=False))
        with self.assertRaises(ValueError): capture_attention(model, ids, 0, 99)
        with self.assertRaises(ValueError): capture_attention(model, ids, 0, 0, max_tokens=2)
