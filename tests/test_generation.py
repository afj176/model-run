import unittest

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, GPT2Config, GPT2LMHeadModel
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional torch/transformers dependencies unavailable")
class GenerationCaptureTests(unittest.TestCase):
    def test_greedy_steps_with_provenance(self):
        from model_run.transformers_capture import capture_generation
        from model_run.trace_store import encode_trace
        cases = [
            Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, attn_implementation="eager", eos_token_id=63)),
            GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=4, n_positions=64, bos_token_id=1, eos_token_id=63, attn_implementation="eager")),
        ]
        ids = torch.tensor([[1, 2, 3]])
        for model in cases:
            with self.subTest(model=type(model).__name__):
                model.eval()
                result = capture_generation(model, ids, max_new_tokens=4, top_k=3)
                self.assertEqual(result["kind"], "generation_capture")
                self.assertIn(result["stopped"], ("max_new_tokens", "eos"))
                self.assertTrue(1 <= len(result["steps"]) <= 4)
                first = result["steps"][0]
                self.assertEqual(first["position"], 3)
                self.assertEqual(first["top_tokens"][0]["token_id"], first["token_id"])
                self.assertEqual(len(first["attention"]), 3)
                self.assertAlmostEqual(sum(first["attention"]), 1.0, places=3)
                self.assertEqual(len(first["lens"]["chosen_probability_by_layer"]), 3)
                # The final lens state is the model's own output, so the chosen token is on top there.
                self.assertIsNotNone(first["lens"]["settled_layer"])
                self.assertLessEqual(first["lens"]["first_layer"], first["lens"]["settled_layer"])
                # Greedy continuation matches the model's own generate().
                with torch.inference_mode():
                    expected = model.generate(ids, max_new_tokens=len(result["steps"]), do_sample=False, pad_token_id=0)[0, 3:].tolist()
                self.assertEqual([s["token_id"] for s in result["steps"]], expected[: len(result["steps"])])
                if len(result["steps"]) > 1:
                    self.assertEqual(len(result["steps"][1]["attention"]), 4)
                encode_trace(result)
                bare = capture_generation(model, ids, max_new_tokens=2, provenance=False)
                self.assertNotIn("attention", bare["steps"][0])
                with self.assertRaises(ValueError): capture_generation(model, ids, max_new_tokens=0)
                with self.assertRaises(ValueError): capture_generation(model, ids, max_new_tokens=500)
