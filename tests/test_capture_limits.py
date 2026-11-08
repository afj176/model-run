import unittest

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError:
    torch = None

from model_run import capture_limits
from model_run.application import Application, prompt_fits
from model_run.trace_store import decode_trace, encode_trace


def trace():
    return {"schema_version": "1.0", "kind": "pytorch_module_capture", "model_class": "fixture",
            "torch_version": "fixture", "capture": {}, "events": [], "limitations": []}


class CaptureLimitTests(unittest.TestCase):
    def tearDown(self):
        capture_limits.configure("standard")

    def test_presets_grow_and_standard_is_what_was_tested(self):
        standard, large, maximum = (capture_limits.PROFILES[p] for p in ("standard", "large", "maximum"))
        self.assertEqual((standard["token_rows"], standard["head_values"], standard["stack_values"], standard["history_runs"]), (128, 262144, 1048576, 16))
        self.assertEqual(standard["trace_bytes"], 64 * 1024 * 1024)
        for key in standard:
            self.assertLess(standard[key], large[key], key)
            self.assertLessEqual(large[key], maximum[key], key)

    def test_custom_values_are_clamped_and_unknown_ones_refused(self):
        described = capture_limits.configure("custom", {"token_rows": 10 ** 9, "prompt_tokens": 1})
        self.assertEqual(described["profile"], "custom")
        self.assertEqual(capture_limits.get("token_rows"), capture_limits.CEILING["token_rows"])
        self.assertEqual(capture_limits.get("prompt_tokens"), capture_limits.FLOOR["prompt_tokens"])
        with self.assertRaises(ValueError):
            capture_limits.configure("custom", {"made_up": 3})
        with self.assertRaises(ValueError):
            capture_limits.configure("huge")
        capture_limits.configure("large")
        self.assertEqual(capture_limits.get("token_rows"), 512)

    def test_a_long_prompt_names_the_setting_to_change(self):
        prompt_fits(1024)
        with self.assertRaises(ValueError) as caught:
            prompt_fits(1025)
        self.assertIn("Standard capture size", str(caught.exception))
        capture_limits.configure("large")
        prompt_fits(1025)

    def test_runs_are_stamped_and_imports_keep_their_own_record(self):
        app = Application()
        try:
            run = app.publish(trace())["id"]
            self.assertEqual(app.trace(run)["capture_limits"]["profile"], "standard")
            imported = app.publish(trace(), imported=True)["id"]
            self.assertNotIn("capture_limits", app.trace(imported))
            self.assertEqual(app.snapshot()["capture_limits"]["profile"], "standard")
        finally:
            app.close()

    def test_a_smaller_setting_trims_history_at_once(self):
        app = Application()
        try:
            for index in range(10):
                app.publish(trace(), run=f"r{index}")
            capture_limits.configure("custom", {"history_runs": 4})
            app.set_capture_limits("custom", {"history_runs": 4})
            self.assertEqual(app.snapshot()["retention"]["runs"], 4)
        finally:
            app.close()

    def test_kept_rows_are_checked(self):
        stored = trace()
        capture_limits.stamp(stored)
        capture_limits.kept(stored, "Prompt tokens", "10", "10")
        capture_limits.kept(stored, "Prompt tokens", "12", "12")  # the same row is replaced, not repeated
        self.assertEqual(decode_trace(encode_trace(stored))["kept"], [{"what": "Prompt tokens", "needed": "12", "kept": "12"}])
        stored["kept"].append({"what": "x", "needed": "1", "kept": "1", "extra": "no"})
        with self.assertRaises(ValueError):
            encode_trace(stored)

    def test_one_run_is_bounded_by_the_active_size(self):
        stored = trace()
        stored["limitations"] = ["x" * 4096] * 64
        with capture_limits.override(trace_bytes=1024):
            with self.assertRaises(ValueError) as caught:
                encode_trace(stored)
        self.assertIn("capture size", str(caught.exception))
        encode_trace(stored)

    def test_memory_check_stops_only_what_will_not_fit(self):
        capture_limits.check_memory(1, "Nothing")
        free = capture_limits.machine_memory()["ram_free"]
        if free:
            with self.assertRaises(ValueError) as caught:
                capture_limits.check_memory(free * 4, "A huge matrix")
            self.assertIn("Shorten the prompt", str(caught.exception))


@unittest.skipIf(torch is None, "optional torch/transformers dependencies unavailable")
class PooledAttentionTests(unittest.TestCase):
    def tearDown(self):
        capture_limits.configure("standard")

    def test_blocks_keep_each_row_a_distribution(self):
        weights = torch.softmax(torch.randn(3, 10, 10), dim=-1)
        pooled, ranges = capture_limits.pool_attention(weights, 4)
        self.assertEqual(ranges, [[0, 3], [3, 6], [6, 9], [9, 10]])
        self.assertEqual(tuple(pooled.shape), (3, 4, 4))
        torch.testing.assert_close(pooled.sum(-1), torch.ones(3, 4))
        # The first block's attention to the second: its three queries' mass on keys 3-5, averaged.
        torch.testing.assert_close(pooled[0, 0, 1], weights[0, 0:3, 3:6].sum(-1).mean())
        same, none = capture_limits.pool_attention(weights, 10)
        self.assertIs(same, weights)
        self.assertIsNone(none)

    def test_a_long_prompt_is_probed_in_blocks(self):
        from model_run.transformers_capture import capture_attention
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                             num_attention_heads=4, num_key_value_heads=2, attn_implementation="eager")).eval()
        ids = torch.tensor([[i % 60 + 1 for i in range(40)]])
        with capture_limits.override(attention_side=16):
            trace = capture_attention(model, ids, 1, 0)
        self.assertEqual(trace["shape"], [14, 14])  # blocks of 3 tokens
        self.assertEqual(trace["pooled"]["token_count"], 40)
        self.assertEqual(trace["input"]["token_count"], 40)
        self.assertEqual(len(trace["layer_stacks"]), 2)
        self.assertEqual(next(r for r in trace["kept"] if r["what"] == "Attention matrix")["kept"], "14 × 14")
        tokens = capture_limits.pooled_tokens([{"id": i, "text": f"t{i}"} for i in range(40)], trace["pooled"]["ranges"])
        self.assertEqual(tokens[0], {"id": 0, "text": "t0t1t2"})
        trace["tokens"] = tokens
        trace["limitations"].append("fixture")
        decode_trace(encode_trace(trace))
        trace["pooled"]["ranges"][1][0] = 4
        with self.assertRaises(ValueError):
            encode_trace(trace)


if __name__ == "__main__":
    unittest.main()
