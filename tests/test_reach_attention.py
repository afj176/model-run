import sys
import types
import unittest
from unittest import mock

from model_run.comfy_bridge import reach_attention, token_labels


class Container:
    def __init__(self, value):
        self.value = value

    def peek(self):
        return self.value


class ComfyAttention:
    def __init__(self, function=None):
        self.function = function


class Core:
    def __init__(self, *slots):
        self.slots = slots

    def modules(self):
        return self.slots


class Pieces:
    def untokenize(self, pairs):
        return [(pair, {1: "\u2581a", 2: "red", 3: "</s>"}[pair[0]]) for pair in pairs]


class Tokenizer:
    def __init__(self, **encoders):
        self.__dict__.update(encoders)

    def decode(self, ids, **kwargs):
        return {1: "", 2: "red", 3: ""}.get(ids[0], "wrong")


class Clip:
    def __init__(self, tokenizer, tokens):
        self.tokenizer, self.tokens = tokenizer, tokens

    def tokenize(self, text):
        return self.tokens


class TokenLabelTests(unittest.TestCase):
    PAIRS = [[(1, 1.0), (2, 1.0), (3, 1.0)]]

    def test_each_encoder_reads_its_own_ids(self):
        # Flux and Anima: a shared decode reads the other encoder's ids, which turned prompts into wrong words.
        clip = Clip(Tokenizer(t5xxl=Pieces()), {"l": [[(9, 1.0)]], "t5xxl": self.PAIRS})
        self.assertEqual(token_labels(clip, "x")[1:], ["red", "</s>"])

    def test_a_single_encoder_fills_what_decode_leaves_empty(self):
        clip = Clip(Tokenizer(umt5xxl=Pieces()), {"umt5xxl": self.PAIRS})
        self.assertEqual(token_labels(clip, "x"), ["a", "red", "</s>"])


class ReachAttentionTests(unittest.TestCase):
    def setUp(self):
        attention = types.ModuleType("comfy.ldm.modules.attention")
        attention.optimized_attention = lambda q, k, v, heads, **kwargs: ("default", q.value, k.value, v.value, heads)
        names = {"comfy": types.ModuleType("comfy"), "comfy.ldm": types.ModuleType("comfy.ldm"),
                 "comfy.ldm.modules": types.ModuleType("comfy.ldm.modules"), "comfy.ldm.modules.attention": attention}
        patcher = mock.patch.dict(sys.modules, names)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_override_sees_tensors_and_the_kernel_still_runs(self):
        own = lambda q, k, v, heads, **kwargs: ("int8", q.value)
        plain, quantized = ComfyAttention(), ComfyAttention(own)
        seen = []
        def override(kernel, q, k, v, heads, *args, **kwargs):
            seen.append((q, k, heads))
            return kernel(q, k, v, heads, *args, **kwargs)
        with reach_attention(Core(plain, quantized), override):
            first = plain.function(Container(1), Container(2), Container(3), 4)
            second = quantized.function(Container(5), Container(6), Container(7), 8)
        self.assertEqual(seen, [(1, 2, 4), (5, 6, 8)])
        self.assertEqual(first, ("default", 1, 2, 3, 4))
        self.assertEqual(second, ("int8", 5))
        self.assertIsNone(plain.function)
        self.assertIs(quantized.function, own)

    def test_slots_are_put_back_after_a_failure(self):
        slot = ComfyAttention()
        with self.assertRaises(RuntimeError):
            with reach_attention(Core(slot), None):
                raise RuntimeError("sampling failed")
        self.assertIsNone(slot.function)


if __name__ == "__main__":
    unittest.main()
