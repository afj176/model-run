import unittest
from model_run.structure import inspect_module_tree

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency unavailable")
class StructureTests(unittest.TestCase):
    def test_hierarchy_and_truncation(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU())
        result = inspect_module_tree(model)
        self.assertEqual(result["kind"], "module_containment")
        self.assertEqual(result["nodes"][1]["parameters"][0]["shape"], [2, 3])
        self.assertFalse(result["truncated"])
        self.assertTrue(inspect_module_tree(model, max_modules=1)["truncated"])

    def test_shared_module_paths(self):
        shared = torch.nn.Linear(1, 1)
        model = torch.nn.Module()
        model.first = shared
        model.second = shared
        self.assertEqual([n["path"] for n in inspect_module_tree(model)["nodes"]], ["", "first", "second"])
