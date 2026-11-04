import unittest
try:
    import torch
except ImportError:
    torch = None
from model_run.adapters import default_registry, TorchModuleAdapter, UnsupportedCapability
from model_run.sessions import ObservationSession
from threading import Event


@unittest.skipIf(torch is None, "optional PyTorch dependency unavailable")
class AdapterTests(unittest.TestCase):
    def test_session_cancellation_preserves_partial_trace(self):
        started, release = Event(), Event()
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Identity()
            def forward(self, x):
                x = self.layer(x)
                started.set()
                release.wait(2)
                return x
        adapter = TorchModuleAdapter(Model().eval())
        with ObservationSession() as session:
            session.start(lambda control: adapter.capture({"x": torch.ones(1)}, ["layer"], control=control)[1])
            try:
                self.assertTrue(started.wait(2))
                session.cancel()
            finally:
                release.set()
            self.assertEqual(session.wait(3)["status"], "cancelled")
            partial = session.take_trace()
            self.assertEqual(partial["status"], "cancelled")
            self.assertEqual(partial["adapter_id"], "pytorch.module")
            self.assertEqual(len(partial["events"]), 1)
        adapter.capture({"x": torch.ones(1)}, ["layer"])
    def test_readiness_changes_with_model_mode(self):
        model = torch.nn.Linear(2, 1)
        adapter = TorchModuleAdapter(model)
        self.assertFalse(adapter.negotiate(["module_outputs"])["accepted"])
        with self.assertRaises(UnsupportedCapability): adapter.capture({"input": torch.ones(1, 2)}, [""])
        model.eval()
        self.assertTrue(adapter.negotiate(["module_outputs"])["accepted"])
        output, trace = adapter.capture({"input": torch.ones(1, 2)}, [""])
        self.assertEqual(list(output.shape), [1, 1])
        self.assertEqual(trace["adapter_id"], "pytorch.module")
        self.assertEqual(adapter.structure()["kind"], "module_containment")
        self.assertFalse(adapter.negotiate(["next_token", "unknown"])["accepted"])

    def test_registry_and_lock_cleanup(self):
        registry = default_registry()
        with self.assertRaises(ValueError): registry.register("pytorch.module", TorchModuleAdapter)
        with self.assertRaises(UnsupportedCapability): registry.bind("missing", object())
        adapter = registry.bind("pytorch.module", torch.nn.Identity().eval())
        with self.assertRaises(ValueError): adapter.capture({}, ["missing"])
        output, _ = adapter.capture({"input": torch.ones(1)}, [""])
        self.assertEqual(output.item(), 1.)
        adapter._capture_lock.acquire()
        try:
            with self.assertRaises(RuntimeError): adapter.capture({"input": torch.ones(1)}, [""])
        finally:
            adapter._capture_lock.release()
