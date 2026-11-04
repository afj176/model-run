from threading import Event, Thread
import unittest
from model_run.host import ModelHost
from model_run.adapters import UnsupportedCapability


class FixtureAdapter:
    def __init__(self, *, started=None, release=None):
        self.started, self.release = started, release
    def capabilities(self):
        return {"adapter_id": "fixture", "features": {"module_outputs": {"status": "available"}}}
    def negotiate(self, names):
        return {"accepted": names == ["module_outputs"]}
    def capture(self, inputs, paths, *, control):
        control.check()
        if self.started:
            self.started.set()
            self.release.wait(2)
        control.check()
        return object(), {"schema_version": "1.0", "kind": "pytorch_module_capture",
            "model_class": "fixture", "torch_version": "fixture", "capture": {}, "events": [], "limitations": []}


class HostTests(unittest.TestCase):
    def test_registration_dispatch_and_release(self):
        released = []
        with ModelHost(max_models=1) as host:
            adapter = FixtureAdapter()
            host.register("first", adapter, release=lambda: released.append("first"))
            with self.assertRaises(RuntimeError): host.register("second", FixtureAdapter())
            host.submit("first", "module_outputs", inputs={}, module_paths=["layer"])
            self.assertEqual(host.wait(3)["run"]["status"], "completed")
            self.assertEqual(host.take_trace()["host_model_id"], "first")
            with self.assertRaises(UnsupportedCapability): host.submit("first", "next_token", inputs={})
            host.unregister("first")
            self.assertEqual(host.models(), [])
            host.register("first", FixtureAdapter())
            host.submit("first", "module_outputs", inputs={})
            self.assertEqual(host.wait(3)["run"]["status"], "completed")
            self.assertEqual(host.take_trace()["host_model_id"], "first")
        self.assertEqual(released, ["first"])
        with self.assertRaises(RuntimeError): host.register("x", FixtureAdapter())

    def test_cross_model_exclusion_and_cancel(self):
        started, release = Event(), Event()
        with ModelHost() as host:
            host.register("a", FixtureAdapter(started=started, release=release))
            host.register("b", FixtureAdapter())
            host.submit("a", "module_outputs", inputs={})
            try:
                self.assertTrue(started.wait(2))
                with self.assertRaises(RuntimeError): host.submit("b", "module_outputs", inputs={})
                with self.assertRaises(RuntimeError): host.unregister("a")
                host.cancel()
            finally:
                release.set()
            self.assertEqual(host.wait(3)["run"]["status"], "cancelled")
            host.submit("b", "module_outputs", inputs={})
            self.assertEqual(host.wait(3)["run"]["status"], "completed")
            host.take_trace()

    def test_release_failure_is_retained_for_retry(self):
        attempts = []
        def release():
            attempts.append(1)
            if len(attempts) == 1:
                raise ValueError("fixture release error")
        host = ModelHost()
        host.register("a", FixtureAdapter(), release=release)
        errors = host.close()
        self.assertEqual(len(errors), 1)
        self.assertEqual(host.models()[0]["state"], "release_failed")
        self.assertEqual(host.close(), [])
        self.assertEqual(host.models(), [])

    def test_duplicate_resource_and_options(self):
        host = ModelHost()
        try:
            adapter = FixtureAdapter()
            host.register("a", adapter)
            with self.assertRaises(ValueError): host.register("b", adapter)
            with self.assertRaises(ValueError): host.submit("a", "module_outputs", inputs={}, options={"control": None})
        finally:
            host.close()

    def test_release_excludes_observation_and_preserves_original_error(self):
        started, release = Event(), Event()
        host = ModelHost()
        def cleanup():
            started.set()
            release.wait(2)
        host.register("a", FixtureAdapter(), release=cleanup)
        host.register("b", FixtureAdapter())
        worker = Thread(target=lambda: host.unregister("a"))
        worker.start()
        try:
            self.assertTrue(started.wait(2))
            with self.assertRaises(RuntimeError): host.submit("b", "module_outputs", inputs={})
        finally:
            release.set()
            worker.join(3)
            host.close()
        self.assertFalse(worker.is_alive())
        def bad_cleanup():
            raise ValueError("cleanup failure")
        with self.assertRaisesRegex(RuntimeError, "original failure"):
            with ModelHost() as failing:
                failing.register("a", FixtureAdapter(), release=bad_cleanup)
                raise RuntimeError("original failure")
