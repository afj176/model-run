from threading import Event
import unittest
from model_run.sessions import ObservationSession
from model_run.control import RunControl, ObservationCancelled


def trace():
    return {"schema_version": "1.0", "kind": "pytorch_module_capture", "model_class": "fixture.Model",
            "torch_version": "fixture", "capture": {}, "events": [], "limitations": []}


class SessionTests(unittest.TestCase):
    def test_completion_retention_and_reuse(self):
        with ObservationSession() as session:
            first = session.start(lambda control: trace())
            self.assertEqual(session.wait(3)["status"], "completed")
            with self.assertRaises(RuntimeError): session.start(lambda control: trace())
            self.assertEqual(session.take_trace(), trace())
            self.assertIsNone(session.take_trace())
            self.assertNotEqual(session.start(lambda control: trace()), first)
            self.assertEqual(session.wait(3)["status"], "completed")
        self.assertIsNone(session.take_trace())
        with self.assertRaises(RuntimeError): session.start(lambda control: trace())

    def test_cancellation_and_exclusive_execution(self):
        started, release = Event(), Event()
        def task(control):
            started.set()
            release.wait(2)
            control.checkpoint("fixture_boundary")
            return trace()
        with ObservationSession() as session:
            session.start(task)
            try:
                self.assertTrue(started.wait(2))
                with self.assertRaises(RuntimeError): session.start(task)
                self.assertTrue(session.cancel())
                self.assertEqual(session.snapshot()["status"], "cancelling")
            finally:
                release.set()
            self.assertEqual(session.wait(3)["status"], "cancelled")
            self.assertIsNone(session.take_trace())
            self.assertFalse(session.cancel())

    def test_failure_recovery_and_invalid_result(self):
        with ObservationSession() as session:
            def fail(_):
                raise ValueError("fixture failure")
            session.start(fail)
            self.assertEqual(session.wait(3)["status"], "failed")
            session.start(lambda control: {"not": "a trace"})
            self.assertEqual(session.wait(3)["status"], "failed")
            session.start(lambda control: trace())
            self.assertEqual(session.wait(3)["status"], "completed")

    def test_retention_limit_and_close_active_run(self):
        from unittest.mock import patch
        with ObservationSession() as session:
            with patch("model_run.trace_store.MAX_TRACE_BYTES", 10):
                session.start(lambda control: trace())
                self.assertEqual(session.wait(3)["status"], "failed")
                self.assertIsNone(session.take_trace())
        started, release = Event(), Event()
        session = ObservationSession()
        def task(control):
            started.set()
            release.wait(2)
            control.check()
            return trace()
        session.start(task)
        try:
            self.assertTrue(started.wait(2))
            session.close()
        finally:
            release.set()
        self.assertEqual(session.wait(3)["status"], "cancelled")
        self.assertTrue(session.snapshot()["closed"])
        self.assertIsNone(session.take_trace())

    def test_partial_trace_and_progress(self):
        with ObservationSession() as session:
            def task(control):
                control.checkpoint("fixture", completed=1, total=2)
                raise ObservationCancelled("fixture cancellation", trace=trace())
            session.start(task)
            result = session.wait(3)
            self.assertEqual(result["status"], "cancelled")
            self.assertEqual(result["progress"]["completed"], 1)
            self.assertEqual(session.take_trace(), trace())

    def test_deadline_at_checkpoint(self):
        from unittest.mock import patch
        with patch("model_run.control.time.monotonic", side_effect=[1.0, 3.0]):
            control = RunControl(timeout_seconds=1)
            with self.assertRaisesRegex(ObservationCancelled, "deadline"):
                control.check()
        with self.assertRaises(ValueError): RunControl(timeout_seconds=-1)

    def test_provider_exit_does_not_leave_session_active(self):
        with ObservationSession() as session:
            def exit_provider(control): raise SystemExit("provider exited")
            session.start(exit_provider)
            self.assertEqual(session.wait(3)["status"], "failed")
            session.start(lambda control: trace())
            self.assertEqual(session.wait(3)["status"], "completed")
