"""One active observation per session, with explicit lifecycle and bounded trace retention."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock
from uuid import uuid4

from .control import ObservationCancelled, RunControl
from .trace_store import decode_trace, encode_trace


class ObservationSession:
    """Runs caller-supplied observation functions returning a trace, never model outputs.

    Sessions do not own model/device allocation and cancellation is cooperative.
    Closing signals cancellation; it cannot forcibly stop a blocked model call.
    """
    def __init__(self):
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-run")
        self._closed = False
        self._future = None
        self._control = None
        self._trace = None
        self._state = {"run_id": None, "status": "idle", "progress": None, "error": None}

    def snapshot(self):
        with self._lock:
            return {**deepcopy(self._state), "closed": self._closed}

    def start(self, operation, *, timeout_seconds=None):
        if not callable(operation):
            raise ValueError("operation must be callable and return a trace")
        with self._lock:
            if self._closed:
                raise RuntimeError("Session is closed")
            if self._state["status"] in ("pending", "running", "cancelling"):
                raise RuntimeError("Session already has an active observation")
            if self._trace is not None:
                raise RuntimeError("Take or discard the previous trace before starting another run")
            control = RunControl(timeout_seconds=timeout_seconds, on_progress=self._progress)
            self._control = control
            run_id = uuid4().hex
            self._state = {"run_id": run_id, "status": "pending", "progress": None, "error": None}
            self._future = self._executor.submit(self._execute, operation, control)
            return run_id

    def _progress(self, event):
        with self._lock:
            self._state["progress"] = deepcopy(event)

    @staticmethod
    def _bounded_trace(trace):
        return decode_trace(encode_trace(trace))

    def _execute(self, operation, control):
        trace, error, status = None, None, "failed"
        try:
            control.check()
            with self._lock:
                if self._state["status"] != "cancelling":
                    self._state["status"] = "running"
            result = operation(control)
            control.check()
            trace = self._bounded_trace(result)
            control.check()
            status = "completed"
        except ObservationCancelled as exc:
            status, error = "cancelled", exc.reason
            if exc.trace is not None:
                try:
                    trace = self._bounded_trace(exc.trace)
                except (TypeError, ValueError, RecursionError):
                    trace = None
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"[:2048]
            partial = getattr(exc, "trace", None)
            if partial is not None:
                try:
                    trace = self._bounded_trace(partial)
                except (TypeError, ValueError, RecursionError):
                    trace = None
        with self._lock:
            # A cancellation accepted before publication wins over completion.
            if status == "completed" and self._state["status"] == "cancelling":
                status, error, trace = "cancelled", "cancelled by caller", None
            self._trace = None if self._closed else trace
            self._state.update(status=status, error=error)

    def cancel(self):
        with self._lock:
            if self._state["status"] not in ("pending", "running", "cancelling"):
                return False
            self._state["status"] = "cancelling"
            self._control.cancel()
            return True

    def wait(self, timeout=None):
        with self._lock:
            future = self._future
        if future is not None:
            future.result(timeout=timeout)
        return self.snapshot()

    def take_trace(self):
        with self._lock:
            if self._state["status"] in ("pending", "running", "cancelling"):
                raise RuntimeError("Observation is still active")
            trace, self._trace = self._trace, None
            return trace

    def close(self, *, wait=False):
        with self._lock:
            self._closed = True
            self._trace = None
        self.cancel()
        self._executor.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close(wait=True)
