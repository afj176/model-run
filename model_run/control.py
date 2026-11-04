"""Cooperative observation control. Does not interrupt executing accelerator kernels."""
import math
import time
from threading import Event


class ObservationCancelled(RuntimeError):
    def __init__(self, reason="cancelled", trace=None):
        super().__init__(reason)
        self.reason = reason
        self.trace = trace


class RunControl:
    def __init__(self, *, timeout_seconds=None, on_progress=None):
        if timeout_seconds is not None and (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be positive and finite")
        if on_progress is not None and not callable(on_progress):
            raise ValueError("on_progress must be callable")
        self._cancel = Event()
        self._deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        self._on_progress = on_progress

    def cancel(self):
        self._cancel.set()

    def check(self):
        if self._cancel.is_set():
            raise ObservationCancelled("cancelled by caller")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise ObservationCancelled("observation deadline exceeded")

    def checkpoint(self, stage, *, completed=None, total=None):
        self.check()
        if self._on_progress is not None:
            self._on_progress({"stage": stage, "completed": completed, "total": total})
        self.check()
