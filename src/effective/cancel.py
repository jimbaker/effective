"""Cancelling an op that is already running, with the cancellation as its recorded result.

A face calls `CancelToken.cancel()` from its own thread, on Esc. Each interpreter doing blocking
work registers what stops it with `on_cancel` (kill the shell's process group, close the model
call's stream), so the blocked call returns in the thread running the op. The interpreter then
answers `Cancelled` with whatever the op produced first.

`Cancelled` is a RESULT, so the op completes and every handler records it: a durable engine
checkpoints it under the reserved `CANCELLED` key and a replay serves it without running the op
again. The workflow never sees the value. The step surface raises `OpCancelled` in its place,
and it is caught in the scope that yielded the step: a scoped body delivers a refusal to its
parent and ends the run on any other exception, `OpCancelled` included.

The token is level-triggered. It stays set until the face calls `reset()`, so every op and poll
after the Esc reads the same answer until the next user turn begins.
"""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

from pydantic import BaseModel

from effective.ops import CompositionRefused, Unretryable

CANCELLED = "__cancelled__"
"""The checkpoint key a cancelled result is stored under, reserved like the substrate's params."""


class Cancelled(BaseModel):
    """Work stopped from outside before it finished: an op an interpreter stopped, with the output
    it produced first, or a spawned task its engine cancelled, as its parent receives it."""

    kind: Literal["cancelled"] = "cancelled"
    partial: str = ""


class OpCancelled(Unretryable):
    """The step named `name` was cancelled while it ran. A retry replays the recorded cancel and
    raises it again, so an uncaught one fails its task on the attempt that raised it."""

    def __init__(self, name: str, partial: str) -> None:
        super().__init__("a step was cancelled while it ran", name)
        self.name = name
        self.partial = partial


class ReservedShape(CompositionRefused):
    """A step's result took the shape reserved for a cancel, which only a `Cancelled` may take."""

    def __init__(self, name: str) -> None:
        super().__init__("a step's result holds the key reserved for a cancel", name)


def served(name: str, raw: Any) -> Any:
    """A step's result as the workflow receives it: a `Cancelled` becomes `OpCancelled`, and a
    result in the reserved shape is refused on every handler."""
    if isinstance(raw, Cancelled):
        raise OpCancelled(name, raw.partial)
    if load_cancelled(raw) is not None:
        raise ReservedShape(name)
    return raw


def dump_cancelled(cancelled: Cancelled) -> dict[str, Any]:
    return {CANCELLED: cancelled.model_dump()}


def load_cancelled(raw: Any) -> Cancelled | None:
    """The `Cancelled` a checkpoint holds, or `None` for any other stored result."""
    match raw:
        case {"__cancelled__": dict(fields)} if len(raw) == 1:
            return Cancelled.model_validate(fields)
        case _:
            return None


def cancelled_or(raw: Any) -> Any:
    """A stored step result as the step answered it: the `Cancelled` it holds, or `raw`."""
    cancelled = load_cancelled(raw)
    return raw if cancelled is None else cancelled


class _Registration:
    """One `on_cancel` block's stop, live while the block runs."""

    def __init__(self, stop: Callable[[], None]) -> None:
        self.stop = stop
        self.live = True
        self.running = False
        self.finished = threading.Event()


class CancelToken:
    """A cancellation flag shared by a face and the interpreters it serves.

    `cancel` runs every live stop in the calling thread, outside the lock, so a stop may block on
    its own cleanup without stalling a registration. A stop runs only while its block does: the
    block's exit waits for a stop already running, and one not yet started never starts. A stop
    that raises leaves the others to run, and `cancel` raises the first failure after them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._registrations: list[_Registration] = []

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            registrations = list(self._registrations)
        failures: list[Exception] = []
        for registration in registrations:
            with self._lock:
                if not registration.live or registration.running:  # each stop runs once
                    continue
                registration.running = True
            try:
                registration.stop()
            except Exception as failure:
                failures.append(failure)
            finally:
                registration.finished.set()
        if failures:
            raise failures[0]

    def reset(self) -> None:
        with self._lock:
            self._cancelled = False

    @contextmanager
    def on_cancel(self, stop: Callable[[], None]) -> Iterator[None]:
        """Run `stop` if the token is cancelled during the block, or at once if it already is."""
        registration = _Registration(stop)
        with self._lock:
            already = self._cancelled
            if not already:
                self._registrations.append(registration)
        if already:
            stop()
        try:
            yield
        finally:
            with self._lock:
                registration.live = False
                running = registration.running
                if registration in self._registrations:
                    self._registrations.remove(registration)
            if running:
                registration.finished.wait()
