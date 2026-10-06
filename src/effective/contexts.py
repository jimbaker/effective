"""In-process task contexts: `DurableHandler` without an engine.

A context is what `DurableHandler` checkpoints through. The engines (`effective.sqlite`, Absurd)
persist each step; these keep it in memory, for runs that need no file and no service:

| context | a step | for |
|---|---|---|
| `LocalCtx` | runs its thunk, keeps nothing | a step-only run in process |
| `RecordingCtx` | runs its thunk, logs the result by name | the record pass |
| `ReplayCtx` | returns the logged result; the thunk never runs | the replay pass: no I/O |
| `ResumeCtx` | returns a logged result, else runs and logs | resuming after a crash |

`await_event` raises on all four, so a workflow that waits needs an engine. `sleep_until` raises
on `LocalCtx` and returns at once on the other three, so an in-process run does not wait out a
timer.
"""

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from effective.keys import Key

_CANNOT_SUSPEND = "an in-process context cannot suspend; a workflow that waits needs an engine"


class LocalCtx:
    """Runs each step's thunk inline and checkpoints nothing."""

    def step(self, name: Key, thunk: Callable[[], Any]) -> Any:
        return thunk()

    def await_event(self, name: Key) -> Any:
        raise NotImplementedError(f"LocalCtx: {_CANNOT_SUSPEND}")

    def sleep_until(self, when: datetime, /, *, name: Key) -> None:
        raise NotImplementedError(f"LocalCtx: {_CANNOT_SUSPEND}")


class RecordingCtx:
    """Runs each step for real and logs its result by name."""

    def __init__(self) -> None:
        self.log: dict[Key, Any] = {}

    def step(self, name: Key, thunk: Callable[[], Any]) -> Any:
        result = thunk()
        self.log[name] = result
        return result

    def await_event(self, name: Key) -> Any:
        raise NotImplementedError(f"RecordingCtx: {_CANNOT_SUSPEND}")

    def sleep_until(self, when: Any, /, *, name: Any = None) -> None:
        return None


class ReplayCtx:
    """Replays a `RecordingCtx.log`: a step returns its logged result, and its thunk never runs.

    The thunk is what would call the model or a tool, so replay performs no I/O. A step whose name
    is absent from the log is a control-flow divergence, and raises `KeyError`."""

    def __init__(self, log: Mapping[Key, Any]) -> None:
        self.log = dict(log)

    def step(self, name: Key, thunk: Callable[[], Any]) -> Any:
        return self.log[name]

    def await_event(self, name: Key) -> Any:
        raise NotImplementedError(f"ReplayCtx: {_CANNOT_SUSPEND}")

    def sleep_until(self, when: Any, /, *, name: Any = None) -> None:
        return None


class ResumeCtx:
    """Resumes after a crash: a logged step replays, and a step missing from the log runs and is
    logged. `ran` lists what ran again, so a caller can tell a resume from a restart."""

    def __init__(self, log: Mapping[Key, Any]) -> None:
        self.log = dict(log)
        self.ran: list[Key] = []

    def step(self, name: Key, thunk: Callable[[], Any]) -> Any:
        if name in self.log:
            return self.log[name]
        result = thunk()
        self.log[name] = result
        self.ran.append(name)
        return result

    def await_event(self, name: Key) -> Any:
        raise NotImplementedError(f"ResumeCtx: {_CANNOT_SUSPEND}")

    def sleep_until(self, when: Any, /, *, name: Any = None) -> None:
        return None
