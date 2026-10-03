"""An asyncio loop on a thread of its own, where a synchronous interpreter runs its coroutines.

A handler calls an interpreter synchronously, so an interpreter written with asyncio hands its
coroutine to the process's `shared_loop()` and blocks for the result. The workflow and the handler
never see the loop: the color stays in the interpreter. A `CancelToken` cancels the coroutine's
task, so the `CancelledError` lands at whatever it awaits, connecting and reading included, and the
caller returns at once with `Stopped` while the task unwinds on the loop.

An async client binds to the first loop that uses it, which is why there is one loop per process:
every async interpreter's client is used on it and no other.
"""

import asyncio
import concurrent.futures
import inspect
import os
import threading
from collections.abc import Coroutine
from contextlib import nullcontext
from typing import Any

from effective.cancel import CancelToken


class Stopped(Exception):
    """The token was cancelled before the coroutine finished."""


def _retrieve(task: asyncio.Task[Any]) -> None:
    """Read a finished task's exception, so one that ends after its caller stopped listening is
    not logged as never retrieved."""
    if not task.cancelled():
        task.exception()


async def _retrieved[T](coroutine: Coroutine[Any, Any, T]) -> T:
    current = asyncio.current_task()
    if current is not None:
        current.add_done_callback(_retrieve)
    return await coroutine


def _close_if_unstarted(coroutine: Coroutine[Any, Any, Any]) -> None:
    """Close a coroutine its cancelled task never started. It runs on the loop after the task's
    cancel, so a coroutine still unstarted here never starts."""
    if inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED:
        coroutine.close()


class LoopThread:
    """One event loop, run forever on a daemon thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._runner = threading.Thread(target=self.loop.run_forever, name="interpreter-loop")
        self._runner.daemon = True
        self._runner.start()

    def run[T](self, coroutine: Coroutine[Any, Any, T], cancel: CancelToken | None = None) -> T:
        """The coroutine's result, or `Stopped` once `cancel` fires. A cancel before the task
        starts means the coroutine never runs. Called from the loop's own thread, it would wait on
        itself forever, so it refuses."""
        if threading.current_thread() is self._runner:
            coroutine.close()
            raise RuntimeError("an interpreter's loop cannot wait on itself")
        if cancel is not None and cancel.cancelled:
            coroutine.close()
            raise Stopped
        future = asyncio.run_coroutine_threadsafe(_retrieved(coroutine), self.loop)
        stopped = threading.Event()

        def stop() -> None:
            stopped.set()
            future.cancel()  # the loop cancels the task, and `result` raises at once
            self.loop.call_soon_threadsafe(_close_if_unstarted, coroutine)

        with cancel.on_cancel(stop) if cancel is not None else nullcontext():
            try:
                return future.result()
            except concurrent.futures.CancelledError:
                if stopped.is_set():  # its own stop, which a reset of the token cannot undo
                    raise Stopped from None
                raise


_shared: LoopThread | None = None
_shared_lock = threading.Lock()


def shared_loop() -> LoopThread:
    """The process's one `LoopThread`, started on first use."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = LoopThread()
        return _shared


def _forget_after_fork() -> None:
    """A forked child inherits the loop but not its thread, so it starts its own on first use."""
    global _shared, _shared_lock
    _shared, _shared_lock = None, threading.Lock()


os.register_at_fork(after_in_child=_forget_after_fork)
