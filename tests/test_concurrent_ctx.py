"""Guards for the concurrent durable-gather ctx wrappers.

Infra-free: no Postgres, just the wrapper contracts. Two things the concurrency story
silently depends on: the Absurd SDK's begin/complete split, and `_PrefixedCtx` propagating
the backend's `concurrent_safe` capability one level down (so nested gather stays parallel).
"""

from datetime import datetime
from typing import Any

from effective.handlers.absurd import ConcurrentAbsurdCtx, _PrefixedCtx
from effective.keys import Key


def test_absurd_sdk_still_exposes_the_begin_complete_split():
    """ConcurrentAbsurdCtx runs the tool lock-free between the SDK's *public*
    begin_step (under lock) and complete_step (under lock). If a future vendored SDK folded
    that split back into a single step(), the lock window would silently vanish, so pin the
    seam the wrapper rides."""
    from absurd_sdk import TaskContext as SdkTaskContext

    assert hasattr(SdkTaskContext, "begin_step")
    assert hasattr(SdkTaskContext, "complete_step")
    assert hasattr(SdkTaskContext, "step")


class _FakeConcurrentCtx:
    """A minimal concurrent-safe ctx (the capability the SQLite engine / ConcurrentAbsurdCtx
    advertise)."""

    concurrent_safe = True

    def step(self, name: Key, thunk, /) -> Any:
        return thunk()

    def await_event(self, name: Key, /) -> Any:
        raise NotImplementedError

    def sleep_until(self, when: datetime, /, *, name: Key | None = None) -> None:
        return None


def test_prefixed_ctx_propagates_concurrent_safe():
    """A gather branch runs over a _PrefixedCtx; a *nested* gather inside it must still see the
    backend's concurrency capability, or it would silently serialize. The __getattr__
    delegation is what carries `concurrent_safe` through the prefix wrapper."""
    wrapped = _PrefixedCtx(_FakeConcurrentCtx(), "gather:0,0;")
    assert wrapped.concurrent_safe is True


def test_prefixed_ctx_namespaces_step_keys():
    """The prefix decorates the durable key (and only the key)."""
    seen: list[Key] = []

    class _Recorder:
        def step(self, name, thunk, /):
            seen.append(name)
            return thunk()

        def await_event(self, name, /):
            raise NotImplementedError

        def sleep_until(self, when, /, *, name: Key | None = None):
            return None

    ctx = _PrefixedCtx(_Recorder(), "gather:2,1;")
    ctx.step(Key.parse("tool:a"), lambda: 7)
    assert [k.stored() for k in seen] == ["gather:2,1;tool:a"]


def test_concurrent_absurd_ctx_advertises_concurrent_safe():
    class _SdkLike:
        def begin_step(self, name):
            return None

    assert ConcurrentAbsurdCtx(_SdkLike()).concurrent_safe is True
