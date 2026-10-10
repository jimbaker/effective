"""`RenamedAwaitCtx`: a fork child's EVENT WORLD.

A fork child re-runs the base workflow under a fresh run id, which rescopes only substrate-minted
names; the *workflow-authored* await names (`review:{message_id}`, with no run id) would collide
with the base's on Absurd's queue-global events. `RenamedAwaitCtx` rescopes those to
`fork:{child_run_id};{name}` on `await_event`/`peek_event`, and leaves `step`/`sleep_until` alone
(renaming a step key would break the prefix seeding that makes the tail durable).

These pins cover the renaming CONTRACT in isolation (which names get scoped, which pass through,
delegation, and the `_supports_peek` unwrap). The end-to-end divergence proof (the substitution
lands, the tail re-parks, two sweep forks stay independent) needs real Absurd, where the ctx is
actually driven.
"""

from datetime import datetime

from effective.handlers.durable import RenamedAwaitCtx, _supports_peek
from effective.keys import Key


class _BaseCtx:
    """A minimal TaskContext WITHOUT the optional `peek_event` capability."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.task_id = "t-base"

    def step(self, name, thunk, /):
        self.calls.append(("step", name))
        return thunk()

    def await_event(self, name, /):
        # `.stored()`: this spy stands in for the ENGINE below the ctx stack, which speaks text
        # (cf. `SdkCtx.await_event`). Keeping the `Key` would let the f-string compose its repr.
        self.calls.append(("await", name.stored()))
        return f"answer:{name.stored()}"

    def sleep_until(self, when, /, *, name: Key | None = None) -> None:
        self.calls.append(("sleep", when))


class _PeekCtx(_BaseCtx):
    """A TaskContext that DOES offer `peek_event` (the gather-branch read path)."""

    def peek_event(self, name, /):
        self.calls.append(("peek", name.stored()))
        return (True, f"peeked:{name.stored()}")


def test_await_and_peek_are_rescoped_to_the_child_namespace():
    inner = _PeekCtx()
    ctx = RenamedAwaitCtx(inner, "r-fork-0")

    assert ctx.await_event(Key.parse("review:m1")) == "answer:fork:r-fork-0;review:m1"
    assert ctx.peek_event(Key.parse("review:m1")) == (True, "peeked:fork:r-fork-0;review:m1")
    assert inner.calls == [
        ("await", "fork:r-fork-0;review:m1"),
        ("peek", "fork:r-fork-0;review:m1"),
    ]


def test_step_and_sleep_pass_through_unrenamed():
    """Steps are already child-task-scoped and are the seed keys; renaming them would break
    seeding. A sleep's name is a POSITION, not an event address, so there is nothing for a
    fork's await-renaming to scope — it passes through untouched."""
    inner = _BaseCtx()
    ctx = RenamedAwaitCtx(inner, "r-fork-0")
    when = datetime(2026, 7, 24, 12, 0, 0)

    assert ctx.step(Key.parse("extract"), lambda: "ok") == "ok"
    ctx.sleep_until(when, name=Key.parse("sleep:0"))
    assert inner.calls == [("step", Key.parse("extract")), ("sleep", when)]  # names untouched


def test_getattr_delegates_everything_else():
    inner = _BaseCtx()
    ctx = RenamedAwaitCtx(inner, "r-fork-0")
    assert ctx.task_id == "t-base"


def test_supports_peek_unwraps_the_fork_ctx_to_probe_the_real_ctx():
    """The wrapper defines `peek_event` itself, so `_supports_peek` must see PAST it to the base —
    otherwise a fork over a non-peek ctx would falsely report support."""
    assert _supports_peek(RenamedAwaitCtx(_PeekCtx(), "r-fork-0")) is True
    assert _supports_peek(RenamedAwaitCtx(_BaseCtx(), "r-fork-0")) is False


def test_event_rename_composes_the_whole_chain_not_one_wrapper():
    """A fork inside a fork stacks TWO renames, and the guard must name what the engine sees.

    Exactly `_PrefixedCtx.prefix`'s F3 defect, reproduced in `event_rename` — by a commit whose
    own message cited that fix as precedent. The first version returned this wrapper's segment
    (`fork:c2;`) while the engine awaited `fork:c1;fork:c2;{name}`, so the refusal named the
    wrong half of the mismatch. Liveness was unaffected (non-empty is non-empty); the diagnostic
    was not, and the nested stack is reachable by following the refusal's own advice."""

    class _Base:
        def __init__(self) -> None:
            self.awaited: list[str] = []

        def step(self, name, thunk, /):
            return thunk()

        def await_event(self, name, /):
            self.awaited.append(name.stored())  # the ENGINE sees the wire name
            return None

        def sleep_until(self, when, /, *, name: Key | None = None):
            return None

    base = _Base()
    nested = RenamedAwaitCtx(RenamedAwaitCtx(base, "c1"), "c2")

    nested.await_event(Key.parse("review:m1"))
    assert base.awaited == ["fork:c1;fork:c2;review:m1"]
    # The property must report what the ENGINE saw, not this wrapper's own segment.
    assert nested.event_rename == "fork:c1;fork:c2;"
    assert base.awaited[0] == f"{nested.event_rename}review:m1"
