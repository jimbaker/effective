"""`ReplayHandler`'s generation boundary.

`ReplayHandler` has no `src/` caller; every construction of it is in `tests/`. A handler nothing
in the substrate drives is a handler whose arms nothing forces complete, so these tests drive a
respawning workflow through replay, as both siblings handle `Respawn`
(`RecordingHandler._respawn`, the durable engine's `_ChainContinues`).

One hazard per test:

1. **The error path reports instead of crashing.** `placed_key` raises for an op whose identity
   is positional and unplaced; called inside the `ReplayMismatch` f-string, it turns an
   unhandled op into `TypeError: unknown op: Respawn(...)` rather than a mismatch report.
2. **A `Respawn` arm.** The recorder writes no trace entry for a generation boundary (the
   generator is abandoned exactly as the durable engine abandons it), so the boundary always
   arrives at `self._i == len(self.recorded)`, where it would otherwise take the "extra op" path.
"""

import pytest

from effective.api import gather, scoped, step
from effective.domain import CallTool
from effective.handlers.recording import RecordingHandler, Respawned
from effective.handlers.replay import ReplayHandler, ReplayMismatch, _named
from effective.keys import Segment, compose_key
from effective.ops import CompositionRefused, Respawn, leaves


def _respawning():
    """A workflow that does one step, then crosses a generation boundary."""
    yield from step("work", CallTool(name="tw", args={}, result_schema=object))
    yield Respawn(task="t", generation=1, state=None, params={}, run_id="r1")
    raise AssertionError("a respawn ends the run; nothing after it may execute")


def test_replay_ends_at_a_generation_boundary_like_the_recorder_does():
    """Reddens if the `Respawn` arm is removed — it raised `TypeError`, not `ReplayMismatch`.

    The two interpreters must agree on the boundary: the recorder returns `Respawned` and ends,
    so replay must too. Anything else and a respawning workflow is simply un-replayable."""
    recorded = RecordingHandler({"work": "did-it"})
    out = recorded.run(_respawning)
    assert isinstance(out, Respawned)

    replayed = ReplayHandler(recorded.trace).run(_respawning)
    assert isinstance(replayed, Respawned)
    assert (replayed.generation, replayed.run_id) == (out.generation, out.run_id)


def test_the_recorder_writes_no_trace_entry_for_the_boundary():
    """Reddens if a `Respawn` ever gains a trace entry.

    This is WHY the missing arm crashed rather than mismatching: with no entry, the boundary
    always lands at `_i == len(recorded)`, i.e. on the "extra op" path. If this changes, the
    arm's placement has to change with it."""
    recorded = RecordingHandler({"work": "did-it"})
    recorded.run(_respawning)
    assert [e.op.__class__.__name__ for e in recorded.trace] == ["Step"]


def test_a_diagnostic_never_raises_on_an_op_with_no_placed_key():
    """Reddens if `_named` goes back to calling `placed_key` unguarded.

    An error path that can fail is an error path that hides the error it exists to name. Any op
    whose identity is positional and unplaced would do this; `Respawn` is the one that did."""
    op = Respawn(task="t", generation=1, state=None, params={}, run_id="r1")
    assert _named(op, "") == "<Respawn, no placed key>"


def test_an_extra_op_still_reports_a_mismatch_and_not_a_crash():
    """Reddens if the guard swallows the real diagnostic. The message must still name the op."""

    def longer():
        yield from step("work", CallTool(name="tw", args={}, result_schema=object))
        yield from step("extra", CallTool(name="tx", args={}, result_schema=object))

    def shorter():
        yield from step("work", CallTool(name="tw", args={}, result_schema=object))

    recorded = RecordingHandler({"work": "did-it"})
    recorded.run(shorter)
    with pytest.raises(ReplayMismatch, match="extra op"):
        ReplayHandler(recorded.trace).run(longer)


# --- the composition law's missing REPLAY leg ------------------------------------------------
#
# `ops.py`'s terminal-law docstring and `test_composition.py` both ratify `scoped ∘ respawn` as
# LEGAL — "an ordinary namespacing; the task ends and both interpreters agree via `Ended`". The
# law was pinned on `RecordingHandler` ONLY, so when the first `Respawn` arm here tested
# `if prefix:` it refused a legal composition and the suite stayed green. These are the leg.

SCOPE = compose_key(t"s:{Segment('one')}")


def _scoped_respawn():
    def inner():
        yield from step("inner", CallTool(name="ti", args={}, result_schema=object))
        yield Respawn(task="t", generation=1, state=None, params={}, run_id="r1")

    return (yield from scoped(SCOPE, inner))


def test_scoped_respawn_replays_to_Respawned_like_the_recorder():
    """Reddens on `if prefix:` — the predicate that conflates a scope frame with a branch.

    A `Scoped` body carries a non-empty prefix and is no kind of branch, so `if prefix:` refused
    this and blamed a "gather branch" that was a scope. The flag has to be structural."""
    recorded = RecordingHandler({"s:one;inner": "v"})
    out = recorded.run(_scoped_respawn)
    assert isinstance(out, Respawned)
    assert isinstance(ReplayHandler(recorded.trace).run(_scoped_respawn), Respawned)


def test_a_Respawned_escapes_a_scope_instead_of_becoming_its_value():
    """Reddens if the `Scoped` arm sends `Respawned` into the parent.

    Fixing the predicate alone is not enough: the boundary must travel OUT. Otherwise the
    workflow resumes past the scope holding a `Respawned`, which is the bug
    `RecordingHandler._run_scoped` documents and closed — here it surfaced as a `ReplayMismatch`
    on the op after the scope."""

    def after_the_scope():
        yield from scoped(SCOPE, _inner_respawn)
        yield from step("after", CallTool(name="ta", args={}, result_schema=object))
        raise AssertionError("the generation boundary ends the run; 'after' must never execute")

    def _inner_respawn():
        yield from step("inner", CallTool(name="ti", args={}, result_schema=object))
        yield Respawn(task="t", generation=1, state=None, params={}, run_id="r1")

    recorded = RecordingHandler({"s:one;inner": "v"})
    assert isinstance(recorded.run(after_the_scope), Respawned)
    assert isinstance(ReplayHandler(recorded.trace).run(after_the_scope), Respawned)


def test_a_respawn_in_a_gather_branch_is_still_refused_on_replay():
    """Reddens if the structural flag stops being set on the gather recursion.

    The refusal is the half that must SURVIVE the predicate fix — `in_gather` has to be sticky
    through a `Scoped`, because a scope inside a branch is still inside the branch.

    Both interpreters refuse and name the branch. The refusal is asserted as the error's one leaf,
    since a gather reports its branches' errors as a group."""

    def branch_respawn():
        yield Respawn(task="t", generation=1, state=None, params={}, run_id="r1")

    def wf():
        yield from gather([branch_respawn])

    with pytest.raises(BaseExceptionGroup) as recorded:
        RecordingHandler({}).run(wf)
    inner = recorded.value.exceptions[0]
    assert isinstance(inner, CompositionRefused)
    assert "gather branch" in str(inner)

    with pytest.raises((CompositionRefused, BaseExceptionGroup)) as replayed:
        ReplayHandler([]).run(wf)
    (leaf,) = leaves(replayed.value)
    assert isinstance(leaf, CompositionRefused)
    assert "gather branch" in str(leaf)


def test_the_table_refuses_an_op_it_has_no_arm_for():
    """Reddens if the wildcard is deleted or turned back into a silent fall-through.

    The whole reason this dispatch is a `match` is that the `isinstance` chain let `Respawn`
    fall through to the leaf path. A tenth arm of `WorkflowOp` must be loud and named.

    NOT `ReplayMismatch`, and the distinction is the point: a mismatch means the workflow
    diverged from its trace, which is an author's problem. An unhandled arm is a SUBSTRATE gap:
    the walk owes an answer it does not have.

    `AssertionError`, from `assert_never`: the one spelling every closed dispatch in the tree
    uses.

    **The cost is pinned here rather than described, because it is sharper than expected.**
    `assert_never` reports the value's REPR, so for a real op, a dataclass, the message is
    informative. For a value with no `__repr__`, which is what a tenth arm would look like on the
    day someone adds one, it degrades to `<... object at 0x...>`: a memory address, with neither
    the walk nor the arm it owes. The assertion below therefore
    checks the type NAME is recoverable at all, since that is the part a reader needs and the
    part this spelling does not guarantee."""

    class Tenth:
        pass

    def wf():
        yield Tenth()

    with pytest.raises(AssertionError, match="unreachable, but got") as refusal:
        ReplayHandler([]).run(wf)
    assert "Tenth" in str(refusal.value), (
        "the arm's type name must survive into the message; only the qualname carries it here"
    )
