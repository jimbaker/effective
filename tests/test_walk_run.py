"""`walk_run`: the per-run ambient, in one place instead of five.

The hazard is MEASURED, not hypothetical: a replay walk that establishes neither half reads a
`layer_run_state` cell incremented across two ops as `[1, 1]` where recording read `[1, 2]`. The
workflow returns a different value on replay than it recorded, the one thing replay exists to
prevent.

The property is that **every walk must agree**, and five separate call sites are five chances to
disagree. So the test that matters is the parametrized one: drive the same probe through every
interpreter and assert they read the same.
"""

import pytest

from effective.api import gather, step
from effective.domain import CallTool
from effective.handlers.base import walk_run
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.keys import Segment, compose_key
from effective.layers import layer_run_state
from effective.ops import CHAIN_DEPTH, CHAIN_GENERATION

CELL = compose_key(t"probe:{Segment('cell')}")


def _bump() -> int:
    state = layer_run_state(CELL)
    state["n"] = state.get("n", 0) + 1
    return state["n"]


def _counting(seen: list[int]):
    """The D1 probe, verbatim from `ReplayHandler.run`'s docstring."""

    def program():
        seen.append(_bump())
        yield from step("a", CallTool(name="ta", args={}, result_schema=object))
        seen.append(_bump())
        yield from step("b", CallTool(name="tb", args={}, result_schema=object))
        return None

    return program


def test_the_run_scope_half_accumulates_across_ops():
    """Reddens if `walk_run` drops `run_scope()` — the measured `[1,2]` vs `[1,1]` defect."""
    seen: list[int] = []
    with walk_run():
        _bump()
        _bump()
        seen = [layer_run_state(CELL)["n"]]
    assert seen == [2]


def test_the_task_run_half_resets_chain_state():
    """Reddens if `walk_run` drops `enter_task_run()`.

    The two halves fail differently and a helper carrying both must be pinned on both: this one
    is per-run CONTROL state, which a value left set by an earlier run in the same context would
    corrupt — `ops.CHAIN_GENERATION` is read when composing a `descend` grant name.

    Both cells, and they reset to DIFFERENT values (`0` vs `None`), which is exactly the sort of
    thing a test asserting only one of them would let drift."""
    CHAIN_DEPTH.set(9)
    CHAIN_GENERATION.set(7)
    with walk_run():
        assert CHAIN_DEPTH.get() == 0
        assert CHAIN_GENERATION.get() is None


def test_the_ambient_does_not_survive_the_block():
    """Reddens if the scope leaks — its lifetime is one attempt by construction, which is what
    keeps a layer's cross-op state from outliving its task and authorizing from process memory."""
    with walk_run():
        _bump()
    with walk_run():
        assert layer_run_state(CELL).get("n") is None


@pytest.mark.parametrize("walk", ["recording", "replay"])
def test_every_walk_reads_the_same_accumulated_value(walk):
    """Reddens if ANY interpreter's entry stops establishing the ambient.

    This is the shape that matters. The hazard is "five entries can disagree" rather than
    "replay is wrong", and only a cross-walk assertion can see that. `tests/_walks.py` is the
    fuller version of this idea over all six walks."""
    recorded: list[int] = []
    handler = RecordingHandler({"a": 1, "b": 2})
    handler.run(_counting(recorded))
    assert recorded == [1, 2]

    if walk == "replay":
        replayed: list[int] = []
        ReplayHandler(handler.trace).run(_counting(replayed))
        assert replayed == recorded


def test_a_gather_branch_shares_the_run_but_gets_its_own_scope():
    """Reddens if a branch re-enters the RUN, or fails to enter its own SCOPE.

    The per-RUN/per-FRAME line: a branch is one run's structure, not a task boundary, so it takes
    `run_scope()` alone and never `walk_run()`.

    **BOTH halves are asserted.** Reading `layer_run_state` (the scope half) alone does not
    check the RUN half: branches that call `walk_run()` unconditionally hand a parent at
    `(1, 3)` a branch at `(0, None)`, while the durable handler passes `(1, 3)` through."""
    seen: list[int] = []
    chain: list[tuple[int | None, int | None]] = []

    def branch():
        seen.append(_bump())
        chain.append((CHAIN_DEPTH.get(), CHAIN_GENERATION.get()))
        yield from step("leaf", CallTool(name="tl", args={}, result_schema=object))
        return None

    def wf():
        CHAIN_DEPTH.set(1)
        CHAIN_GENERATION.set(3)
        yield from gather([branch, branch])
        return None

    RecordingHandler({"gather:0,0;leaf": 1, "gather:0,1;leaf": 2}).run(wf)
    assert seen == [1, 1], "each branch counts in its OWN scope, so neither sees the other's 1"
    assert chain == [(1, 3), (1, 3)], "a branch INHERITS the run's chain state; it must not reset"
