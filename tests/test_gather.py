"""The applicative combinator `gather` on the recording/replay core.

`gather` is an effect (`yield from gather([b0, b1, ...])`), not a host-language
helper. RecordingHandler runs the branches under structured concurrency
(asyncio.TaskGroup), joins them in **branch order** (never completion order), and
records each branch's LEAF ops keyed by their `gather:{g},{i};` path (the gather itself
is pure structure, with no entry of its own). ReplayHandler
re-runs the branch generators sequentially, feeding each leaf its recorded result and
reconstructing the aggregated join: the same re-execution model the durable handler uses.
These tests pin those semantics.
"""

import time
from collections.abc import Callable
from typing import Any

import pytest

from effective.api import append_ledger, await_event, call_tool, gather
from effective.handlers.absurd import DurableHandler
from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayHandler
from effective.keys import Key
from effective.ops import LedgerRow, leaves

RESPONSES = {"tool:b0": 100, "tool:b1": 200, "tool:b2": 300}


class _SeqCtx:
    """A non-durable, non-concurrent ctx: a step runs its thunk inline. No `concurrent_safe`
    attribute, so the durable handler runs gather branches SEQUENTIALLY (L5's sequential mode)."""

    def step(self, name: Key, thunk: Callable[[], Any]) -> Any:
        return thunk()

    def await_event(self, name: Key) -> Any:
        raise NotImplementedError

    def sleep_until(self, when: Any, *, name: Key | None = None) -> None:
        raise NotImplementedError


class _RaiseDomain:
    def run(self, op: Any) -> Any:
        if getattr(op, "name", None) == "boom":
            raise ValueError("branch boom")
        return "ok"


def _branch(name: str, delay: float, counter: list[str], done: list[str] | None = None):
    """A thunk-returning factory: sleeps `delay` (in its own thread), then one tool op.

    `counter` records that the body STARTED; `done`, when a caller passes one, records that it
    FINISHED. Only the concurrency test needs the second, and it needs it because start order
    cannot tell overlap from serialization — both run b0 first."""

    def thunk():
        counter.append(name)  # records that this branch BODY actually ran
        time.sleep(delay)
        if done is not None:
            done.append(name)
        v = yield from call_tool(name, {}, int)
        return (name, v)

    return thunk


def test_gather_joins_results_in_branch_order_not_completion_order():
    """The join is by branch index, and the branches really did run concurrently.

    Concurrency is asserted by COMPLETION ORDER, not by elapsed time. The slow branch is
    started first, so under structured concurrency the fast one finishes first and under
    serialization it cannot. The old form allowed 10ms over the summed time and measured the
    machine instead: it failed at 0.1632s in a loaded gate run and passed three times alone."""
    counter: list[str] = []
    done: list[str] = []

    def parent():
        # b0 is slow and started first, so finishing b1 first is only possible concurrently.
        return (
            yield from gather(
                [_branch("b0", 0.15, counter, done), _branch("b1", 0.01, counter, done)]
            )
        )

    result = RecordingHandler(responses=RESPONSES).run(parent)

    assert result == [("b0", 100), ("b1", 200)]  # branch order, despite b1 finishing first
    assert counter == ["b0", "b1"]  # both bodies ran, slow one first
    assert done == ["b1", "b0"], f"branches did not overlap (serialized?): {done}"


def test_gather_records_leaves_by_path_and_replays_by_reexecution():
    counter: list[str] = []

    def parent():
        out = yield from gather([_branch("b0", 0.0, counter), _branch("b1", 0.0, counter)])
        return {"branches": out}

    handler = RecordingHandler(responses=RESPONSES)
    recorded = handler.run(parent)
    assert recorded == {"branches": [("b0", 100), ("b1", 200)]}
    assert counter == ["b0", "b1"]  # both branch bodies ran once during recording
    # 1b: the gather is pure structure — the trace holds the branch LEAVES, each keyed by
    # its full path `gather:{g},{i};leaf`, the same string the durable checkpoint uses. No
    # gather-node entry.
    assert [e.key.stored() for e in handler.trace] == [
        "gather:0,0;step;tool:b0",
        "gather:0,1;step;tool:b1",
    ]

    # Replay RE-RUNS the branch generators (sequentially, no concurrency), feeding each leaf
    # its recorded result and reconstructing the aggregated join — the durable re-execution
    # model, uniform with how replay already re-runs the top-level workflow.
    replayed = ReplayHandler(handler.trace).run(parent)
    assert replayed == recorded
    assert counter == ["b0", "b1", "b0", "b1"]  # branches re-ran on replay (leaves fed from trace)


def test_gather_ledger_merges_in_branch_order():
    from effective.api import append_ledger

    def _led_branch(name: str, eid: str):
        def thunk():
            yield from append_ledger(LedgerRow(event_id=Key.parse(eid), kind=name))
            return name

        return thunk

    def parent():
        return (yield from gather([_led_branch("a", "e-a"), _led_branch("b", "e-b")]))

    handler = RecordingHandler()
    result = handler.run(parent)
    assert result == ["a", "b"]
    # the canonical record is ordered by branch index, independent of completion timing:
    assert [row.event_id.stored() for row in handler.ledger] == ["e-a", "e-b"]


def test_gather_branch_failure_cancels_siblings_as_exception_group():
    def boom():
        def thunk():
            raise ValueError("branch boom")
            yield  # make `thunk` a generator function

        return thunk

    counter: list[str] = []

    def parent():
        return (yield from gather([_branch("b0", 0.0, counter), boom()]))

    # The structured-concurrency scope surfaces a branch failure as an ExceptionGroup
    # (and cancels siblings); the ValueError is aggregated inside it.
    with pytest.raises(BaseExceptionGroup) as exc_info:
        RecordingHandler(responses=RESPONSES).run(parent)
    matched, _ = exc_info.value.split(ValueError)
    assert matched is not None


def _await_branch(name: str):
    def thunk():
        payload = yield from await_event(name, dict)
        return payload

    return thunk


def test_gather_branch_await_parks_with_the_qualified_name():
    """V1 (recorder): a branch's un-canned await parks the gather as a
    ``Suspended`` whose ``awaiting`` is the branch-QUALIFIED name; resuming
    delivers into the live branch generator and joins in branch order. Legal
    in-process only — the recorder holds live generators (no crash-durability
    claim); the durable path replays instead."""
    counter: list[str] = []

    def parent():
        return (yield from gather([_branch("b0", 0.0, counter), _await_branch("ev")]))

    parked = RecordingHandler(responses=RESPONSES).run(parent)
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == "gather:0,1;ev"
    assert counter == ["b0"]  # the sibling's body ran in the round
    result = parked.resume({"ok": True})
    assert result == [("b0", 100), {"ok": True}]


def test_gather_two_parked_branches_resume_serialized_lowest_first():
    """V1 serialized wakes, in-memory: with branches 0 and 2 parked, ``awaiting``
    names branch 0's event; its resume re-parks on branch 2's; the second
    resume completes the join."""

    def parent():
        return (
            yield from gather([_await_branch("ev0"), _branch("b1", 0.0, []), _await_branch("ev2")])
        )

    parked = RecordingHandler(responses=RESPONSES).run(parent)
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == "gather:0,0;ev0"
    parked = parked.resume({"n": 0})
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == "gather:0,2;ev2"
    assert parked.resume({"n": 2}) == [{"n": 0}, ("b1", 200), {"n": 2}]


def test_L2_resume_name_guard_rejects_a_mismatched_event(recording_handler_await):
    # L2: the recorder binds by POSITION (it wakes THIS slot regardless of name), so delivering
    # the wrong event's payload would silently swap it — the engines bind by name. The optional
    # `name=` guard makes that swap a loud error instead. Correct name (and no name) still work.
    parked = recording_handler_await
    assert parked.awaiting.stored() == "gather:0,1;ev"
    with pytest.raises(ValueError, match="not 'gather:0,1;other'"):
        parked.resume({"ok": True}, name=Key.parse("gather:0,1;other"))
    # the guard did not consume the park; the correctly-named resume still completes
    assert parked.resume({"ok": True}, name=Key.parse("gather:0,1;ev")) == [
        ("b0", 100),
        {"ok": True},
    ]


def test_L2_serialized_gather_resume_name_guard_names_the_lowest_branch():
    # L2 for GatherSuspended: with branches 0 and 2 parked, `awaiting` is branch 0's event; the
    # guard rejects delivering under branch 2's name (which would positionally wake branch 0 and
    # bind ev2's payload to ev0 — P1's silent swap).
    def parent():
        return (
            yield from gather([_await_branch("ev0"), _branch("b1", 0.0, []), _await_branch("ev2")])
        )

    parked = RecordingHandler(responses=RESPONSES).run(parent)
    assert parked.awaiting.stored() == "gather:0,0;ev0"
    with pytest.raises(ValueError, match="not 'gather:0,2;ev2'"):
        parked.resume({"who": "ev2"}, name=Key.parse("gather:0,2;ev2"))  # would swap onto branch 0
    parked = parked.resume(
        {"who": "ev0"}, name=Key.parse("gather:0,0;ev0")
    )  # correct, lowest-first
    assert parked.awaiting.stored() == "gather:0,2;ev2"
    assert parked.resume({"who": "ev2"}, name=Key.parse("gather:0,2;ev2")) == [
        {"who": "ev0"},
        ("b1", 200),
        {"who": "ev2"},
    ]


def _boom_branch():
    def thunk():
        yield from call_tool("boom", {}, str)  # _RaiseDomain raises on this tool

    return thunk


def _ok_branch():
    def thunk():
        return (yield from call_tool("fine", {}, str))

    return thunk


def test_L5_concurrent_gather_branch_failure_is_an_exceptiongroup_uncatchable():
    # L5, concurrent mode (RecordingHandler's TaskGroup): a branch failure surfaces as an
    # ExceptionGroup AND is NOT catchable by a try/except around `yield from gather(...)` — it
    # propagates from the handler loop, never delivered into the workflow via gen.throw.
    def raising():
        def thunk():
            raise ValueError("branch boom")
            yield  # unreachable — makes thunk() a generator

        return thunk

    def parent():
        try:
            yield from gather([raising(), _branch("b1", 0.0, [])])
        except ValueError:
            return "CAUGHT"  # must NOT happen — the workflow cannot catch a branch failure
        return "NO-RAISE"

    with pytest.raises(BaseExceptionGroup) as ei:
        RecordingHandler(responses=RESPONSES).run(parent)
    assert any(isinstance(e, ValueError) for e in ei.value.exceptions)  # the branch's raise


def test_L5_sequential_gather_branch_failure_is_uncatchable():
    # Sequential mode (durable handler over a ctx without `concurrent_safe`): a branch's crash
    # propagates from the handler, and the workflow's try/except does not catch it.
    def parent():
        try:
            yield from gather([_boom_branch(), _ok_branch()])
        except ValueError:
            return "CAUGHT"  # must NOT happen
        return "NO-RAISE"

    with pytest.raises((ValueError, ExceptionGroup)) as raised:
        DurableHandler(_SeqCtx(), _RaiseDomain()).run(parent)
    assert [str(leaf) for leaf in leaves(raised.value)] == ["branch boom"]


@pytest.fixture
def recording_handler_await():
    def parent():
        return (yield from gather([_branch("b0", 0.0, []), _await_branch("ev")]))

    parked = RecordingHandler(responses=RESPONSES).run(parent)
    assert isinstance(parked, Suspended)
    return parked


def test_gather_branch_await_with_a_canned_response_never_parks():
    """Children inherit ``responses`` with bare-name lookup, so a canned branch
    await binds inline: no park, no wall."""

    def parent():
        return (yield from gather([_await_branch("ev")]))

    result = RecordingHandler(responses={"ev": {"canned": True}}).run(parent)
    assert result == [{"canned": True}]


def test_resumed_gather_trace_orders_by_branch_and_replays():
    """The deferred merge keeps the canonical record in branch-index order with
    the delivered event under its PREFIXED key — so ``ReplayHandler`` re-runs a
    resumed-gather trace unchanged (re-execution, zero re-parking)."""
    counter: list[str] = []

    def parent():
        out = yield from gather([_branch("b0", 0.0, counter), _await_branch("ev")])
        return {"out": out}

    handler = RecordingHandler(responses=RESPONSES)
    parked = handler.run(parent)
    assert isinstance(parked, Suspended)
    recorded = parked.resume({"ok": True})
    assert recorded == {"out": [("b0", 100), {"ok": True}]}
    assert [e.key.stored() for e in handler.trace] == [
        "gather:0,0;step;tool:b0",
        "gather:0,1;event;ev",  # the delivered event, prefixed like every branch leaf
    ]
    counter.clear()
    assert ReplayHandler(handler.trace).run(parent) == recorded
    assert counter == ["b0"]  # replay re-ran the branch body, fed from the trace


def test_nested_gather_park_composes_and_resumes():
    """A park inside a nested gather qualifies with the FULL path and resumes
    through both levels."""

    def outer_gathering():
        def thunk():
            inner = yield from gather([_await_branch("ev")])
            return inner[0]

        return thunk

    def parent():
        return (yield from gather([outer_gathering(), _branch("b1", 0.0, [])]))

    parked = RecordingHandler(responses=RESPONSES).run(parent)
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() == "gather:0,0;gather:0,0;ev"
    assert parked.resume({"deep": True}) == [{"deep": True}, ("b1", 200)]


def test_gather_suspended_resume_after_completion_is_loud():
    """Resume-once: a second resume on a completed gather must raise, not
    silently double-merge the children."""

    def parent():
        return (yield from gather([_await_branch("ev")]))

    parked = RecordingHandler().run(parent)
    assert isinstance(parked, Suspended)
    assert parked.resume({"ok": 1}) == [{"ok": 1}]
    with pytest.raises(RuntimeError, match="resumes exactly once"):
        parked.resume({"ok": 2})


def test_parked_gather_defers_the_ledger_merge_until_completion():
    """A parked round must not half-merge: the sibling's ledger append reaches
    the parent only at completion, still in branch-index order."""

    def led_branch(eid: str):
        def thunk():
            yield from append_ledger(LedgerRow(event_id=Key.parse(eid), kind="k"))
            return eid

        return thunk

    def parent():
        return (yield from gather([_await_branch("ev"), led_branch("e-b1")]))

    handler = RecordingHandler()
    parked = handler.run(parent)
    assert isinstance(parked, Suspended)
    assert handler.ledger == []  # nothing merged while parked
    assert parked.resume({"ok": 1}) == [{"ok": 1}, "e-b1"]
    assert [row.event_id.stored() for row in handler.ledger] == ["e-b1"]


def test_two_same_arity_gathers_get_distinct_positional_keys():
    """Two distinct gathers with the SAME branch count do not collide. `op_key(Gather)` is
    undefined: a gather's identity is its POSITION (the g-th gather), which only a handler
    walking the execution tree knows, not a content function of the op. Both handlers key
    gathered leaves by `gather:{g},{i};`, so the two gathers occupy disjoint `gather:0:` /
    `gather:1:` namespaces: the keys are injective and the monitor/count-by-key stream does not
    conflate them. Even the same tool in both gathers gets distinct keys: identity is the path,
    not the content."""
    counter: list[str] = []

    def parent():
        a = yield from gather([_branch("b0", 0.0, counter), _branch("b1", 0.0, counter)])
        b = yield from gather([_branch("b0", 0.0, counter), _branch("b2", 0.0, counter)])
        return (a, b)

    handler = RecordingHandler(responses=RESPONSES)
    recorded = handler.run(parent)
    keys = [e.key.stored() for e in handler.trace]
    assert keys == [
        "gather:0,0;step;tool:b0",
        "gather:0,1;step;tool:b1",
        "gather:1,0;step;tool:b0",  # same tool as gather:0:0 — distinct by path, not content
        "gather:1,1;step;tool:b2",
    ]
    assert len(set(keys)) == len(keys)  # injective across the two same-arity gathers
    # and it round-trips: replay re-executes to the same path keys
    assert ReplayHandler(handler.trace).run(parent) == recorded
