"""Two positions in one run never share a key: every walk places a repeated key at `#n`."""

from typing import Any
from uuid import uuid4

import psycopg
import pytest
from _conformance import PG_DSN, CountingDomain, Fault
from _steps import placed_step

from effective.api import await_event, call_tool, scoped
from effective.budget import Grant, MeasuredBudget, depth_grant_name
from effective.checkpoints import is_step_checkpoint
from effective.cost import Usage
from effective.fork import OpIndex, fork_at, measured_drive
from effective.handlers.base import op_key
from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayHandler, ReplayMismatch
from effective.keys import Key, Name, compose_key
from effective.ops import AwaitEvent

LANE = compose_key(t"lane:{Name('p')}")
GO = compose_key(t"go")
GRANT = depth_grant_name("r1", depth=1, generation=0)


def _asks_a():
    return (yield from call_tool("a", {}, int))


def _repeats(run_id: str = ""):
    """`a` twice, then a scope entered twice that asks `a` once each time."""
    first = yield from call_tool("a", {}, int)
    second = yield from call_tool("a", {}, int)
    third = yield from scoped(LANE, _asks_a)
    fourth = yield from scoped(LANE, _asks_a)
    return [first, second, third, fourth]


NUMBERED = [placed_step("a", "tool", frames, n) for frames in ("", "lane:p;") for n in (1, 2)]


def _recorded() -> RecordingHandler:
    recorder = RecordingHandler(responses={"tool:a": 10})
    assert recorder.run(_repeats) == [10, 10, 10, 10]
    return recorder


def test_the_recorder_numbers_each_repeat():
    assert [entry.key.stored() for entry in _recorded().trace] == NUMBERED


def test_replay_matches_a_numbered_trace():
    assert ReplayHandler(_recorded().trace).run(_repeats) == [10, 10, 10, 10]


def test_replay_refuses_a_trace_that_aliases_its_repeats():
    trace = _recorded().trace
    trace[1] = trace[0]
    with pytest.raises(ReplayMismatch):
        ReplayHandler(trace).run(_repeats)


def test_the_engines_checkpoint_the_keys_the_recorder_records(backend):
    """The recorded key and the durable checkpoint name are the same string at every
    occurrence."""
    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, _repeats, CountingDomain(), Fault(None), ())
    task_id = backend.spawn(run_id, run_id)
    assert backend.run_until_result(task_id).result == [10, 10, 10, 10]
    steps = [key for key in backend.checkpoint_keys(task_id) if is_step_checkpoint(key)]
    assert steps == [entry.key.stored() for entry in _recorded().trace] == NUMBERED


class _Tool:
    def __init__(self) -> None:
        self.calls = 0

    def run_metered(self, op: Any) -> tuple[Any, Usage]:
        self.calls += 1
        return 20, Usage()


def _thrice():
    answers = []
    for _ in range(3):
        answers.append((yield from call_tool("a", {}, int)))
    return answers


def test_a_fork_tail_continues_its_prefix_count():
    """Forked at the second `a`, the tail's ask is the third."""
    recorder = RecordingHandler(responses={"tool:a": 10})
    recorder.run(_thrice)
    tail = fork_at(_thrice, recorder.trace, OpIndex(1), 99, _Tool())
    assert tail.result == [10, 99, 20]
    assert [entry.key.stored() for entry in tail.trace] == [placed_step("a", "tool", "", 3)]


def _export(backend, task_id, run_id: str):
    if backend.name == "sqlite":
        from effective.bridge_sqlite import export_measured_prefix

        return export_measured_prefix(backend.app.conn, str(task_id), run_id)
    from effective.bridge_absurd import export_measured_prefix

    with psycopg.connect(PG_DSN) as conn:
        return export_measured_prefix(conn, task_id, run_id)


def test_a_measured_fork_replays_a_bridged_prefix_of_repeats_free(backend):
    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, _repeats, CountingDomain(), Fault(None), ())
    task_id = backend.spawn(run_id, run_id)
    backend.run_until_result(task_id)
    entries, _ = _export(backend, task_id, run_id)
    assert [entry.key.stored() for entry in entries] == NUMBERED

    domain = _Tool()
    tail = measured_drive(
        _repeats, MeasuredBudget(run_id=run_id, overall=1.0), domain, {}, recorded=entries
    )
    assert (tail.result, domain.calls) == ([10, 10, 10, 10], 0)
    assert [driven.key for driven in tail.trace] == [Key.parse(key) for key in NUMBERED]


def _two_awaits():
    first = yield from await_event(GO, int)
    second = yield from await_event(GO, int)
    return [first, second]


def test_the_recorder_numbers_a_resumed_await():
    parked = RecordingHandler().run(_two_awaits)
    assert isinstance(parked, Suspended)
    parked = parked.resume(1)
    assert isinstance(parked, Suspended)
    assert parked.resume(2) == [1, 2]
    asked = op_key(AwaitEvent(GO, int))
    assert [entry.key for entry in parked.handler.trace] == [asked, asked.occurrence(2)]


def _scopes_then_a():
    lanes = [(yield from scoped(LANE, _asks_a)), (yield from scoped(LANE, _asks_a))]
    return [*lanes, (yield from call_tool("a", {}, int))]


def test_a_fork_replays_a_prefix_that_re_enters_a_scope():
    recorder = RecordingHandler(responses={"tool:a": 10})
    recorder.run(_scopes_then_a)
    assert fork_at(_scopes_then_a, recorder.trace, OpIndex(2), 99, _Tool()).result == [10, 10, 99]


def _grant_then_a_then_grant():
    first = yield from await_event(GRANT, Grant)
    called = yield from call_tool("a", {}, int)
    second = yield from await_event(GRANT, Grant)
    return [first, called, second]


def test_a_fork_tail_asks_a_settlement_name_at_the_prefix_count():
    """Forked at the step between two asks, the tail's ask is the second, and only the second
    occurrence's grant answers it."""
    answer = Grant(add_depth=0)
    recorder = RecordingHandler(
        responses={GRANT.stored(): answer, GRANT.occurrence(2).stored(): answer, "tool:a": 10}
    )
    recorder.run(_grant_then_a_then_grant)
    tail = fork_at(
        _grant_then_a_then_grant,
        recorder.trace,
        OpIndex(1),
        99,
        _Tool(),
        grants={GRANT.occurrence(2): answer},
    )
    assert (tail.parked_at, tail.result) == (None, [answer, 99, answer])
