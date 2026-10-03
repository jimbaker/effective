"""`cut_short_by_a_race`, pinned on a race whose loser is held until the choice is stored.

**The turnstile cannot force a race's answer, and that is what this row records.** It orders op
ADMISSIONS, while a race turns on branch COMPLETIONS, and it releases the next turn from its
layer's `finally`, which runs before a branch body returns. Two measurements, on SQLite under CPU
contention: a schedule ordering branch 1's steps first still answered branch 0 in 5 of 200 runs,
since both completions landed in one batch and `decide` ranks a batch by index; and a schedule
that lets every branch finish cancels nothing, so there is no stopped branch to allow. A turn also
spans the domain call, so a branch held in the domain holds its turn and the order deadlocks.

What does force a race is the two-event hold of `tests/test_race_durable.py`: the winner waits
until the loser is inside its first call, and the loser waits for the choice. `_ReleasedByChoice`
watches for that one checkpoint rather than any settle, so the release is the choice itself.
"""

import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from _schedules import Turnstile
from _shapes import allowed, cut_short_by_a_race, placed_step, run
from test_race_durable import _described, _Tools

from effective.api import append_ledger, call_tool, direct_tool_key, quorum, race, scoped
from effective.choice import Chosen
from effective.handlers import base
from effective.handlers.base import step_key
from effective.keys import Index, Key, Run, compose_key, race_choice, race_prefix
from effective.keys.frame import scope_prefix
from effective.keys.grammar import TERM_SEPARATOR
from effective.ops import LedgerRow

HOLD = 10.0
"""How long a hold waits before it fails its row. It orders nothing: every release here is an
event the run itself sets, so a hold that runs out is a release that never came, and `until` says
so rather than letting the row carry on ten seconds late."""


def until(event: threading.Event) -> None:
    assert event.wait(HOLD), "a hold was never released"


class _ReleasedByChoice:
    """Wraps a ctx and sets `settled` when the race's choice lands, and for nothing else.

    `test_race_durable`'s `_Settled` fires on every settle, so a loser is released by whatever the
    winner checkpointed first. Keying on the choice names the event the assertion is about."""

    def __init__(self, ctx: Any, settled: threading.Event, choice: Key) -> None:
        self._ctx, self._settled, self._choice = ctx, settled, choice

    def settle(self, name: Key, value: Any) -> Any:
        stored = self._ctx.settle(name, value)
        if name == self._choice:
            self._settled.set()
        return stored

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)


def branch(*names: str):
    def body():
        for name in names:
            last = yield from call_tool(name, {}, str)
        return last

    return body


def racing(_run_id: str):
    return _described((yield from quorum(1, [branch("x-a", "x-b"), branch("y-a", "y-b")])))


SCOPE = compose_key(t"d:{Index(0)}")


def racing_in_a_scope(_run_id: str):
    """The same race one frame down, so a reader that dropped the enclosing frames is wrong."""
    return (yield from scoped(SCOPE, lambda: racing(_run_id)))


def needing_both(_run_id: str):
    """`want` equal to the branch count: nothing can be stopped, so nothing is cut short."""
    return _described((yield from quorum(2, [branch("x-a", "x-b"), branch("y-a", "y-b")])))


def test_the_allowance_names_the_branch_the_race_stopped(backend):
    """Branch 1 is held inside its first call until the choice, so its second op is admitted after
    the flag and is not admitted at all. `cut_short_by_a_race` must name that branch and no other.

    An allowance too generous would name branch 0 as well, and one too narrow would name neither,
    so the equality is what the reader is held to."""
    started, settled = threading.Event(), threading.Event()
    tools = _Tools(
        {
            "x-a": lambda: until(started),
            "y-a": lambda: (started.set(), until(settled)),
        }
    )
    outcome = run(
        backend,
        racing,
        tools,
        wrap=lambda ctx: _ReleasedByChoice(ctx, settled, race_choice(0)),
        max_attempts=1,
    )
    assert outcome.snap.result == ["chosen", [["won", 0, "x-b"], ["stopped", 1, None]]]
    assert sorted(tools.calls) == ["x-a", "x-b", "y-a"]
    loser = race_prefix(0, 1).removesuffix(TERM_SEPARATOR)
    assert cut_short_by_a_race(backend, outcome) == frozenset({(loser,)})


def test_a_quorum_that_needs_every_branch_cuts_nothing_short(backend):
    """The reader's other answer, and the one an over-generous allowance would get wrong."""
    outcome = run(backend, needing_both, _Tools(), max_attempts=1)
    assert outcome.snap.result == ["chosen", [["won", 0, "x-b"], ["won", 1, "y-b"]]]
    assert cut_short_by_a_race(backend, outcome) == frozenset()


def test_the_allowance_carries_the_frames_enclosing_its_race(backend):
    """A race one scope down: the branch is named by the whole path, not by its own frame.

    Without this the enclosing frames are unread, since a top-level race has none and any reader
    that dropped them would answer alike."""
    started, settled = threading.Event(), threading.Event()
    tools = _Tools(
        {
            "x-a": lambda: until(started),
            "y-a": lambda: (started.set(), until(settled)),
        }
    )
    outcome = run(
        backend,
        racing_in_a_scope,
        tools,
        wrap=lambda ctx: _ReleasedByChoice(
            ctx, settled, race_choice(0).prefixed(scope_prefix(SCOPE))
        ),
        max_attempts=1,
    )
    assert outcome.snap.result == ["chosen", [["won", 0, "x-b"], ["stopped", 1, None]]]
    loser = race_prefix(0, 1).removesuffix(TERM_SEPARATOR)
    assert cut_short_by_a_race(backend, outcome) == frozenset({(SCOPE.stored(), loser)})


# --- what the choice decides, and what it does not ----------------------------------------------

BOTH_ANSWER = "the same either way"
"""Both routes compute it, so the race's winner cannot change what the run returns."""


def route_id(run_id: str, index: int) -> Key:
    return compose_key(t"race-route:{Run(run_id)},{Index(index)}")


def route(run_id: str, index: int, tool: str):
    """One route: a call, then the row saying this route ran. A loser reaches the call and is
    stopped at the row, so the choice is what decides which row exists."""

    def body():
        yield from call_tool(tool, {}, str)
        yield from append_ledger(
            LedgerRow(event_id=route_id(run_id, index), kind="route", by=tool)
        )
        return BOTH_ANSWER

    return body


def racing_two_routes(run_id: str):
    match (yield from quorum(1, [route(run_id, 0, "x"), route(run_id, 1, "y")])):
        case Chosen(winners=winners):
            return [won.value for won in winners]
        case other:
            return _described(other)


HOLDS = {"branch 0 wins": ("y", "x"), "branch 1 wins": ("x", "y")}
"""Which tool sets the handshake and waits for the choice, and which waits for the handshake.

The loser enters its call before the winner proceeds, so the loser is stopped at its ROW rather
than before its call, and the two runs differ by the row alone."""


@pytest.mark.parametrize("hold", list(HOLDS))
def test_the_choice_decides_the_rows_and_not_the_answer(backend, hold):
    """The claim the branch-and-bound refutation left behind, measured both ways: the effects a
    cancelling shape leaves depend on the choice, and its answer does not.

    A schedule cannot force this, which is what the module docstring records. The two-event hold
    can, so the comparison is between two runs whose winner was chosen rather than observed."""
    loser, winner = HOLDS[hold]
    started, settled = threading.Event(), threading.Event()
    tools = _Tools(
        {
            loser: lambda: (started.set(), until(settled)),
            winner: lambda: until(started),
        }
    )
    outcome = run(
        backend,
        racing_two_routes,
        tools,
        wrap=lambda ctx: _ReleasedByChoice(ctx, settled, race_choice(0)),
        max_attempts=1,
    )

    assert outcome.snap.result == [BOTH_ANSWER]
    assert sorted(tools.calls) == ["x", "y"]
    won = 0 if winner == "x" else 1
    assert backend.ledger_ids(outcome.run_id) == [route_id(outcome.run_id, won).stored()]


# --- the allowance, fired: a branch cut with an op the schedule names still ahead of it ---------

AT = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
"""The deadline, and where branch 0's last call leaves the held clock. A fixed date."""

AHEAD = (AT - timedelta(hours=1)).timestamp()
"""Where the held clock starts: far enough before `AT` that the race launches both branches. It
orders nothing, since nothing but branch 0's own call moves the clock."""


def _placed(tool: str, index: int) -> str:
    """Where a call in branch `index` of the first race runs, composed as the handler composes it,
    which is the name a `placed_step` turnstile orders it by."""
    return step_key(direct_tool_key(tool).stored()).prefixed(race_prefix(0, index)).stored()


def _cut_mid_flight(_run_id: str):
    def ahead():
        yield from call_tool("a-1", {}, str)
        return (yield from call_tool("a-clock", {}, str))

    def held_then_tail():
        yield from call_tool("b-1", {}, str)
        yield from call_tool("b-hold", {}, str)
        return (yield from call_tool("b-tail", {}, str))

    return _described((yield from race([ahead, held_then_tail], deadline=AT)))


PREFIXES = {
    "branch 0 first": (_placed("a-1", 0), _placed("b-1", 1)),
    "branch 1 first": (_placed("b-1", 1), _placed("a-1", 0)),
}
"""The two orders of the branches' first steps. The rest of the order is the same under both."""


@pytest.mark.parametrize("prefix", list(PREFIXES))
def test_the_allowance_forgives_an_op_the_turnstile_never_sees(backend, monkeypatch, prefix):
    """A branch cut with an op the schedule names still ahead of it, and the allowance forgiving
    exactly that op.

    `b-hold` is outside the order, so it holds no turn while it waits for the choice, and that is
    what lets the turnstile and the choice hold coexist: an ordered call that waited would keep its
    turn and deadlock the order. Branch 0's last call moves the clock onto the deadline as it ends,
    so its success is late and the race answers `TimedOut`; branch 1, released by the choice,
    yields `b-tail` and is stopped at its admission, which a loser meets before any layer does, so
    `b-tail` never reaches the turnstile at all.

    The allowance is checked both ways in this one row: taken, the check passes; withheld, it
    names `b-tail` as the op nothing allows."""
    clock = [AHEAD]
    monkeypatch.setattr(base, "race_clock", lambda: clock[0])
    started, settled = threading.Event(), threading.Event()

    def onto_the_deadline() -> None:
        until(started)
        clock[0] = AT.timestamp()

    tools = _Tools(
        {
            "b-hold": lambda: (started.set(), until(settled)),
            "a-clock": onto_the_deadline,
        }
    )
    order = (*PREFIXES[prefix], _placed("a-clock", 0), _placed("b-tail", 1))
    turnstile = Turnstile(order, placed_step)

    outcome = run(
        backend,
        _cut_mid_flight,
        tools,
        layers=lambda _run_id: (turnstile.layer(),),
        wrap=lambda ctx: _ReleasedByChoice(ctx, settled, race_choice(0)),
        max_attempts=1,
    )

    assert outcome.snap.result == ["timeout", [["unchosen", 0, "a-clock"], ["stopped", 1, None]]]
    assert sorted(tools.calls) == ["a-1", "a-clock", "b-1", "b-hold"]
    assert _placed("b-tail", 1) not in turnstile.ended
    cut = cut_short_by_a_race(backend, outcome)
    assert cut == frozenset({(race_prefix(0, 1).removesuffix(TERM_SEPARATOR),)})
    turnstile.check(allowed(order, cut))
    with pytest.raises(AssertionError, match="nothing allows"):
        turnstile.check()
