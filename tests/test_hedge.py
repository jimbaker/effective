"""A hedge: a first attempt, and a second started at an instant while the first runs on.

The spelling is a BOUNDED race inside an UNBOUNDED race's branch. The outer race names no
deadline, so its first attempt is nobody's loser until something answers; the hedge arm runs a
bounded race over a timer, and the timeout that ends it is what starts the second attempt. The
first attempt keeps running across that instant and is cancelled when the second answers.

| what a hedge needs             | what carries it                                          |
|--------------------------------|----------------------------------------------------------|
| a second attempt starting at an | the inner race's deadline, whose `TimedOut` is the start |
| instant                        | signal                                                   |
| a first attempt still running   | the OUTER race, which names no deadline and so cannot    |
| when it does                   | make the first attempt a loser                           |
| the first attempt cancelled     | the outer choice, which stops it at its next admission   |
| when the second answers        |                                                          |

**A race branch may not park, and this spelling never asks one to.** The timer is a step whose
domain call moves the held clock past the bound, so the inner race's only success lands late and
the race answers `TimedOut`. Nothing sleeps and nothing waits out a margin.

**On a ctx whose branches cannot overlap the hedge degenerates**, and the row below pins it: a
sequential interpreter runs the first attempt to its end before it reaches the hedge arm, so the
second attempt never starts and the first always wins. That is the path a deployed Absurd worker
takes, and it is the limit that IS structural.
"""

import threading
from datetime import UTC, datetime, timedelta
from typing import Any

from _shapes import cut_short_by_a_race, run
from test_race_durable import _Tools
from test_race_shapes import _ReleasedByChoice, until

from effective.api import call_tool, quorum
from effective.choice import Chosen, Impossible, TimedOut
from effective.handlers import base
from effective.keys import race_choice, race_prefix
from effective.keys.grammar import TERM_SEPARATOR

START = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
"""Where the held clock stands when the run starts. A fixed date: the clock is held either way."""

WHEN_TO_HEDGE = START + timedelta(seconds=1)
"""The instant the second attempt starts. The timer's own step moves the clock past it, so no
row waits for it to arrive."""

PAST_IT = START + timedelta(seconds=2)
"""Where the timer leaves the clock. Past the instant by a second, which orders nothing: the
comparison is between two instants the race read."""


class _Held:
    """The clock the races read, moved by the timer's own step."""

    def __init__(self) -> None:
        self._at = START.timestamp()

    def __call__(self) -> float:
        return self._at

    def move_to(self, at: datetime) -> None:
        self._at = at.timestamp()


class _Sequential:
    """A ctx whose branches cannot overlap, which is what a deployed Absurd worker hands a race."""

    concurrent_safe = False

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)


def timer():
    return call_tool("tick", {}, str)


def hedge_arm():
    """The second attempt, and the instant that starts it.

    The inner race's timeout is the whole mechanism: the timer's step leaves the clock past the
    bound, so its success is late and the race answers `TimedOut` rather than taking it."""
    match (yield from quorum(1, [timer], deadline=WHEN_TO_HEDGE)):
        case TimedOut():
            return (yield from call_tool("fast", {}, str))
        case Chosen(winners=winners):
            return f"the timer was in time: {[won.value for won in winners]}"
        case Impossible():
            return "the timer could not answer"


def first_attempt():
    first = yield from call_tool("slow-1", {}, str)
    second = yield from call_tool("slow-2", {}, str)
    return f"{first}/{second}"


def hedged(_run_id: str):
    match (yield from quorum(1, [first_attempt, hedge_arm])):
        case Chosen(winners=winners):
            return [won.value for won in winners]
        case other:
            return ["the hedge answered nothing", type(other).__name__]


def _paths(monkeypatch) -> list[str]:
    """The path each race of a run took, in the order the races started.

    The sequential cell's ANSWER is one the concurrent path usually gives as well, since the first
    attempt usually wins either way; the path is what only a ctx whose branches cannot overlap
    produces, so the row names it rather than inferring it from the answer."""
    taken: list[str] = []
    in_order, concurrently = base.Racing.in_order, base.Racing.concurrently

    def sequential(self: base.Racing) -> list[Any]:
        taken.append("in_order")
        return in_order(self)

    async def overlapping(self: base.Racing) -> list[Any]:
        taken.append("concurrently")
        return await concurrently(self)

    monkeypatch.setattr(base.Racing, "in_order", sequential)
    monkeypatch.setattr(base.Racing, "concurrently", overlapping)
    return taken


def _moving(clock: _Held, before: dict[str, Any]) -> _Tools:
    """The domain: each tool answers its own name, and `tick` leaves the clock past the bound."""

    def tick() -> None:
        before.get("tick", lambda: None)()
        clock.move_to(PAST_IT)

    return _Tools({**before, "tick": tick})


def test_the_second_attempt_starts_at_the_instant_and_the_first_is_cut(backend, monkeypatch):
    """The hedge, whole: the first attempt is inside its first call when the second starts, and
    it is stopped at its second call once the second answers.

    The first attempt is released by the outer CHOICE rather than by the second attempt's call,
    so its next op is admitted after the choice and cannot be admitted at all. Released by the
    call, it would race the choice and could win, which is the run this row must not report."""
    clock = _Held()
    monkeypatch.setattr(base, "race_clock", clock)
    taken = _paths(monkeypatch)
    started, settled = threading.Event(), threading.Event()
    tools = _moving(
        clock,
        {
            "slow-1": lambda: (started.set(), until(settled)),
            "tick": lambda: until(started),
        },
    )

    outcome = run(
        backend,
        hedged,
        tools,
        wrap=lambda ctx: _ReleasedByChoice(ctx, settled, race_choice(0)),
        max_attempts=1,
    )

    assert outcome.snap.result == ["fast"]
    assert tools.calls == ["slow-1", "tick", "fast"]
    assert taken == ["concurrently", "concurrently"]
    cut = race_prefix(0, 0).removesuffix(TERM_SEPARATOR)
    assert cut_short_by_a_race(backend, outcome) == frozenset({(cut,)})


def test_a_sequential_ctx_runs_the_first_attempt_to_its_end(backend, monkeypatch):
    """The limit that is structural: branches that cannot overlap make the hedge a no-op.

    No hold here, since there is nothing to hold: the first attempt has answered before the hedge
    arm is reached, so the timer never ticks and the second attempt never starts."""
    clock = _Held()
    monkeypatch.setattr(base, "race_clock", clock)
    taken = _paths(monkeypatch)
    tools = _moving(clock, {})

    outcome = run(backend, hedged, tools, wrap=_Sequential, max_attempts=1)

    assert taken == ["in_order"]
    assert outcome.snap.result == ["slow-1/slow-2"]
    assert tools.calls == ["slow-1", "slow-2"]
    cut = race_prefix(0, 1).removesuffix(TERM_SEPARATOR)
    assert cut_short_by_a_race(backend, outcome) == frozenset({(cut,)})
