"""A race that names a deadline, on the recording and replay core.

Nothing here orders an event with a sleep or waits out a margin. A race reads one clock,
`handlers.base.race_clock`, so a row holds that clock and the deadline fires where the row says
rather than where the host's load puts it. A branch lands on a NAMED instant by moving the held
clock as it ends, which is what makes the tie at the deadline decidable at all.

Two facts about a held clock, both of them why the rows below are shaped as they are:

| holding the clock | does |
|---|---|
| at or past the deadline | decides the race before a branch is launched, so nothing is asked |
| before the deadline | leaves the wait real, so only a branch moving the clock arrives at it |
"""

import asyncio
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any

import pytest

from effective.api import call_tool, quorum, race
from effective.choice import (
    Choice,
    Chosen,
    Impossible,
    Refusal,
    Stopped,
    TimedOut,
    Unchosen,
    Won,
)
from effective.domain import CallTool
from effective.govern import Refused
from effective.handlers import base
from effective.handlers.base import BranchRaised
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.ops import CompositionRefused, Step

DEADLINE = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
"""The instant every row races against. A fixed date, since the clock is held anyway."""

A_MICROSECOND = timedelta(microseconds=1)
"""The resolution the tie is pinned at. `timestamptz` keeps microseconds, and a float here has
room for one: the ulp at this epoch is 2.4e-7 s, so a microsecond is about four of them, and the
margin holds for any instant before 2106."""

A_LONG_WAY_OFF = timedelta(hours=1)
"""How far before the deadline a held clock starts when the row wants the deadline NOT to fire on
its own. It orders nothing: a host slow enough to exhaust it buys one more wake and the same
answer, and the rows run identically at one second and at ten days."""

EARLY = (DEADLINE - A_LONG_WAY_OFF).timestamp()
"""An instant the deadline is still ahead of, as a bare float for the rows that build a `Racing`
of their own rather than running a workflow."""

LONG = 20_000
"""Steps a looping branch takes before it returns on its own. A stopped branch stops long before
this, and one nothing stops is seen returning."""


class _Clock:
    """The clock a race reads, held at an instant the row moves."""

    def __init__(self, at: datetime) -> None:
        self._at = at.timestamp()

    def __call__(self) -> float:
        return self._at

    def move_to(self, at: datetime) -> None:
        self._at = at.timestamp()


class _Tools(Mapping[str, object]):
    """Answers every tool call with its own name, and moves the clock for the calls named in
    `moves`, so a branch ends on the instant the row chose."""

    def __init__(self, clock: _Clock, moves: Mapping[str, datetime] | None = None) -> None:
        self._clock, self._moves = clock, dict(moves or {})

    def __getitem__(self, key: str) -> object:
        if (at := self._moves.get(key)) is not None:
            self._clock.move_to(at)
        if key.startswith("tool:"):
            return key.removeprefix("tool:")
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


class _Held(_Tools):
    """`_Tools`, and a tool whose name starts with one of `holds` waits for `released` before it
    answers: a branch sits inside a domain call while the race decides without it."""

    def __init__(
        self,
        clock: _Clock,
        moves: Mapping[str, datetime],
        released: threading.Event,
        holds: tuple[str, ...] = ("tool:b",),
    ) -> None:
        super().__init__(clock, moves)
        self._released, self._holds = released, holds

    def __getitem__(self, key: str) -> object:
        if key.startswith(self._holds):
            assert self._released.wait(10), "the choice never landed"
        return super().__getitem__(key)


def _returns(name: str):
    def branch():
        return (yield from call_tool(name, {}, str))

    return branch


def _refuses_at_once(reason: str):
    """A branch that refuses before yielding an op, so no choice can stop it anywhere else."""

    def branch():
        raise Refused(
            Step(name="gate", op=CallTool(name="gate", args={}, result_schema=str)), reason
        )
        yield  # a generator, which never reaches its first yield

    return branch


def _forever(name: str):
    """A branch that ends on its own only after `LONG` steps; a loser stops at an admission."""

    def branch():
        for n in range(LONG):
            yield from call_tool(f"{name}{n}", {}, str)
        return "ran out"

    return branch


def _run(
    monkeypatch, clock: Callable[[], float], program, responses: Mapping[str, object]
) -> tuple[Any, Any]:
    monkeypatch.setattr(base, "race_clock", clock)
    handler = RecordingHandler(responses=responses)
    return handler.run(program), handler


def test_a_deadline_the_race_never_reaches_answers_what_no_deadline_would(monkeypatch):
    """The `agree` row: a bound nothing reaches is a bound that changed nothing.

    The losing branch refuses before its first op, so it has no admission a choice could stop it
    at and its ending is the same whatever the schedule. A branch that runs ops would end
    `Stopped` or `Unchosen` depending on where the choice caught it, which is a fact about the
    order rather than about the deadline."""

    def racing(deadline):
        def program():
            return (yield from race([_returns("a"), _refuses_at_once("no")], deadline=deadline))

        return program

    clock = _Clock(DEADLINE - A_LONG_WAY_OFF)
    bounded, _ = _run(monkeypatch, clock, racing(DEADLINE), _Tools(clock))
    unbounded, _ = _run(monkeypatch, clock, racing(None), _Tools(clock))
    assert bounded == unbounded == Chosen((Won(0, "a"),), (Won(0, "a"), Refusal(1, "no")))


@pytest.mark.parametrize(
    "held_at",
    [DEADLINE, DEADLINE + timedelta(seconds=1)],
    ids=["at the deadline", "a second past it"],
)
def test_a_deadline_already_reached_stops_every_branch_before_one_runs(monkeypatch, held_at):
    """A race reads its deadline before it launches a branch, so one whose deadline has already
    arrived stops both at their first admission. This is the cancellation a forced schedule could
    not reach: no completion order decides it, and the instant itself counts as arrival."""

    def program():
        return (yield from race([_forever("a"), _forever("b")], deadline=DEADLINE))

    clock = _Clock(held_at)
    answer, handler = _run(monkeypatch, clock, program, _Tools(clock))
    assert answer == TimedOut((Stopped(0), Stopped(1)))
    assert ReplayHandler(handler.trace).run(program) == answer


def test_a_timed_out_race_records_a_choice_that_names_no_winner(monkeypatch):
    def program():
        return (yield from race([_forever("a")], deadline=DEADLINE))

    clock = _Clock(DEADLINE)
    _, handler = _run(monkeypatch, clock, program, _Tools(clock))
    choice = next(e.result for e in handler.trace if e.key.stored() == "race:0;choice")
    assert choice == {"kind": "timeout", "winners": [], "batch": []}


@pytest.mark.parametrize(
    ("ends_at", "expected"),
    [
        (DEADLINE - A_MICROSECOND, Chosen((Won(0, "a"),), (Won(0, "a"),))),
        (DEADLINE, TimedOut((Unchosen(0, "a"),))),
        (DEADLINE + A_MICROSECOND, TimedOut((Unchosen(0, "a"),))),
    ],
    ids=["a microsecond early", "at the deadline", "a microsecond late"],
)
def test_a_branch_that_ends_at_the_deadline_is_late_to_the_microsecond(
    monkeypatch, ends_at, expected
):
    """The bound is the instant, not the interval. The branch moves the clock to `ends_at` on its
    way out, so the instant it ended on is the row's rather than the host's, and a late success is
    `Unchosen` rather than stopped: it ran, and the choice did not take it."""

    def program():
        return (yield from race([_returns("a")], deadline=DEADLINE))

    clock = _Clock(DEADLINE - A_LONG_WAY_OFF)
    answer, handler = _run(monkeypatch, clock, program, _Tools(clock, {"tool:a": ends_at}))
    assert answer == expected
    choice = next(e.result for e in handler.trace if e.key.stored() == "race:0;choice")
    assert choice["batch"] == [0], "the branch that ended is missing from the batch that explains"
    assert ReplayHandler(handler.trace).run(program) == answer


def test_a_deadline_stops_a_branch_that_is_already_running(monkeypatch):
    """The deadline arrives while branch 1 is inside a domain call, because branch 0 moves the
    clock past it as it ends. Branch 0 ran and lost to its own lateness; branch 1 is stopped at
    the admission after the call it was in.

    Every one of branch 1's calls is held until the choice lands, so it cannot run out however
    slow the parent is: left to loop it would record `Unchosen` under a parent that took long
    enough, which is a fact about the host. It is stopped either at its first admission or at the
    one after the call it was held in, and which of those is the schedule's business."""
    choice_landed = threading.Event()
    read = base.Racing.read

    def reading(self: base.Racing, *args: Any) -> bool:
        failing = read(self, *args)
        if self.decided():
            choice_landed.set()
        return failing

    monkeypatch.setattr(base.Racing, "read", reading)

    def program():
        return (yield from race([_returns("a"), _forever("b")], deadline=DEADLINE))

    clock = _Clock(DEADLINE - A_LONG_WAY_OFF)
    moves = {"tool:a": DEADLINE + timedelta(seconds=1)}
    answer, handler = _run(monkeypatch, clock, program, _Held(clock, moves, choice_landed))
    assert answer == TimedOut((Unchosen(0, "a"), Stopped(1)))
    asked = [e.key.stored() for e in handler.trace if e.key.stored().startswith("race:0,1;")]
    assert len(asked) <= 1, "branch 1 ran past the call it was held in"
    assert ReplayHandler(handler.trace).run(program) == answer


def test_a_refusal_still_makes_a_bounded_race_impossible(monkeypatch):
    """A deadline is not a third authority: a race that can no longer reach its quorum answers
    impossible while the deadline is still ahead of it."""

    def refuses():
        yield from call_tool("before", {}, str)
        raise Refused(
            Step(name="gate", op=CallTool(name="gate", args={}, result_schema=str)), "no"
        )

    def program():
        return (yield from race([refuses], deadline=DEADLINE))

    clock = _Clock(DEADLINE - A_LONG_WAY_OFF)
    answer, _ = _run(monkeypatch, clock, program, _Tools(clock))
    assert answer == Impossible((Refusal(0, "no"),))


def test_a_quorum_of_none_answers_before_it_reads_a_deadline(monkeypatch):
    """`want` of 0 yields no op, so its deadline never reaches a handler."""

    def program():
        return (yield from quorum(0, [_returns("a")], deadline=DEADLINE))

    clock = _Clock(DEADLINE + timedelta(seconds=1))
    answer, handler = _run(monkeypatch, clock, program, _Tools(clock))
    assert answer == Chosen((), (Stopped(0),))
    assert handler.trace == []


@pytest.mark.parametrize(
    ("ends_at", "kind"),
    [(DEADLINE - A_MICROSECOND, "winners"), (DEADLINE, "timeout")],
    ids=["a microsecond early", "at the deadline"],
)
def test_a_sequential_race_reads_its_deadline_at_a_branch_boundary(monkeypatch, ends_at, kind):
    """`Racing.in_order` drives a ctx whose branches cannot overlap, and a branch boundary is the
    only place it has to read a clock. The promise a sequential ctx can keep is that the deadline
    is read between branches, never inside one, and what branch 0 ended on decides the kind."""
    clock = _Clock(DEADLINE - A_LONG_WAY_OFF)
    monkeypatch.setattr(base, "race_clock", clock)
    ran: list[int] = []
    decided_after: list[list[int]] = []
    chosen: list[Choice] = []

    def run(i: int) -> str:
        ran.append(i)
        clock.move_to(ends_at)
        return f"branch {i}"

    def choose(choice: Choice) -> None:
        chosen.append(choice)
        decided_after.append(list(ran))

    racing = base.Racing(
        want=1,
        branches=2,
        run=run,
        choose=choose,
        decided=lambda: bool(chosen),
        enclosed=lambda: False,
        tree=threading.Lock(),
        deadline=DEADLINE.timestamp(),
    )
    racing.in_order()
    assert [choice.kind for choice in chosen] == [kind]
    assert decided_after == [[0]], "the deadline was not read at the first branch boundary"


class _Ending(Mapping[str, object]):
    """Answers every tool with its own name, and sets `ending` as the branch's last call runs."""

    def __init__(self, ending: threading.Event) -> None:
        self._ending = ending

    def __getitem__(self, key: str) -> object:
        if not key.startswith("tool:"):
            raise KeyError(key)  # the scoped name, which the handler asks first
        self._ending.set()
        return key.removeprefix("tool:")

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


def test_a_branch_is_stamped_where_it_ran_not_where_the_loop_noticed(monkeypatch):
    """A branch runs in its own thread and the wake that reads it runs on the caller's, and a
    loaded host can put a long gap between the two. The instant a branch ENDED is the one its own
    thread read, and this clock proves which thread read it by answering differently to each: a
    branch thread is always told a microsecond early, and the caller's thread is told an hour
    early until the branch is on its way out, then an hour late. Stamping on the caller's thread
    would make this winner late, and the deadline is never reached before the branch runs, so no
    thread here is racing another."""
    loop_thread, ending = threading.current_thread(), threading.Event()

    def two_faced() -> float:
        if threading.current_thread() is not loop_thread:
            return (DEADLINE - A_MICROSECOND).timestamp()
        return (
            DEADLINE + A_LONG_WAY_OFF if ending.is_set() else DEADLINE - A_LONG_WAY_OFF
        ).timestamp()

    monkeypatch.setattr(base, "race_clock", two_faced)

    def program():
        return (yield from race([_returns("a")], deadline=DEADLINE))

    handler = RecordingHandler(responses=_Ending(ending))
    assert handler.run(program) == Chosen((Won(0, "a"),), (Won(0, "a"),))


def test_a_deadline_bounds_one_wake_and_the_barrier_waits_unbounded(monkeypatch):
    """Once the deadline has decided the race, the wakes that follow are the barrier waiting for
    the branches to end and no clock shortens them. A race that kept bounding them would wake at
    `timeout=0` for as long as a loser took to drain, which is invisible in the answer and
    visible here: the caller's thread reads the clock at most three times per wake, and a race of
    `n` branches wakes at most once per branch and once for the deadline.

    Measured: 4 reads, 30 runs of 30; a race that never stops bounding reads 25."""
    caller, choice_landed = threading.current_thread(), threading.Event()
    clock = _Clock(DEADLINE - A_LONG_WAY_OFF)
    reads = [0]

    def counted() -> float:
        if threading.current_thread() is caller:
            reads[0] += 1
        return clock()

    read = base.Racing.read

    def reading(self: base.Racing, *args: Any) -> bool:
        failing = read(self, *args)
        if self.decided():
            choice_landed.set()
        return failing

    monkeypatch.setattr(base.Racing, "read", reading)

    def loser():
        for n in range(LONG):
            yield from call_tool(f"b{n}", {}, str)
        return "ran out"

    def program():
        return (yield from race([_returns("a"), loser], deadline=DEADLINE))

    moves = {"tool:a": DEADLINE + timedelta(seconds=1)}
    answer, _ = _run(monkeypatch, counted, program, _Held(clock, moves, choice_landed))
    assert answer == TimedOut((Unchosen(0, "a"), Stopped(1)))
    assert reads[0] <= 3 * (2 + 1), f"the deadline bounded more than one wake: {reads[0]} reads"


def test_a_branch_that_ended_late_says_so_even_where_the_clock_reads_early(monkeypatch):
    """The two reads that decide a timeout come off different threads: the branch stamps when it
    ended, and the wake asks whether the deadline has arrived. A clock that steps back between
    them (NTP, a migrated VM) would leave a race answering `Impossible` with a success in its own
    endings, which is neither true nor a shape any caller matches. A branch that ended at or
    after the deadline is itself evidence the deadline arrived, so it decides alone."""
    loop_thread = threading.current_thread()

    def stepped_back() -> float:
        early = threading.current_thread() is loop_thread
        return (DEADLINE - A_LONG_WAY_OFF if early else DEADLINE + A_MICROSECOND).timestamp()

    def program():
        return (yield from race([_returns("a")], deadline=DEADLINE))

    answer, handler = _run(monkeypatch, stepped_back, program, _Tools(_Clock(DEADLINE)))
    assert answer == TimedOut((Unchosen(0, "a"),))
    assert ReplayHandler(handler.trace).run(program) == answer


A_BREATH_OF_REAL_TIME = 0.05
"""How far ahead of a RUNNING clock the one row below puts its deadline.

Every other row holds the clock still, which cannot express the one property left: that the
deadline WAKES a race whose branches are all still running. Nothing is ordered by this interval.
Both branches are blocked until the choice lands, so the timeout wake is the only way the race
can move at all, and a race that stopped bounding its wake hangs here and fails rather than
answering differently."""


def test_the_deadline_wakes_a_race_whose_branches_are_all_still_running(monkeypatch):
    """The wake the deadline bounds is the wake it ends. Both branches sit inside a domain call
    until the choice is saved, so no completion can wake the race: only the deadline can."""
    choice_landed = threading.Event()
    offset = DEADLINE.timestamp() - time.time() - A_BREATH_OF_REAL_TIME
    read = base.Racing.read

    def reading(self: base.Racing, *args: Any) -> bool:
        failing = read(self, *args)
        if self.decided():
            choice_landed.set()
        return failing

    monkeypatch.setattr(base.Racing, "read", reading)
    waits: list[float | None] = []
    real_wait = asyncio.wait

    async def recording_wait(tasks: Any, **kw: Any) -> Any:
        waits.append(kw.get("timeout"))
        return await real_wait(tasks, **kw)

    monkeypatch.setattr(asyncio, "wait", recording_wait)

    def program():
        return (yield from race([_forever("a"), _forever("b")], deadline=DEADLINE))

    held = _Held(_Clock(DEADLINE), {}, choice_landed, holds=("tool:a", "tool:b"))
    answer, handler = _run(monkeypatch, lambda: time.time() + offset, program, held)
    assert answer == TimedOut((Stopped(0), Stopped(1)))
    assert [timeout for timeout in waits if timeout is not None], (
        "no wake was bounded, so the row watched the deadline arrive before the race started "
        "and measured nothing"
    )
    assert ReplayHandler(handler.trace).run(program) == answer


def test_a_clock_crossing_between_two_reads_cannot_disarm_the_deadline(monkeypatch):
    """The race reads its clock twice per wake, once to rule on the batch and once to decide
    whether to keep bounding. An advancing clock that crosses the deadline BETWEEN those two
    reads must not drop the bound while the race has decided nothing, or the next wake waits on
    a completion that may never come. Nothing steps back here and nothing sleeps.

    The invariant, asserted over every wake: a wake is unbounded only after the race has
    something to show for it."""
    crossed, released = threading.Event(), threading.Event()
    monkeypatch.setattr(
        base, "race_clock", lambda: DEADLINE.timestamp() if crossed.is_set() else EARLY
    )
    waits: list[tuple[float | None, list[str]]] = []
    chosen: list[Choice] = []
    real_wait, read = asyncio.wait, base.Racing.read

    async def recording_wait(tasks: Any, **kw: Any) -> Any:
        waits.append((kw.get("timeout"), [choice.kind for choice in chosen]))
        return await real_wait(tasks, **kw)

    def reading(self: base.Racing, batch: list[int], *rest: Any) -> bool:
        failing = read(self, batch, *rest)
        if batch:  # the batch has been ruled on and nothing decided; now the clock crosses
            crossed.set()
        return failing

    monkeypatch.setattr(asyncio, "wait", recording_wait)
    monkeypatch.setattr(base.Racing, "read", reading)

    def run(i: int) -> int:
        assert i == 0 or released.wait(10), "the race never decided"
        return i

    racing = base.Racing(
        want=2,
        branches=2,
        run=run,
        choose=chosen.append,
        decided=lambda: bool(chosen),
        enclosed=lambda: False,
        tree=threading.Lock(),
        deadline=DEADLINE.timestamp(),
    )

    async def drive() -> list[Any]:
        task = asyncio.create_task(racing.concurrently())
        while len(waits) < 2:
            await asyncio.sleep(0)
        released.set()
        return await task

    asyncio.run(drive())
    assert [timeout for timeout, decided in waits if timeout is None and not decided] == []
    assert [choice.kind for choice in chosen] == ["timeout"]


def test_a_naive_deadline_is_refused_because_it_names_no_instant():
    """A naive datetime reads as the local time of whichever host converts it, so the same value
    is two instants on two workers and the rule a restart rests on stops holding. Measured: the
    deadline here converts six hours apart between UTC and America/Denver."""

    def program():
        return (yield from race([_returns("a")], deadline=DEADLINE.replace(tzinfo=None)))

    with pytest.raises(CompositionRefused, match="absolute instant"):
        RecordingHandler(responses=_Tools(_Clock(DEADLINE))).run(program)


@pytest.mark.parametrize(
    "stops_it",
    ["a choice", "a branch that raised", "an enclosing choice", "a choice saved before it began"],
)
def test_a_race_that_can_decide_no_more_stops_bounding_its_wakes(monkeypatch, stops_it):
    """The mirror of the row above. A deadline is a reason to wake a race that still has
    something to decide, and each of these ends that: a saved choice, a programming error before
    any choice, and an enclosing race's choice. A bound outliving them would wake the race at
    `timeout=0` for as long as a loser took to drain, which the answer never shows.

    Branch 0 ends the race's deciding and branch 1 is released only once that has happened, so
    the wake after it is the one under test. The last parameter is a race that arrives with
    nothing left to decide, which is a durable retry served its stored choice: there the wake
    under test is the first one."""
    released, crossed = threading.Event(), threading.Event()
    enclosing, raised, chosen = [stops_it == "a choice saved before it began"], [False], []
    waits: list[tuple[float | None, bool]] = []
    real_wait = asyncio.wait

    def alive() -> bool:
        return not (chosen or raised[0] or enclosing[0])

    async def recording_wait(tasks: Any, **kw: Any) -> Any:
        waits.append((kw.get("timeout"), alive()))
        return await real_wait(tasks, **kw)

    monkeypatch.setattr(asyncio, "wait", recording_wait)
    monkeypatch.setattr(
        base, "race_clock", lambda: DEADLINE.timestamp() if crossed.is_set() else EARLY
    )

    def run(i: int) -> Any:
        if i == 1:
            assert released.wait(10), "the race never stopped deciding"
            return "second"
        match stops_it:
            case "a branch that raised":
                raised[0] = True
                return BranchRaised(ValueError("a programming error"))
            case "an enclosing choice":
                enclosing[0] = True
                return "first"
            case "a choice saved before it began":
                return "first"
            case _:
                crossed.set()  # this branch's own completion is late, which decides the timeout
                return "first"

    read = base.Racing.read

    def reading(self: base.Racing, batch: list[int], *rest: Any) -> bool:
        failing = read(self, batch, *rest)
        if batch:
            released.set()
        return failing

    monkeypatch.setattr(base.Racing, "read", reading)
    racing = base.Racing(
        want=2,
        branches=2,
        run=run,
        choose=chosen.append,
        decided=lambda: bool(chosen),
        enclosed=lambda: enclosing[0],
        tree=threading.Lock(),
        deadline=DEADLINE.timestamp(),
    )
    asyncio.run(racing.concurrently())
    assert not alive(), "the race never stopped deciding, so the row measured nothing"
    assert [timeout for timeout, live in waits if timeout is not None and not live] == []


def test_a_deadline_whose_zone_answers_no_offset_is_refused_too():
    """`tzinfo` is not the question: a `tzinfo` whose `utcoffset` answers `None` leaves the
    datetime naive, and the conversion raises `TypeError` rather than naming the cause. The
    refusal asks what makes an instant absolute, which is the offset."""

    class _NoOffset(tzinfo):
        def utcoffset(self, dt: datetime | None) -> timedelta | None:
            return None

        def tzname(self, dt: datetime | None) -> str | None:
            return None

        def dst(self, dt: datetime | None) -> timedelta | None:
            return None

    def program():
        return (yield from race([_returns("a")], deadline=DEADLINE.replace(tzinfo=_NoOffset())))

    with pytest.raises(CompositionRefused, match="absolute instant"):
        RecordingHandler(responses=_Tools(_Clock(DEADLINE))).run(program)
