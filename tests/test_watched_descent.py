"""A descent watched by one deadline, on both engines, spelled by `unfold` and by `fix`.

The shape: a linear drill whose every level races its work against a single instant the run read
before it started. While the instant is ahead of the run the level's branches all run and the
descent goes deeper; once the run's clock reaches it the race decides `TimedOut` and the level
answers from what the LEDGER holds rather than from what it was about to compute. That is a
watchdog over an internal trace: the property is decided over the run's own durable progress, and
the bound is the instant any watcher may carry.

The cut level's branches start and are stopped at their first admission, which is what the
endings record says: two `stopped` and no call. A race reads its deadline before launching a
branch, so nothing of that level's work runs.

**The clock is held and the run moves it, so no row here sleeps or waits out a margin.** Each
level stamps `handlers.base.race_time` at its own depth as its first op, which is this suite's
model of work taking time, and the level's stamp is what its race measures the bound against. The
cut lands at `CUT_AT` because that is where the stamps reach the bound, not because a host was
slow.

**Why the stamp is absolute and not an increment.** A stamp says where the clock stands at a
depth, so re-running one is idempotent and SKIPPING one, which a retry does for every level whose
step is already recorded, leaves the clock where the crashed attempt put it: where the level being
resumed needs it. An increment says how far the clock has moved, so a retry that replays the levels
above the crash without re-running their steps arrives at the deadline somewhere else, and
`sweep`'s cells stop converging.

**The convergence it proves is under this clock's evolution, and no other.** A crash before a
level's choice commits leaves the retry to decide that race again, and if the clock had moved to
the bound in between, the retry would legitimately cut there instead. Ledger idempotency preserves
the rows already appended; it cannot preserve a decision that never reached the store.

**What this row does NOT reach.** Its path is `deadline_of`, `Racing.expired`,
`before_any_branch`, `decide` with `expired`, the stored choice and the loser's stop. Everything
downstream of when a branch ENDED is outside it, and four mutants show where the edge is: `in_time`
answering True, `concurrently`'s wake bound dropped, `in_order`'s end stamp lost, and `read`
dropping the late-success evidence each leave this file green. `tests/test_race_deadline.py` is
where those live.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any

import pytest
from _shapes import (
    Program,
    Shape,
    agree,
    cut_short_by_a_race,
    independent,
    interleave,
    run,
    sweep,
    sweep_pairs,
)

from effective.api import Effect, append_ledger, call_tool, quorum, scoped
from effective.choice import Chosen, Impossible, TimedOut
from effective.combinators import Answered, Decision, Deeper, Level, fix, unfold
from effective.domain import CallTool, DomainOp
from effective.handlers import base
from effective.keys import Index, Key, Run, compose_key, race_prefix
from effective.keys.grammar import TERM_SEPARATOR
from effective.ops import LedgerRow

LEVELS = 4
"""The descent's budget: a watched run that is never cut answers at level `LEVELS`."""

CUT_AT = 2
"""The level whose race meets the bound. Above 0, so a judge that only runs at the root cannot
reach it, and below `LEVELS`, so the cut is what ends the descent rather than the budget."""

CUT_RACE = CUT_AT
"""The cut race's ordinal, which equals its depth because each level runs exactly one race. A
level that raced twice would break the identity and not the row, so the two are named apart."""

BRANCHES = 2
"""Branches per level, and the quorum each level wants. Two, so the level has siblings for a
schedule to order; all of them wanted, so nothing is cut by completion order."""

START = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
"""Where a run's clock stands at level 0. A fixed date: the clock is held either way."""

A_LEVEL = timedelta(seconds=1)
"""What one level of the descent costs the run's clock. It orders nothing: the stamps are
absolute, so a level's own depth decides where its clock stands."""

BOUNDS = {
    "the stamp meets it": START + CUT_AT * A_LEVEL,
    "the stamp passes it": START + CUT_AT * A_LEVEL - A_LEVEL / 2,
}
"""The one instant the whole descent races, placed two ways. Level `CUT_AT` is cut under both:
its stamp lands ON the first and strictly PAST the second, and the level above it stamps before
either, so the answer is the same.

Each placement kills a mutant the other cannot. An arrival test that wants the clock strictly past
the instant misses the first, since the stamp only meets it; one that wants the clock exactly AT
the instant misses the second, since the stamp overshoots."""


def work_id(run_id: str, depth: int, branch: int) -> Key:
    """One branch's durable progress: the run, the level, and the branch within it."""
    return compose_key(t"watch-work:{Run(run_id)},{Index(depth)},{Index(branch)}")


class Descent:
    """The domain: stamps the clock, hands out the bound, answers a branch's work, and counts the
    run's durable progress.

    `progress` reads the LEDGER, which is the canonical record a crash leaves intact. A reading
    taken from this object's own call log would count the discarded attempt's calls too, since one
    domain lives across a run's attempts.

    It reads the store from inside a running task, which is sound because a ledger append commits
    on its own: `PostgresLedger.append` opens a transaction per row rather than folding the row
    into the worker's checkpoint transaction, so a plain SELECT sees every row already appended
    and waits on nothing. That is the two-bookkeepers rule as a precondition, and the row would
    catch its loss: appends moved inside the worker's transaction leave the judge counting
    fewer."""

    def __init__(self, backend: Any, clock: Held, bound: datetime) -> None:
        self._backend, self._clock, self._bound = backend, clock, bound
        self.calls: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name="stamp", args={"depth": int(depth)}):
                self.calls.append(f"stamp:{depth}")
                self._clock.at(depth)
                return depth
            case CallTool(name="bound", args={}):
                self.calls.append("bound")
                return self._bound.isoformat()
            case CallTool(name="work", args={"depth": int(depth), "branch": int(branch)}):
                self.calls.append(f"work:{depth},{branch}")
                return f"{depth},{branch}"
            case CallTool(name="progress", args={"run": str(run_id)}):
                self.calls.append("progress")
                return len(self._backend.ledger_payloads(run_id))
        raise TypeError(f"the descent answers stamp, bound, work and progress, not {op!r}")


class Held:
    """The clock every race in this file reads, standing where the deepest stamp put it."""

    def __init__(self) -> None:
        self._at = START.timestamp()

    def __call__(self) -> float:
        return self._at

    def at(self, depth: int) -> None:
        self._at = (START + depth * A_LEVEL).timestamp()


# --- the level, written once and closed two ways ------------------------------------------------


def work(run_id: str, depth: int, branch: int) -> Effect[str]:
    value = yield from call_tool("work", {"depth": depth, "branch": branch}, str)
    yield from append_ledger(
        LedgerRow(event_id=work_id(run_id, depth, branch), kind="work", at=value)
    )
    return value


def progress(run_id: str) -> Effect[int]:
    return call_tool("progress", {"run": run_id}, int)


def level_scope(depth: int) -> Key:
    return compose_key(t"d:{Index(depth)}")


def watched(run_id: str, bound: datetime, depth: int, final: bool) -> Effect[list[Any] | None]:
    """One level: stamp the clock, race the branches against the bound, and judge what is left.

    `None` says the descent goes deeper; a list is the run's answer. The judge reads the ledger at
    every level that ends the descent, so a spelling that judged at the root alone would answer
    from a trace nothing had written."""
    yield from call_tool("stamp", {"depth": depth}, int)
    branches = [partial(work, run_id, depth, i) for i in range(BRANCHES)]
    match (yield from quorum(BRANCHES, branches, deadline=bound)):
        case Chosen() if not final:
            return None
        case Chosen():
            return ["done", depth, (yield from progress(run_id))]
        case TimedOut():
            return ["cut", depth, (yield from progress(run_id))]
        case Impossible():
            return ["impossible", depth, (yield from progress(run_id))]


def watching(run_id: str, bound: datetime) -> Callable[[Any, Level], Effect[Decision[Any, Any]]]:
    def node(_ctx: Any, level: Level) -> Effect[Decision[Any, Any]]:
        answer = yield from watched(run_id, bound, level.depth, level.final)
        return Deeper(None) if answer is None else Answered(answer)

    return node


def open_watching(run_id: str, bound: datetime, budget: int):
    def close(again: Callable[[int], Effect[list[Any]]]) -> Callable[[int], Effect[list[Any]]]:
        def body(depth: int) -> Effect[list[Any]]:
            answer = yield from scoped(
                level_scope(depth), partial(watched, run_id, bound, depth, depth == budget)
            )
            return (yield from again(depth + 1)) if answer is None else answer

        return body

    return close


def bound_of() -> Effect[datetime]:
    """The one instant the run races, read before the descent starts and checkpointed there.

    A level that read its own would race an instant the retry recomputes."""
    raw = yield from call_tool("bound", {}, str)
    return datetime.fromisoformat(raw)


def by_unfold(run_id: str) -> Effect[list[Any]]:
    bound = yield from bound_of()
    return (yield from unfold(None, watching(run_id, bound), budget=LEVELS))


def by_fix(run_id: str) -> Effect[list[Any]]:
    bound = yield from bound_of()
    return (yield from fix(open_watching(run_id, bound, LEVELS))(0))


SPELLINGS: dict[str, Program] = {"unfold": by_unfold, "fix": by_fix}


def reached(depth: int) -> list[str]:
    """The levels whose branches ran: every one above the cut."""
    return [f"{d},{i}" for d in range(depth) for i in range(BRANCHES)]


def expected_rows(run_id: str) -> list[str]:
    return sorted(work_id(run_id, d, i).stored() for d in range(CUT_AT) for i in range(BRANCHES))


ANSWER = ["cut", CUT_AT, BRANCHES * CUT_AT]
"""Derived from the design: the levels above `CUT_AT` each write `BRANCHES` rows, and the level
that meets the bound writes none, so the judge counts what the descent got through."""


@pytest.fixture
def held(monkeypatch):
    """The clock the races read, replaced for the whole row: `agree` and `sweep` run many runs and
    each of them stamps its own levels."""
    clock = Held()
    monkeypatch.setattr(base, "race_time", clock)
    return clock


@pytest.fixture(params=list(BOUNDS))
def bound(request) -> datetime:
    return BOUNDS[request.param]


def watched_row(backend, held: Held, bound: datetime) -> Shape:
    return Shape(
        spellings=SPELLINGS,
        domain=lambda: Descent(backend, held, bound),
        answer=lambda: ANSWER,
        ledger_ids=expected_rows,
        calls=lambda descent: sorted(descent.calls),
        stopped=cut_short_by_a_race,
    )


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_a_crash_after_every_step_converges(backend, held, bound, spelling):
    sweep(backend, watched_row(backend, held, bound), spelling)


def test_two_crashes_converge(backend, held):
    """Each checkpoint crashed, then each later one on the resumed attempt. A stamp skipped
    twice still leaves the clock where the next level to run needs it. One placement of the bound
    suffices, since the two cut the same level by the same stamps."""
    bound = BOUNDS["the stamp passes it"]
    pairs = sweep_pairs(backend, watched_row(backend, held, bound), "unfold")
    assert pairs > 0


def test_the_cut_level_is_the_one_the_stamps_reach(backend, held, bound):
    """`agree` and the cut, in one cell: the two spellings answer alike, place the same names and
    write the same rows, and what they answer is the level the stamps reached."""
    outcomes = agree(backend, watched_row(backend, held, bound))
    for spelling, outcome in outcomes.items():
        assert outcome.snap.result == ANSWER, spelling
        assert sorted(outcome.domain.calls) == sorted(
            ["bound", "progress", *[f"stamp:{d}" for d in range(CUT_AT + 1)]]
            + [f"work:{v}" for v in reached(CUT_AT)]
        ), spelling


def test_the_run_says_it_stopped_the_cut_level_and_no_other(backend, held, bound):
    """The endings name the branches the bound cut, carrying the level they ran in.

    An allowance too generous would name a live level's branches, which ran to their end; one too
    narrow would name none, and the descent's own answer would be the only evidence a level was
    cut at all."""
    outcome = run(backend, by_unfold, Descent(backend, held, bound))

    assert outcome.snap.result == ANSWER
    assert cut_short_by_a_race(backend, outcome) == frozenset(
        (level_scope(CUT_AT).stored(), race_prefix(CUT_RACE, i).removesuffix(TERM_SEPARATOR))
        for i in range(BRANCHES)
    )


def test_the_descent_is_schedule_independent(backend, held, bound):
    """Predicted before it ran: both branches of every live level are wanted, so no schedule can
    change which of them the quorum takes, and the cut level's branches are stopped either way.
    `interleave` holds each run to `ANSWER`, so a descent that runs to its budget fails its cell.
    """
    seen = interleave(backend, watched_row(backend, held, bound))
    assert independent(seen), seen
