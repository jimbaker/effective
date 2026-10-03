"""The WALK axis: one program, every interpreter, and an assertion that they agree.

Sibling to `_conformance.py`, and the distinction is the whole reason this file exists.
`_conformance` parametrizes over **engines** (SQLite 0<->1, Absurd 0<->N) and asks whether the
durable semantics hold on each. This one parametrizes over **walks**, the interpreters that
drive a workflow and assign identities, and asks whether they *agree with each other*.

Identity bugs are walk disagreements, invisible to an engine axis:

- a `layer_run_state` cell reading `[1, 2]` on the recording walk and `[1, 1]` on replay, so a
  workflow RETURNS a different value on replay than it recorded (a `ReplayHandler` that
  establishes no run scope);
- an await occurrence that exists on one walk of six and nowhere else.

**Why a harness rather than more hand-written cases.** The walks are an inductive family: each
is a fold over the same op stream, recursing the same way into a `Scoped` body and a `Gather`
branch. A property that must hold of the family is tested by parameterizing over it, not by
writing it out per member, since a member nobody wrote a case for is a member nobody notices is
missing.

**What agreement does and does not prove.** It catches DIVERGENCE. It is silent on a defect that
is uniform across walks: `walks_agree` is green on a `depth-grant:` name aliased by every walk,
because every walk composes the same wrong name, agreeably. So `Agreement` returns what was
agreed, and a caller asserts on it in the same breath: the two questions answered by one call,
because a pairing held by convention drifts.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from _durable import DSN, absurd, pg_ready, run_until_result

from effective.bridge_sqlite import park_name
from effective.budget import MeasuredBudget
from effective.cost import Usage
from effective.domain import DomainOp
from effective.fork import live_drive, measured_drive
from effective.handlers.absurd import DurableHandler
from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayHandler
from effective.keys import Key
from effective.sqlite import SqliteApp

type Program = Callable[[], Any]


# --- what one walk observed -------------------------------------------------------------


@dataclass(frozen=True)
class Returned:
    """The workflow ran to completion."""

    value: Any


@dataclass(frozen=True)
class Parked:
    """The workflow suspended on a name nobody had answered.

    An authority name exists to be parked on, so this is the outcome aliasing defects live in,
    and it is comparable across every walk that can park: recording, `live_drive` and the durable
    engines all report `q:0;depth-grant:r,0,1` for one park inside one scope, byte for byte.
    The framed form is the ADDRESS (what an emitter sends), so it is compared as-is and the
    harness does no normalization: there is no second, untested projection here to disagree
    with `unframed`."""

    name: Key


@dataclass(frozen=True)
class Refused:
    """The workflow raised. Compared by type and message, not identity."""

    kind: str
    message: str


type Outcome = Returned | Parked | Refused


@dataclass(frozen=True)
class Ran:
    """One walk's observation of one program."""

    walk: str
    outcome: Outcome
    keys: tuple[str, ...] | None = None
    """The op-key sequence, or `None` where this walk does not produce a comparable one.

    Durable walks report `None` rather than a normalized approximation. Their key stream is
    CHECKPOINTS, and an await writes no checkpoint on SQLite and an engine-internal one on Absurd
    — so a durable key stream and an in-process trace are different alphabets, and squaring them
    would mean inventing a projection whose only consumer is this file. `keys` therefore compares
    across the four walks that genuinely produce a trace, and says so."""


class _CannotRun(Exception):
    """A walk declining a program it structurally cannot run — recorded in `Agreement.skipped`,
    never swallowed. Distinct from a `Refused` OUTCOME, which is the workflow raising."""


class Walk(Protocol):
    name: str

    def run(self, program: Program, answers: dict[Key, Any], steps: tuple[str, ...]) -> Ran: ...


# --- the six ------------------------------------------------------------------------------


def _canned(answers: dict[Key, Any], steps: tuple[str, ...]) -> dict[str, Any]:
    """The recording walk's `responses` table: every step at `STEP_RESULT`, plus the answers."""
    return {name: STEP_RESULT for name in steps} | {k.stored(): v for k, v in answers.items()}


def _outcome_of(result: Any) -> Outcome:
    return Parked(result.awaiting) if isinstance(result, Suspended) else Returned(result)


STEP_RESULT = "v"
"""What every `Step` resolves to, on every walk.

**One constant rather than a per-step table, and that is the design rather than a shortcut.**
The recording walk is driven by a canned `responses` dict while every other walk is driven by a
DOMAIN — two different surfaces — so a probe whose steps returned different values per walk
would report a divergence in the domain plumbing and call it a walk disagreement. Holding the
value fixed makes the harness measure IDENTITY, which is what it is for; a probe that needs a
step to return something particular is asking a domain question and belongs in `_conformance`."""


class _Tool:
    """A domain that answers every step with `STEP_RESULT`."""

    def run(self, op: DomainOp) -> Any:
        return STEP_RESULT

    def run_metered(self, op: DomainOp) -> tuple[Any, Usage]:
        return STEP_RESULT, Usage()


@dataclass
class RecordingWalk:
    name: str = "recording"

    def run(self, program: Program, answers: dict[Key, Any], steps: tuple[str, ...]) -> Ran:
        handler = RecordingHandler(responses=_canned(answers, steps))
        outcome = _outcome_of(handler.run(program))
        return Ran(self.name, outcome, tuple(e.key.stored() for e in handler.trace))


@dataclass
class ReplayWalk:
    """Derived, not independent: replay needs a trace, so it re-records first and then re-drives.

    Its contribution is not a second opinion on the value — it is that the workflow **re-derives**
    every identity rather than reading it back, so anything the recording walk got from ambient
    state has to be reconstructible. A `ReplayMismatch` is the finding; it surfaces as
    `Refused`."""

    name: str = "replay"

    def run(self, program: Program, answers: dict[Key, Any], steps: tuple[str, ...]) -> Ran:
        recorder = RecordingHandler(responses=_canned(answers, steps))
        first = recorder.run(program)
        if isinstance(first, Suspended):
            # **The one walk that cannot answer an unanswered park**, and it is structural rather
            # than a gap to paper over: a trace ending at a park holds no result for the await, so
            # there is nothing to replay PAST and `ReplayHandler` reports an extra op. Resuming it
            # here would make this walk answer a different question from the other five — they
            # parked, it completed — and the harness would report that as a divergence.
            #
            # So it declines, with its reason, and `walks_agree` records it in `skipped`. A probe
            # that wants replay in the comparison supplies the answer; then every walk completes
            # and all six agree on a `Returned`.
            raise _CannotRun("the recording parked; a trace ending at a park has no op to replay")
        try:
            return Ran(
                self.name,
                _outcome_of(ReplayHandler(recorder.trace).run(program)),
                tuple(e.key.stored() for e in recorder.trace),
            )
        except Exception as exc:  # a mismatch IS the observation, not an error to raise
            return Ran(self.name, Refused(type(exc).__name__, str(exc)[:200]))


@dataclass
class SqliteWalk:
    name: str = "sqlite"

    def run(self, program: Program, answers: dict[Key, Any], steps: tuple[str, ...]) -> Ran:
        app = SqliteApp(":memory:")
        task_name = f"w-{uuid.uuid4().hex[:8]}"

        @app.register_task(task_name)
        def task(params, ctx):
            return DurableHandler(ctx, _Tool()).run(program)

        task_id = app.spawn(task_name, {"run_id": "r"})
        for name, value in answers.items():
            app.emit_event(name.stored(), value)
        app.work_batch()
        snapshot = app.run_until_result(task_id)
        # The relation, not the column: `tasks.waiting_event` outlives the park it names, so a
        # raw read reports a park for a run that is sleeping, retrying or finished — inside the
        # instrument that decides whether the two engines agree.
        waiting = park_name(app.conn, task_id)
        app.close()
        if snapshot is not None and snapshot.state == "completed":
            return Ran(self.name, Returned(snapshot.result))
        return Ran(
            self.name,
            Parked(Key.parse(waiting)) if waiting else Refused("Unfinished", str(snapshot)),
        )


@dataclass
class AbsurdWalk:
    name: str = "absurd"

    def run(self, program: Program, answers: dict[Key, Any], steps: tuple[str, ...]) -> Ran:
        app = absurd()
        task_name = f"w-{uuid.uuid4().hex[:8]}"

        @app.register_task(task_name, default_max_attempts=1)
        def task(params, ctx):
            return DurableHandler(ctx, _Tool()).run(program)

        spawned = app.spawn(task_name, {"run_id": "r"})
        task_id = spawned["task_id"] if isinstance(spawned, dict) else spawned
        for name, value in answers.items():
            app.emit_event(name.stored(), value)
        app.work_batch()
        snapshot = run_until_result(app, task_id, max_batches=6)
        if snapshot is not None and snapshot.state == "completed":
            return Ran(self.name, Returned(snapshot.result))
        # A park is a WAIT ROW, not a task state — `sleeping` is what the scheduler calls it, and
        # the name it is waiting on lives in `w_default.event_name`.
        import psycopg

        with psycopg.connect(DSN) as conn:
            waiting = conn.execute(
                "SELECT event_name FROM absurd.w_default WHERE task_id=%s", (str(task_id),)
            ).fetchone()
        if waiting and waiting[0]:
            return Ran(self.name, Parked(Key.parse(waiting[0])))
        return Ran(self.name, Refused("Unfinished", str(snapshot and snapshot.state)))


@dataclass
class LiveForkWalk:
    name: str = "live_drive"

    def run(self, program: Program, answers: dict[Key, Any], steps: tuple[str, ...]) -> Ran:
        tail = live_drive(program(), None, _Tool(), dict(answers))
        outcome = Parked(tail.parked_at) if tail.parked_at is not None else Returned(tail.result)
        return Ran(self.name, outcome, tuple(e.key.stored() for e in tail.trace))


@dataclass
class MeasuredForkWalk:
    name: str = "measured_drive"

    def run(self, program: Program, answers: dict[Key, Any], steps: tuple[str, ...]) -> Ran:
        tail = measured_drive(
            program, MeasuredBudget(run_id="r", overall=1e9), _Tool(), dict(answers)
        )
        outcome = Parked(tail.tripped_at) if tail.tripped_at is not None else Returned(tail.result)
        return Ran(self.name, outcome, tuple(e.key.stored() for e in tail.trace))


ALL_WALKS: tuple[Walk, ...] = (
    RecordingWalk(),
    ReplayWalk(),
    SqliteWalk(),
    AbsurdWalk(),
    LiveForkWalk(),
    MeasuredForkWalk(),
)
"""Every walk that drives a workflow and assigns identities.

Held against `placing(...)`'s call sites, which is the mint every one of them passes through —
so a new interpreter is a new `placing` site, and a `placing` site with no walk here is a walk
nobody is watching. `--walk-coverage` is the gate that would make that a violation rather than a
thing to notice; until it exists, this docstring is the claim and `test_walks.py` checks it."""

ALL_WALK_NAMES: tuple[str, ...] = tuple(w.name for w in ALL_WALKS)


def ran_here(*names: str) -> tuple[str, ...]:
    """The walks a caller should expect in `Agreement.ran` ON THIS MACHINE — the names it passed,
    less `absurd` when there is no Postgres to run it on.

    `walks_agree` drops the absurd walk by name when `pg_ready()` is false, so a test that spells
    out the full tuple asserts something this harness never promised. It went unnoticed because
    the gates run against a live container; `just test-fast` forces a DEAD `DATABASE_URL` on
    purpose, and there both `ran` assertions in `test_walks.py` failed on every machine.

    The exemption is exactly one name under exactly one condition, which is what keeps the
    assertion sharp: a walk lost to `_CannotRun`, or the absurd walk lost while Postgres IS up,
    still reddens the caller. `Agreement.ran` and `Agreement.skipped` partition the walks, so
    pinning `ran` pins both."""
    return names if pg_ready() else tuple(n for n in names if n != "absurd")


# --- the assertion --------------------------------------------------------------------------


@dataclass(frozen=True)
class Agreement:
    """What every walk that ran agreed on — and, as loudly, which walks did not run."""

    outcome: Outcome
    keys: tuple[str, ...] | None
    ran: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...] = ()
    keys_from: tuple[str, ...] = ()
    observations: tuple[Ran, ...] = field(default=(), repr=False)


class Divergence(AssertionError):
    """The walks disagreed. The message is a MATRIX, not a diff of two values — the reader needs
    to see which walks clustered, because that is what names the mechanism (an engine pair
    against the in-process pair reads differently from one lone driver)."""


def _distinct(rows: Sequence[tuple[str, Any]]) -> list[Any]:
    """The distinct values, by EQUALITY rather than hashing.

    An outcome may carry a `list` or a `dict` — the ambient probe returns one — so a `set` is not
    available. Equality is the right relation anyway: two walks agree when their observations are
    equal, and `Key.__eq__` deliberately ignores the declared scope, so a durable walk's parsed
    key compares equal to a composed one exactly as it should."""
    seen: list[Any] = []
    for _, value in rows:
        if not any(value == other for other in seen):
            seen.append(value)
    return seen


def _matrix(dimension: str, rows: Sequence[tuple[str, Any]], skipped) -> str:
    width = max((len(n) for n, _ in rows), default=0)
    shown = _distinct(rows)
    lines = [f"walk-invariance failed on `{dimension}` ({len(shown)} distinct):", ""]
    common = max(shown, key=lambda v: sum(1 for _, o in rows if o == v))
    for name, value in rows:
        mark = "" if value == common else "   <-- diverged"
        lines.append(f"  {name:<{width}}  {value!r}{mark}")
    lines.append("")
    lines.append(f"  not run: {', '.join(f'{w} ({why})' for w, why in skipped) or '(none)'}")
    return "\n".join(lines)


def walks_agree(
    program: Program,
    *,
    answers: dict[Key, Any] | None = None,
    steps: tuple[str, ...] = (),
    walks: Sequence[Walk] = ALL_WALKS,
) -> Agreement:
    """Drive `program` through every walk and assert they agree — returning what they agreed on.

    **Returns the agreement rather than merely asserting it**, so one call answers both questions
    a reader has. Agreement alone is consistency, never correctness: a defect uniform across
    walks passes, and this function would have been green on the `depth-grant:` aliasing it was
    built in response to. The caller therefore asserts on the returned value in the same breath,
    and the pairing is enforced by the signature instead of by convention.

    A walk that cannot run is recorded in `skipped` with its reason and never dropped silently —
    a harness that quietly runs four of six turns a green into a claim about a domain it never
    covered, which is the defect the hand-written tests this replaces already shipped twice.
    `Agreement.ran` is assertable, and a suite that cares should assert it."""
    answers = answers or {}
    observed: list[Ran] = []
    skipped: list[tuple[str, str]] = []
    for walk in walks:
        if walk.name == "absurd" and not pg_ready():
            skipped.append((walk.name, "no Podman test Postgres (just pgt-up)"))
            continue
        try:
            observed.append(walk.run(program, answers, steps))
        except _CannotRun as declined:
            skipped.append((walk.name, str(declined)))

    outcomes = [(r.walk, r.outcome) for r in observed]
    if len(_distinct(outcomes)) > 1:
        raise Divergence(_matrix("outcome", outcomes, skipped))

    keyed = [(r.walk, r.keys) for r in observed if r.keys is not None]
    if len(_distinct(keyed)) > 1:
        raise Divergence(_matrix("keys", keyed, skipped))

    return Agreement(
        outcome=outcomes[0][1],
        keys=keyed[0][1] if keyed else None,
        ran=tuple(r.walk for r in observed),
        skipped=tuple(skipped),
        keys_from=tuple(name for name, _ in keyed),
        observations=tuple(observed),
    )
