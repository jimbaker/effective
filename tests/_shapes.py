"""A recursion shape spelled more than one way, compared on both engines and crashed at every step.

A `Shape` is one row of a conformance table: the spellings, the domain they run against, and what
every run of them must show. `agree` runs each spelling once and compares them; `sweep` crashes
one spelling at each of its checkpoints and holds every crashed run to the same checks as the run
that did not crash.

| check                      | `agree` | `sweep`: base, each crash  | `interleave`, with `answer` |
|----------------------------|---------|----------------------------|-----------------------------|
| the run completes          | each    | each                       | each                        |
| no checkpoint name repeats | each    | each                       | each                        |
| `answer`                   | each    | each                       | each                        |
| `ledger_ids`, when given   | each    | each                       | each                        |
| names outside the ledger   | equal   | equal to the base run's    |                             |
| `calls`, when given        | equal   |                            |                             |
| `count`, when given        |         | the base run's checkpoints |                             |

A row without an `answer` is held to the first spelling's result in `agree` and to the base run's
in `sweep`. `sweep` also holds each crashed run's rows to what the base run's rows say.

`interleave` runs every spelling under every schedule, a named order a `Turnstile` forces, and
reports what each run shows. A row's schedules default to two derived from a first run: the
steps of its innermost gather branches in index order, and the same with every gather's branches
reversed. Whether a row is schedule-independent is the test's prediction to state, so
`interleave` reports and `independent` compares. A row without an `answer` is held to nothing but
ending, since the rows that predict dependence expect runs that differ.
"""

import json
import sys
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from _conformance import Fault, FaultPosition, private
from _schedules import Label, Turnstile, branch_of

from effective.api import Effect
from effective.budget import Grant
from effective.cost import Contract
from effective.govern import Policy, govern
from effective.keys import race_prefix
from effective.keys.frame import (
    RACE_ARM,
    branch_frames,
    is_branch_frame,
    split_frames,
)
from effective.keys.grammar import ARITY_SEPARATOR, TAG_SEPARATOR, TERM_SEPARATOR
from effective.layers import OpLayer, current_placement, op_layer
from effective.ops import Step, WorkflowOp

type Program = Callable[[str], Effect[Any]]


@dataclass(frozen=True)
class Outcome:
    snap: Any
    task: UUID
    run_id: str
    domain: Any


def no_layers(_run_id: str) -> Sequence[OpLayer[Any]]:
    return ()


def governing(policy_for: Callable[[str], Policy]) -> Callable[[str], Sequence[OpLayer[Any]]]:
    """The layers of a run gated by `policy_for(run_id)`, from its run id."""
    return lambda run_id: (govern(policy_for(run_id), gate="spend", run_id=run_id),)


@dataclass(frozen=True)
class Schedule:
    """An order over the ops `label` names."""

    order: tuple[str, ...]
    label: Label


@dataclass(frozen=True)
class Shape:
    """A row of a conformance table.

    | field        | what it is                                                  |
    |--------------|-------------------------------------------------------------|
    | `spellings`  | each way of writing the shape, by name                      |
    | `domain`     | builds the domain a run answers its ops from, fresh per run |
    | `answer`     | the result every run returns                                |
    | `ledger_ids` | a run's ledger ids, sorted, from its run id                 |
    | `count`      | a run's number of checkpoints, from the run                 |
    | `calls`      | what the spellings' domains must agree on                   |
    | `layers`     | the op layers a run's handler holds, from its run id        |
    | `contract`   | the contract a run is spawned under                         |
    | `schedules`  | the orders `interleave` forces, by name; derived when empty |
    | `observe`    | what else a run shows, from the backend and the run         |
    | `stopped`    | the branches a run cut short, from the backend and the run  |
    """

    spellings: Mapping[str, Program]
    domain: Callable[[], Any]
    answer: Callable[[], Any] | None = None
    ledger_ids: Callable[[str], list[str]] | None = None
    count: Callable[[Outcome], int] | None = None
    calls: Callable[[Any], Any] | None = None
    layers: Callable[[str], Sequence[OpLayer[Any]]] = field(default=no_layers)
    contract: Contract = Contract.V0
    schedules: Mapping[str, Schedule] = field(default_factory=dict)
    observe: Callable[[Any, Outcome], Any] | None = None
    stopped: Callable[[Any, Outcome], Collection[tuple[str, ...]]] | None = None


def run(
    backend,
    program: Program,
    domain: Any,
    *,
    fault: Fault | None = None,
    layers: Callable[[str], Sequence[OpLayer[Any]]] = no_layers,
    contract: Contract = Contract.V0,
    wrap: Callable[[Any], Any] | None = None,
    **spawn: Any,
) -> Outcome:
    """One run of `program` to where it stops, as its own task.

    `wrap` wraps the ctx, which is how a row watches the store from inside the run: a race's loser
    is released by the choice landing, and nothing else in the run can say when that was."""
    name, run_id = private("shape"), str(uuid4())
    backend.register(name, program, domain, fault or Fault(), layers(run_id), wrap)
    task = backend.spawn(name, run_id, contract=contract, **spawn)
    return Outcome(backend.run_until_result(task), task, run_id, domain)


def run_shape(backend, shape: Shape, spelling: str, fault: Fault | None = None) -> Outcome:
    return run(
        backend,
        shape.spellings[spelling],
        shape.domain(),
        fault=fault,
        layers=shape.layers,
        contract=shape.contract,
    )


def placed(backend, outcome: Outcome) -> list[str]:
    """The run's checkpoint names that do not carry its run id: every op but the ledger rows."""
    return sorted(k for k in backend.checkpoint_keys(outcome.task) if outcome.run_id not in k)


def ledger_ids(backend, run_id: str) -> list[str]:
    return sorted(backend.ledger_ids(run_id))


def _holds(backend, shape: Shape, outcome: Outcome, answer: Any, why: object) -> None:
    assert outcome.snap.state == "completed", (why, outcome.snap)
    names = backend.checkpoint_keys(outcome.task)
    assert len(names) == len(set(names)), (why, "a checkpoint name repeats")
    assert not any("#" in k for k in names), (why, "an occurrence was minted for a repeat")
    assert outcome.snap.result == answer, why
    if shape.ledger_ids is not None:
        assert ledger_ids(backend, outcome.run_id) == shape.ledger_ids(outcome.run_id), why


def said(backend, outcome: Outcome) -> tuple[str, ...]:
    """What the run's ledger rows SAY, as canonical JSON. Each row's `event_id` names it with the
    run id, so it is left to the comparison of which rows exist."""
    rows = backend.ledger_payloads(outcome.run_id)
    return tuple(sorted(json.dumps(row | {"event_id": None}, sort_keys=True) for row in rows))


def agree(backend, shape: Shape) -> dict[str, Outcome]:
    """Every spelling once: each holds, places the same names, and writes rows that say the
    same."""
    outcomes = {spelling: run_shape(backend, shape, spelling) for spelling in shape.spellings}
    first, *_ = outcomes.values()
    answer = shape.answer() if shape.answer is not None else first.snap.result
    for spelling, outcome in outcomes.items():
        _holds(backend, shape, outcome, answer, spelling)
    assert len({tuple(placed(backend, outcome)) for outcome in outcomes.values()}) == 1
    assert len({said(backend, outcome) for outcome in outcomes.values()}) == 1
    if shape.calls is not None:
        first_calls, *other_calls = (shape.calls(outcome.domain) for outcome in outcomes.values())
        assert all(calls == first_calls for calls in other_calls)
    return outcomes


def sweep(
    backend, shape: Shape, spelling: str, position: FaultPosition = FaultPosition.AFTER_THUNK
) -> None:
    """A crash at `position` of each checkpoint of one spelling converges to the run without
    one."""
    base, answer, names = _base(backend, shape, spelling)
    for name in names:
        fault = _aimed(name, base, position)
        crashed = run_shape(backend, shape, spelling, fault)
        assert not fault.armed, f"the crash at {name!r} never fired"
        _converges(backend, shape, base, crashed, answer, name)


def sweep_pairs(
    backend, shape: Shape, spelling: str, position: FaultPosition = FaultPosition.AFTER_THUNK
) -> int:
    """`sweep` with a crash in each of two attempts: at one checkpoint, then at another the next
    attempt executes again. Returns how many pairs of checkpoints were crashed so.

    Each pair is tried in the order the store lists it and, when the second crash never fires,
    in the other, so a listing out of commit order loses no pair. A pair crashed in neither order
    has its checkpoints in two branches of one gather or race, which one attempt commits past a
    crash in either; any other pair is a failure. A checkpoint is not paired with itself. The
    pairs grow as the square of a run's checkpoints, so a row opts in."""
    base, answer, names = _base(backend, shape, spelling)
    crashed_twice = 0
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            if any(
                _crashes_twice(backend, shape, spelling, base, answer, position, a, b)
                for a, b in ((first, second), (second, first))
            ):
                crashed_twice += 1
            else:
                assert _apart(first, second), f"{first!r} and {second!r} never crashed twice"
    return crashed_twice


def _base(backend, shape: Shape, spelling: str) -> tuple[Outcome, Any, list[str]]:
    """The run a sweep crashes, held to the row, with its answer and its checkpoints."""
    base = run_shape(backend, shape, spelling)
    answer = shape.answer() if shape.answer is not None else base.snap.result
    _holds(backend, shape, base, answer, "base")
    names = backend.checkpoint_keys(base.task)
    if shape.count is not None:
        assert len(names) == shape.count(base)
    return base, answer, names


def _crashes_twice(
    backend,
    shape: Shape,
    spelling: str,
    base: Outcome,
    answer: Any,
    position: FaultPosition,
    first: str,
    second: str,
) -> bool:
    fault = _aimed(first, base, position, then=_aimed(second, base, position))
    crashed = run_shape(backend, shape, spelling, fault)
    assert fault.fired, f"the crash at {first!r} never fired"
    assert backend.task_attempts(crashed.task) == 1 + fault.fired, (first, second)
    _converges(backend, shape, base, crashed, answer, (first, second))
    return fault.fired == 2


def _apart(one: str, other: str) -> bool:
    """Do two checkpoints run in different branches of one gather or race?"""
    for mine, theirs in zip(branch_frames(one), branch_frames(other), strict=False):
        if mine != theirs:
            return (
                is_branch_frame(mine)
                and is_branch_frame(theirs)
                and mine.rpartition(ARITY_SEPARATOR)[0] == theirs.rpartition(ARITY_SEPARATOR)[0]
            )
    return False


def _aimed(name: str, base: Outcome, position: FaultPosition, then: Fault | None = None) -> Fault:
    """A crash at one of `base`'s checkpoints. A ledger row is aimed at by its placement, which no
    other name contains, since its name carries the run id and each crash is a new run."""
    placement, run_id, _ = name.partition(base.run_id)
    if run_id:
        return Fault(on_name=placement, position=position, then=then)
    return Fault(named=name, position=position, then=then)


def _converges(
    backend, shape: Shape, base: Outcome, crashed: Outcome, answer: Any, why: object
) -> None:
    _holds(backend, shape, crashed, answer, why)
    assert placed(backend, crashed) == placed(backend, base), why
    assert said(backend, crashed) == said(backend, base), why


def granted_by(grantor: Callable[[int], Effect[Grant]] | None, depth: int) -> Effect[int]:
    """How many more levels a grant at `depth` gives, for a reference spelling: read here rather
    than through the combinators' own reading, so a defect there reaches one spelling only."""
    if grantor is None:
        return 0
    grant = yield from grantor(depth)
    return 0 if grant.stop else grant.add_depth


def granting_at(depth: int, *, levels: int) -> Callable[[int], Effect[Grant]]:
    """Grants `levels` more when the budget runs out at `depth`, and stops anywhere else. It
    decides from the depth it is asked at, so a replay asks and answers the same."""

    def grantor(asked: int) -> Effect[Grant]:
        yield from ()
        return Grant(add_depth=levels) if asked == depth else Grant(stop=True)

    return grantor


RECORDER_DEPTH = 2 * sys.getrecursionlimit()
ENGINE_DEPTH = sys.getrecursionlimit() + sys.getrecursionlimit() // 10
"""How deep a depth arm runs: the recorder twice the recursion limit, and an engine, which
checkpoints every level, a tenth past it."""


class Ones(Mapping[str, int]):
    """A recorder's responses in which every tool answers 1."""

    def __getitem__(self, key: str) -> int:
        return 1

    def __contains__(self, key: object) -> bool:
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


# --- the schedule axis --------------------------------------------------------------------------


def placed_step(op: WorkflowOp) -> str | None:
    """Order a step by where it runs, which no other op in the run shares."""
    match op, current_placement():
        case Step(), placement if placement is not None:
            return placement.stored()
        case _:
            return None


def _branch_key(frames: tuple[str, ...], rank: Mapping[tuple[str, ...], int], sign: int) -> tuple:
    """Sibling branches of one gather or race by `sign` times their index, and everything else
    in the order a first run reached it."""
    key = []
    for depth, frame in enumerate(frames):
        tag, _marker, coordinates = frame.partition(":")
        if is_branch_frame(frame):
            g, _comma, i = coordinates.partition(",")
            key.append((rank[(*frames[:depth], f"{tag}:{g}")], sign * int(i)))
        else:
            key.append((rank[frames[: depth + 1]], 0))
    return tuple(key)


def branch_schedules(backend, shape: Shape, spelling: str) -> dict[str, Schedule]:
    """The steps of the innermost branches of one run, in index order and with every gather's or
    race's branches reversed. A branch's own steps keep their order, so both are reachable."""
    reached: list[str] = []

    @op_layer
    def spy(op: WorkflowOp):
        if (name := placed_step(op)) is not None:
            reached.append(name)
        return (yield op)

    first = run(
        backend,
        shape.spellings[spelling],
        shape.domain(),
        layers=lambda run_id: (spy, *shape.layers(run_id)),
        contract=shape.contract,
        max_attempts=1,
    )
    # A refused step is in `reached`, since the spy records it before its admission.
    assert first.snap.state in ("completed", "failed"), (
        "the run schedules derive from",
        first.snap,
    )
    groups: dict[tuple[str, ...], list[str]] = {}
    rank: dict[tuple[str, ...], int] = {}
    for name in reached:
        frames, _identity = split_frames(name)
        groups.setdefault(branch_of(name), []).append(name)
        for depth, frame in enumerate(frames):
            tag, _marker, coordinates = frame.partition(":")
            prefix = (*frames[:depth], f"{tag}:{coordinates.partition(',')[0]}")
            rank.setdefault(prefix if is_branch_frame(frame) else frames[: depth + 1], len(rank))
    leaves = [b for b in groups if b and not any(o != b and o[: len(b)] == b for o in groups)]
    assert len(leaves) > 1, f"{spelling} reaches one branch"

    def ordered(sign: int) -> Schedule:
        by = sorted(leaves, key=lambda b: _branch_key(b, rank, sign))
        return Schedule(tuple(name for b in by for name in groups[b]), placed_step)

    return {"index order": ordered(1), "branches reversed": ordered(-1)}


RAN_TO_ITS_END = ("won", "unchosen")
"""The endings of a branch that reached its last op. `refusal`, `stopped` and `raised` end one
where it stands."""


def cut_short_by_a_race(backend, outcome: Outcome) -> frozenset[tuple[str, ...]]:
    """The branches this run's race endings say did not reach their last op, for `Shape.stopped`.

    Named by the BRANCH and not by the op: how far a stopped branch got before the choice landed
    is what a schedule varies, so an op-level rule would predict what is under test. A branch
    carries the frames enclosing its race, whether a scope or a race branch: an `endings` record
    always sits under its own race's frame, so a nested race reads as a scoped one does."""
    cut: set[tuple[str, ...]] = set()
    for key, state in backend.checkpoint_states(outcome.task).items():
        frames, identity = split_frames(key.stored())
        if identity != "endings" or not frames:
            continue
        tag, _marker, ordinal = frames[-1].partition(TAG_SEPARATOR)
        if tag != RACE_ARM:
            continue
        for index, ending in enumerate(state):
            if ending.get("ending") in RAN_TO_ITS_END:
                continue
            branch = race_prefix(int(ordinal), index).removesuffix(TERM_SEPARATOR)
            cut.add((*frames[:-1], branch))
    return frozenset(cut)


def allowed(order: Sequence[str], cut: Collection[tuple[str, ...]]) -> list[str]:
    """The ops of `order` that ran inside a branch the run cut short, which a turnstile may
    legitimately never see."""
    return [op for op in order if any(branch_frames(op)[: len(b)] == tuple(b) for b in cut)]


@dataclass(frozen=True)
class Seen:
    """What one run under one schedule shows."""

    state: str
    answer: Any
    said: tuple[str, ...]
    placed: frozenset[str]
    observed: Any


def interleave(backend, shape: Shape) -> dict[tuple[str, str], Seen]:
    """Every spelling under every schedule, by `(spelling, schedule)`. Each run ends, completed or
    failed, and takes its whole schedule across two sibling branches. A row with an `answer` holds
    every run to `agree`'s checks, so a run that never reaches the property fails its cell.

    A row whose run parks cannot be interleaved: its resume re-admits the ops the turnstile
    already released."""
    first, *_ = shape.spellings
    schedules = shape.schedules or branch_schedules(backend, shape, first)
    seen: dict[tuple[str, str], Seen] = {}
    for spelling in shape.spellings:
        for name, schedule in schedules.items():
            turnstile = Turnstile(schedule.order, schedule.label)
            outcome = run(
                backend,
                shape.spellings[spelling],
                shape.domain(),
                layers=lambda run_id, t=turnstile: (t.layer(), *shape.layers(run_id)),
                contract=shape.contract,
                max_attempts=1,
            )
            assert outcome.snap.state in ("completed", "failed"), (spelling, name, outcome.snap)
            if shape.answer is not None:
                _holds(backend, shape, outcome, shape.answer(), (spelling, name))
            cut = () if shape.stopped is None else shape.stopped(backend, outcome)
            turnstile.check(allowed(schedule.order, cut))
            seen[spelling, name] = Seen(
                state=outcome.snap.state,
                answer=outcome.snap.result,
                said=said(backend, outcome),
                placed=frozenset(placed(backend, outcome)),
                observed=None if shape.observe is None else shape.observe(backend, outcome),
            )
    return seen


def independent(seen: Mapping[tuple[str, str], Seen]) -> bool:
    """Does every run show the same, whichever spelling and schedule it ran?"""
    first, *rest = seen.values()
    return all(other == first for other in rest)
