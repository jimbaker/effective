"""What a refused run leaves on the canonical record, and what its caller can do with it.

ROLE: adversarial. A canonical state appends, then the budget refuses a later ask. The
run stays uncommitted and raises `RunRefused`, carrying what it built; a caller that wants the run
on the record commits it with `committing`. Both engines.
"""

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault

from effective.api import Effect, append_ledger, ask_llm, gather
from effective.combinators import Grant
from effective.cost import Contract, MeteredInterpreter, Usage
from effective.govern import ChildRefused
from effective.handlers.recording import RecordingHandler
from effective.keys import Name, Run, compose_key
from effective.machine.evidence import CommandRun
from effective.machine.outcomes import Exhausted, Finish, Outcome, Park, ParkReason
from effective.machine.spec import Ctx, Evidence, Report, StateSpec
from effective.machine.specs import fuse
from effective.machine.trampoline import (
    RunRefused,
    Session,
    committing,
    run_machine,
    running_under,
)
from effective.ops import CHAIN_GENERATION, LedgerRow

COST = 0.001
LIMIT = 0.0015
"""Between one ask and two, so the refusal lands after the state's row and its first ask."""
ASKS = 3


class Step(StrEnum):
    SPEND = "spend"


class Said(StrEnum):
    DONE = "done"


def finish(_state: Step, _verdict: Said | Exhausted[Step]) -> Outcome[Step]:
    return Finish()


def spending(ctx: Ctx[Step]) -> Effect[Report[Said]]:
    """Appends its row, then asks until the budget refuses."""
    note = compose_key(t"note:{Name('spent')},{Name(ctx.run_id)}")
    yield from append_ledger(LedgerRow(event_id=note, kind="noted"))
    for n in range(ASKS):
        yield from ask_llm(f"ask-{n}", "spend", str)
    return Report(Said.DONE, summary="spent")


SPECS = {Step.SPEND: StateSpec(Step.SPEND, spending, canonical=True)}


def writing(_ctx: Ctx[Step]) -> Effect[Evidence]:
    return Evidence(summary="wrote", tree={"changed.py": "new"})
    yield  # pragma: no cover  -- a worker is a generator


def spending_judge(_ctx: Ctx[Step], _evidence: Evidence) -> Effect[Said]:
    """Asks until the budget refuses, after its worker has returned."""
    for n in range(ASKS):
        yield from ask_llm(f"judge-{n}", "spend", str)
    return Said.DONE


JUDGED = {Step.SPEND: StateSpec(Step.SPEND, fuse(writing, spending_judge))}


def writing_through_a_helper(ctx: Ctx[Step]) -> Effect[Evidence]:
    """A worker that runs a fused helper inline, whose judge is refused, so it never returns."""
    report = yield from fuse(writing, spending_judge)(ctx)
    return Evidence(summary=report.summary, tree=report.tree)


def unreached(_ctx: Ctx[Step], _evidence: Evidence) -> Effect[Said]:
    raise AssertionError("the worker was refused before it returned")
    yield  # pragma: no cover  -- a judge is a generator


HELPED = {Step.SPEND: StateSpec(Step.SPEND, fuse(writing_through_a_helper, unreached))}


def refused_run(run_id: str) -> Effect[Session]:
    return (yield from run_machine(Run(run_id), "spend", SPECS, finish, start=Step.SPEND))


CAUGHT: list[RunRefused] = []
"""The refusals a parent caught, taken while they are still objects."""


@pytest.fixture(autouse=True)
def _fresh_catch():
    CAUGHT.clear()


class Delegating(StrEnum):
    WORK = "work"


type Wrap = Callable[[Callable[[], Effect[Session]]], Effect[Session]]


def delegating(
    wrap: Wrap, specs: Mapping[Step, StateSpec] = SPECS
) -> Callable[[Ctx[Delegating]], Effect[Report[Said]]]:
    """A state that runs the refused machine inside itself, through `wrap`, and catches it."""

    def state(ctx: Ctx[Delegating]) -> Effect[Report[Said]]:
        placed = running_under(None, ctx)

        def child() -> Effect[Session]:
            return run_machine(
                Run(ctx.run_id), "spend", specs, finish, start=Step.SPEND, under=placed
            )

        try:
            yield from wrap(child)
        except RunRefused as refused:
            CAUGHT.append(refused)
            return Report(Said.DONE, summary="the child was refused")
        return Report(Said.DONE, summary="the child finished")

    return state


def bare(child: Callable[[], Effect[Session]]) -> Effect[Session]:
    return (yield from child())


def parent(
    wrap: Wrap, specs: Mapping[Step, StateSpec] = SPECS
) -> Callable[[str], Effect[dict[str, Any]]]:
    def run_parent(run_id: str) -> Effect[dict[str, Any]]:
        session = yield from run_machine(
            Run(run_id),
            "delegate",
            {Delegating.WORK: StateSpec(Delegating.WORK, delegating(wrap, specs), canonical=True)},
            lambda _s, _v: Finish(),
            start=Delegating.WORK,
        )
        concluded = session.concluded
        return {
            "summary": None if concluded is None else concluded.summary,
            "closed": [session.commit_id.stored(), session.outcome_id.stored()],
        }

    return run_parent


def domain() -> MeteredInterpreter:
    return MeteredInterpreter(
        llm=lambda _op: ("ans", Usage(prompt_tokens=1, completion_tokens=1, cost=COST)),
        tools=lambda _op: CommandRun(exit_code=0),
    )


def run(backend, body, attempts: int | None):
    run_id = f"x{uuid4().hex}"
    name = compose_key(t"refused:{Run(run_id)}").stored()
    backend.register(name, body, domain(), Fault(), (), budget_limit=LIMIT, on_exhaust="fail")
    task = backend.spawn(name, run_id, max_attempts=attempts, contract=Contract.V1)
    return run_id, task, backend.run_until_result(task)


def test_a_refused_run_leaves_its_middle_and_no_closing_pair(backend):
    run_id, task, snap = run(backend, refused_run, attempts=None)

    assert snap.state == "failed", snap
    assert "RunRefused" in str(snap.failure), snap.failure
    assert backend.task_attempts(task) == 1
    assert backend.ledger_kinds(run_id) == ["noted"]
    assert not any(key.startswith("artifact:") for key in backend.checkpoint_keys(task))


def test_a_refused_run_carries_what_it_built(backend):
    """What the caller catches: the run parked at the refused state, its ids composed and
    unappended, and its commitment unset, with the refusal that stopped it as the cause."""
    _run_id, _task, snap = run(backend, parent(bare), attempts=1)
    assert snap.state == "completed", snap
    (refused,) = CAUGHT
    session = refused.session
    assert session.stopped == Park(Step.SPEND, ParkReason.REFUSED)
    assert session.turns == ()
    assert session.commitment is None
    assert type(refused.__cause__).__name__ == "BudgetRefused"


def test_a_parent_that_catches_the_refusal_closes_its_own_run(backend):
    run_id, _task, snap = run(backend, parent(bare), attempts=1)

    assert snap.state == "completed", snap
    assert snap.result["summary"] == "the child was refused"
    assert backend.ledger_kinds(run_id) == ["noted", "machine-committed", "machine-finished"]
    # The closing pair is the PARENT's, read from its own account: the child's is absent.
    assert backend.ledger_ids(run_id)[1:] == snap.result["closed"]
    assert "under:" not in "".join(snap.result["closed"])


def test_committing_writes_the_refused_run_before_its_refusal_climbs(backend):
    """The skipped postamble, run by the caller that chose it. The child's pair lands at the ids
    it carried, parked; the refusal still reaches the parent, which closes its own run."""
    run_id, _task, snap = run(backend, parent(committing), attempts=1)

    assert snap.state == "completed", snap
    assert snap.result["summary"] == "the child was refused"
    (refused,) = CAUGHT
    child = [refused.session.commit_id.stored(), refused.session.outcome_id.stored()]
    assert backend.ledger_kinds(run_id) == [
        "noted",
        "machine-committed",
        "machine-parked",
        "machine-committed",
        "machine-finished",
    ]
    assert backend.ledger_ids(run_id)[1:] == child + snap.result["closed"]


def nested(child: Callable[[], Effect[Session]]) -> Effect[Session]:
    return (yield from committing(lambda: committing(child)))


def test_a_refused_run_is_committed_once_however_many_frames_catch_it(backend):
    """The inner frame commits and re-raises the same refusal, so the outer frame catches a run
    that is already on the record. Committing it again appends its ids from a second placement in
    one task, which the store refuses, and a refusal the parent meant to catch fails the task."""
    run_id, _task, snap = run(backend, parent(nested), attempts=1)

    assert snap.state == "completed", snap
    (refused,) = CAUGHT
    assert refused.session.commitment is not None, "the carried run says it was committed"
    assert backend.ledger_kinds(run_id) == [
        "noted",
        "machine-committed",
        "machine-parked",
        "machine-committed",
        "machine-finished",
    ]


def test_a_refused_judge_leaves_the_tree_its_worker_returned(backend):
    """The worker wrote a file and returned; its judge was refused. The refusal carries that tree,
    and `committing` records it."""
    _run_id, _task, snap = run(backend, parent(committing, JUDGED), attempts=1)

    assert snap.state == "completed", snap
    (refused,) = CAUGHT
    assert refused.tree == {"changed.py": "new"}
    commitment = refused.session.commitment
    assert commitment is not None
    assert commitment.files == ("changed.py",)
    assert commitment.changed == ("changed.py",), "the run was seeded empty, so the file is new"

    assert type(refused.__cause__).__name__ == "BudgetRefused"


def test_a_helpers_refused_judge_leaves_the_last_returned_tree(backend):
    """The helper's worker wrote a file and the helper's judge was refused, inside a worker that
    never returned. The state's own worker left nothing, so the run carries the tree it started
    with, and `committing` records that."""
    _run_id, _task, snap = run(backend, parent(committing, HELPED), attempts=1)

    assert snap.state == "completed", snap
    (refused,) = CAUGHT
    assert refused.tree == {}
    commitment = refused.session.commitment
    assert commitment is not None
    assert commitment.files == ()


def refused_branch() -> Effect[Report[Said]]:
    raise ChildRefused("branch refused")
    yield  # pragma: no cover  -- a branch is a generator


def gathering(_ctx: Ctx[Step]) -> Effect[Report[Said]]:
    yield from gather([refused_branch])
    return Report(Said.DONE, summary="gathered")


def test_a_refused_gather_ends_the_visit_as_a_refused_run():
    specs = {Step.SPEND: StateSpec(Step.SPEND, gathering)}
    with pytest.raises(RunRefused) as caught:
        RecordingHandler().run(
            lambda: run_machine(Run("gathered"), "go", specs, finish, start=Step.SPEND)
        )
    assert isinstance(caught.value.__cause__, ExceptionGroup)
    assert caught.value.session.turns == ()


GRANT_REFUSAL = ChildRefused("grantor refused")


def refusing_grantor(_depth: int) -> Effect[Grant]:
    raise GRANT_REFUSAL
    yield  # pragma: no cover  -- a grantor is a generator


def unvisited(_ctx: Ctx[Step]) -> Effect[Report[Said]]:
    raise AssertionError("the grantor answers before the final visit")
    yield  # pragma: no cover


def test_a_grantors_refusal_arrives_as_it_was_raised():
    """It arrives between visits, with no state to park at."""
    specs = {Step.SPEND: StateSpec(Step.SPEND, unvisited)}
    with pytest.raises(ChildRefused) as caught:
        RecordingHandler().run(
            lambda: run_machine(
                Run("granted"),
                "go",
                specs,
                finish,
                start=Step.SPEND,
                budget=0,
                grantor=refusing_grantor,
            )
        )
    assert caught.value is GRANT_REFUSAL


def concluding(_ctx: Ctx[Step]) -> Effect[Report[Said]]:
    return Report(Said.DONE, summary="a later generation concluded")
    yield  # pragma: no cover


def test_a_later_generation_reports_the_ids_it_stored():
    specs = {Step.SPEND: StateSpec(Step.SPEND, concluding)}
    handler = RecordingHandler({"tool:run_suite": CommandRun(exit_code=0)})

    def second_generation() -> Effect[Session]:
        token = CHAIN_GENERATION.set(2)
        try:
            return (yield from run_machine(Run("chained"), "go", specs, finish, start=Step.SPEND))
        finally:
            CHAIN_GENERATION.reset(token)

    match handler.run(second_generation):
        case Session() as session:
            pass
        case parked:
            raise AssertionError(f"the run parked: {parked!r}")
    stored = [row.event_id.stored() for row in handler.ledger]
    assert stored == [session.commit_id.stored(), session.outcome_id.stored()]
    assert all("generation=2" in event_id for event_id in stored)
    assert handler.ledger[-1].get("summary") == "a later generation concluded"
