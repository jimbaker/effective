"""A nested run's address names every machine above it, and an audit names the machine it reads.

ROLE: adversarial. Two tasks of one run each place a child machine at the same state
and visit, under different ancestors. A second task's write of an id already on the record is
dropped without a word, so an address that forgot the ancestors would lose the second child's
rows. Both engines.
"""

from collections.abc import Callable
from enum import StrEnum
from functools import partial
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault

from effective.api import Effect, append_ledger, scoped
from effective.domain import DomainOp
from effective.keys import Name, Run, compose_key
from effective.machine.evidence import CommandRun
from effective.machine.outcomes import Advance, Finish
from effective.machine.spec import Ctx, Report, StateSpec
from effective.machine.trampoline import (
    Placement,
    Under,
    appending_states,
    canonical_violations,
    run_machine,
    running_under,
)
from effective.ops import LedgerRow


class Step(StrEnum):
    WORK = "work"


class Said(StrEnum):
    DONE = "done"


def working(_ctx: Ctx[Step]) -> Effect[Report[Said]]:
    return Report(Said.DONE, summary="worked")
    yield  # pragma: no cover  -- a worker is a generator


SPECS = {Step.WORK: StateSpec(Step.WORK, working)}


class GreenSuite:
    def run(self, _op: DomainOp[Any]) -> Any:
        return CommandRun(exit_code=0)


def placed(ancestor: str, run_id: str) -> Effect[list[str]]:
    session = yield from run_machine(
        Run(run_id),
        "place",
        SPECS,
        lambda _state, _verdict: Finish(),
        start=Step.WORK,
        under=(Under(ancestor, 0), Under(Step.WORK.value, 0)),
    )
    return [session.commit_id.stored(), session.outcome_id.stored()]


def test_two_ancestries_keep_two_addresses_across_tasks(backend):
    run_id = f"p{uuid4().hex}"
    reported: list[str] = []
    for ancestor in ("left", "right"):
        name = compose_key(t"placed:{Name(ancestor)},{Run(run_id)}").stored()
        backend.register(name, partial(placed, ancestor), GreenSuite(), Fault(), [])
        snap = backend.run_until_result(backend.spawn(name, run_id))
        assert snap.state == "completed", snap
        reported += snap.result
    assert len(set(reported)) == 4, reported
    assert backend.ledger_ids(run_id) == reported


class Inner(StrEnum):
    WORK = "work"


def noting(_ctx: Ctx[Inner]) -> Effect[Report[Said]]:
    yield from append_ledger(
        LedgerRow(event_id=compose_key(t"note:{Name('inner')}"), kind="noted")
    )
    return Report(Said.DONE, summary="noted")


def running_inner(ctx: Ctx[Step]) -> Effect[Report[Said]]:
    yield from run_machine(
        Run(ctx.run_id),
        "note",
        {Inner.WORK: StateSpec(Inner.WORK, noting, canonical=True)},
        lambda _state, _verdict: Finish(),
        start=Inner.WORK,
        under=running_under(None, ctx),
    )
    return Report(Said.DONE, summary="ran the inner machine")


def nesting(run_id: str) -> Effect[str]:
    yield from run_machine(
        Run(run_id),
        "outer",
        {Step.WORK: StateSpec(Step.WORK, running_inner, canonical=True)},
        lambda _state, _verdict: Finish(),
        start=Step.WORK,
    )
    return "done"


def test_two_state_types_sharing_a_value_each_claim_their_own_appends(backend):
    """Both machines have a `work` state. The inner one's row carries two `state:work` frames, and
    only the machine placed under the outer WORK owns it."""
    run_id = f"e{uuid4().hex}"
    name = compose_key(t"audited:{Run(run_id)}").stored()
    backend.register(name, nesting, GreenSuite(), Fault(), [])
    task = backend.spawn(name, run_id)
    snap = backend.run_until_result(task)
    assert snap.state == "completed", snap

    noted = [key for key in backend.checkpoint_keys(task) if "ledger;note:inner" in key]
    assert len(noted) == 1, noted
    assert appending_states(Step, noted) == {}
    assert appending_states(Inner, noted, under=Under(Step.WORK.value, 0)) == {Inner.WORK: noted}


class Outer(StrEnum):
    LEFT = "left"
    RIGHT = "right"


def noting_where(outer: Ctx[Outer]) -> Callable[[Ctx[Step]], Effect[Report[Said]]]:
    def noted(_ctx: Ctx[Step]) -> Effect[Report[Said]]:
        note = compose_key(t"note:{Name(outer.state.value)}")
        yield from append_ledger(LedgerRow(event_id=note, kind="noted"))
        return Report(Said.DONE, summary="noted")

    return noted


def running_a_child(ctx: Ctx[Outer]) -> Effect[Report[Said]]:
    yield from run_machine(
        Run(ctx.run_id),
        "child",
        {Step.WORK: StateSpec(Step.WORK, noting_where(ctx))},
        lambda _state, _verdict: Finish(),
        start=Step.WORK,
        under=running_under(None, ctx),
    )
    return Report(Said.DONE, summary="ran a child")


def two_children(run_id: str) -> Effect[str]:
    """LEFT at visit 0 and RIGHT at visit 1 each run a child machine on `Step`."""
    yield from run_machine(
        Run(run_id),
        "parent",
        {state: StateSpec(state, running_a_child, canonical=True) for state in Outer},
        lambda state, _verdict: Advance(Outer.RIGHT) if state is Outer.LEFT else Finish(),
        start=Outer.LEFT,
    )
    return "done"


AUDITED: dict[str, tuple[Placement | None, str | None]] = {
    "left": (Under(Outer.LEFT.value, 0), "left"),
    "right": (Under(Outer.RIGHT.value, 1), "right"),
    "the wrong visit": (Under(Outer.LEFT.value, 1), None),
    "the wrong state": (Under("elsewhere", 0), None),
    "too shallow": (None, None),
    "too deep": ((Under(Outer.LEFT.value, 0), Under(Step.WORK.value, 0)), None),
}


@pytest.mark.parametrize("placed", AUDITED)
def test_an_audit_claims_the_appends_of_the_placement_it_names(backend, placed):
    """Every state and visit of the placement decides, as well as its depth."""
    under, owner = AUDITED[placed]
    run_id = f"a{uuid4().hex}"
    name = compose_key(t"audited-children:{Run(run_id)}").stored()
    backend.register(name, two_children, GreenSuite(), Fault(), [])
    task = backend.spawn(name, run_id)
    assert backend.run_until_result(task).state == "completed"

    notes = [key for key in backend.checkpoint_keys(task) if "ledger;note:" in key]
    assert len(notes) == 2, notes
    claimed = [key for key in notes if owner is not None and key.endswith(f"note:{owner}")]
    specs = {Step.WORK: StateSpec(Step.WORK, working)}
    assert appending_states(Step, notes, under=under) == ({Step.WORK: claimed} if claimed else {})
    violations = canonical_violations(Step, specs, notes, under=under)
    assert sum(map(len, violations.values())) == len(claimed)


def scoping_a_state(_ctx: Ctx[Step]) -> Effect[Report[Said]]:
    """Appends under an ordinary scope that reads like a machine's state frame."""

    def append() -> Effect[None]:
        yield from append_ledger(
            LedgerRow(event_id=compose_key(t"note:{Name('foreign')}"), kind="noted")
        )

    yield from scoped(compose_key(t"state:{Name('foreign')}"), append)
    return Report(Said.DONE, summary="scoped")


def naming_a_state(_ctx: Ctx[Step]) -> Effect[Report[Said]]:
    """Appends a row whose own id reads like a machine's state frame."""
    yield from append_ledger(
        LedgerRow(event_id=compose_key(t"state:{Name('foreign')}"), kind="noted")
    )
    return Report(Said.DONE, summary="named")


FOREIGN = {"an extra scope": scoping_a_state, "an event id": naming_a_state}


@pytest.mark.parametrize("shape", FOREIGN)
def test_a_state_term_outside_the_machine_frames_hides_no_append(backend, shape):
    specs = {Step.WORK: StateSpec(Step.WORK, FOREIGN[shape])}

    def body(run_id: str) -> Effect[str]:
        yield from run_machine(
            Run(run_id), "go", specs, lambda _state, _verdict: Finish(), start=Step.WORK
        )
        return "done"

    run_id = f"s{uuid4().hex}"
    name = compose_key(t"foreign-state:{Run(run_id)}").stored()
    backend.register(name, body, GreenSuite(), Fault(), [])
    task = backend.spawn(name, run_id)
    assert backend.run_until_result(task).state == "completed"

    assert backend.ledger_kinds(run_id) == ["noted", "machine-committed", "machine-finished"]
    notes = [key for key in backend.checkpoint_keys(task) if "foreign" in key]
    assert len(notes) == 1, notes
    assert set(canonical_violations(Step, specs, notes)) == {Step.WORK}
