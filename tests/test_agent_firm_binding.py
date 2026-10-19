"""An approval of a settlement binds the terms it was asked about and the plan that settles them,
on both engines.

| control                         | what holds                                                    |
|---------------------------------|---------------------------------------------------------------|
| approved as asked               | a resume deriving other terms is refused, naming its approver |
| approved as planned             | a resume under a changed plan is refused before any leg       |
| continued as planned            | a continuation under a changed plan is refused before its leg |
| a denial is the approver's      | another policy's refusal is raised, whatever its sentence     |
| who denied                      | a denial names its ruler                                      |
| an approval names what it rules | an approval naming no digest is refused                       |
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition, failure_kind
from _firm import CONTROLLER, FOUNDING, PROPOSER, gate, manifest, resolved, ruling
from _odoo import fresh_round
from pydantic import ValidationError

import examples.agent_firm.ar as ar
from effective.cost import Usage
from effective.govern import GateState, Proceed, Verdict, govern, refuse
from effective.ops import WorkflowOp
from effective.parked import answer
from examples.agent_firm.ar import (
    APPLY_MEMO,
    MEMO,
    REMIND,
    Planned,
    Round,
    Sent,
    claim_of,
    settle,
)
from examples.agent_firm.cases import SLA_CAP
from examples.agent_firm.critic import AS_CLAIMED
from examples.agent_firm.engage import CritiqueWire
from examples.agent_firm.erp.port import Kind
from examples.agent_firm.resolve import (
    APPROVAL,
    NOT_APPROVED,
    approval,
    approved,
    resolve,
    terms_of,
)
from examples.agent_firm.saga import Unlanded
from examples.agent_firm.world import World, scripted

CAPPED = {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025"], "reason": "4.2 caps"}
OK = {"ok": True}


@dataclass
class Firm:
    """A scripted world, what its tools were told, and the round it settles."""

    world: World
    told: list[str] = field(default_factory=list)
    round: Round = field(default_factory=fresh_round)


def _firm(
    notify: Callable[[], None] = lambda: None, remind: Callable[[], dict[str, Any]] = lambda: OK
) -> Firm:
    """A firm whose notice runs `notify` before it lands, and whose collection answers `remind`."""

    def tool(name: str):
        def told(args: Mapping[str, Any]) -> dict[str, Any]:
            firm.told.append(name)
            if name == REMIND:
                return remind()
            notify()
            return OK

        return told

    world = World(
        erp=scripted(),
        llm=lambda op: AS_CLAIMED,
        critic=lambda op: (CritiqueWire.model_validate(CAPPED), Usage()),
        critiques=frozenset({CritiqueWire}),
        tools={"notify": tool("notify"), REMIND: tool(REMIND)},
    )
    firm = Firm(world)
    return firm


def _memos(firm: Firm) -> list[Decimal]:
    entries = firm.world.erp.entries(SLA_CAP.invoice.id)
    return [e.amount for e in entries if e.kind is Kind.CREDIT_MEMO]


UNBOUND = {"answers": {APPROVAL: {"decision": "approve"}}}
"""An approval naming no effect."""


def _refused_by_the_approval(snapshot: Any) -> None:
    """`snapshot` failed on a ruling for another effect, naming the approval policy and its
    ruler."""
    assert (snapshot.state, failure_kind(snapshot)) == ("failed", "Unbound"), snapshot
    assert APPROVAL in str(snapshot.failure), snapshot.failure
    assert CONTROLLER in str(snapshot.failure), snapshot.failure


@pytest.mark.adversarial
def test_a_resume_deriving_other_terms_is_refused_naming_its_approver(backend):
    """Approved for a $1,000 credit, resumed by code that derives $2,000: refused before the
    claim, so nothing posts."""
    firm, run_id, credits = _firm(), f"r-{uuid4().hex[:8]}", [Decimal(1000)]

    def program(_params: Any):
        settled = settle(SLA_CAP, credits[0], "4.2 caps", firm.round)
        return (yield from approved(settled, claim_of(SLA_CAP, firm.round)))

    backend.register(run_id, program, firm.world.domain(), Fault(), (gate(run_id),))
    task_id = backend.spawn(run_id, run_id)
    backend.run_until_result(task_id)
    (park,) = backend.parked(task_id)
    credits[0] = Decimal(2000)
    answer(backend.engine, park, ruling(backend, run_id, park), by=CONTROLLER)
    _refused_by_the_approval(backend.run_until_result(task_id))
    assert (_memos(firm), firm.told) == ([], [])


type Replan = Callable[[list[Planned]], list[Planned]]
"""How the plan a settlement was approved under differs from the plan it resumes under."""


def _without(name: str) -> Replan:
    return lambda plan: [leg for leg in plan if leg.name != name]


def _sending(name: str, undo: bool = False, **args: Any) -> Replan:
    """The leg `name` sending its act, or its undo, with `args` changed."""

    def changed(leg: Planned) -> Planned:
        sent = leg.undo if undo else leg.act
        assert sent is not None
        resent = sent.model_copy(update={"args": sent.args | args})
        return leg.model_copy(update={"undo" if undo else "act": resent})

    return lambda plan: [changed(leg) if leg.name == name else leg for leg in plan]


EARLIER_PLANS: dict[str, Replan] = {
    "a-leg-posting-added": _without(APPLY_MEMO),
    "a-leg-posting-nothing-added": _without(REMIND),
    "an-undo-changed": _sending(MEMO, undo=True, reason="reversed in error"),
    "the-notice-changed": _sending("notify", credit="999.00"),
    "the-reminder-changed": _sending(REMIND, owed="999.00"),
}
"""A settlement's plan when it was approved, against the plan of the code it resumes under."""


@pytest.mark.adversarial
@pytest.mark.parametrize("earlier", EARLIER_PLANS.values(), ids=EARLIER_PLANS.keys())
def test_a_settlement_resumed_under_another_plan_is_refused_before_any_leg(
    backend, monkeypatch, earlier
):
    """Approved under one plan, crashed once its notice was sent, and resumed under a plan that
    differs: refused before a leg runs again, so the ERP and the customer hear nothing more."""
    planned = ar.settlement
    current: list[Replan] = [earlier]
    since: dict[str, Any] = {}

    def settlement(*args: Any) -> list[Planned]:
        return current[0](planned(*args))

    def deployed() -> None:
        current[0] = lambda plan: plan
        since.update(residual=firm.world.erp.residual(SLA_CAP.invoice.id), told=len(firm.told))

    monkeypatch.setattr(ar, "settlement", settlement)
    firm = _firm(deployed)
    crash = Fault(on_name="notify", position=FaultPosition.AFTER_THUNK)
    run = resolved(backend, firm, manifest(FOUNDING, firm.round), fault=crash)
    _refused_by_the_approval(run.snapshot)
    assert firm.world.erp.residual(SLA_CAP.invoice.id) == since["residual"]
    assert len(firm.told) == since["told"], firm.told


def _closed_as_unapproved(op: WorkflowOp, state: GateState) -> Verdict:
    """A policy beside the approval, refusing every claim with the approval's sentence, on the
    controller's word."""
    if terms_of(op) is None:
        return Proceed()
    return refuse(NOT_APPROVED, policy="books-closed", by=CONTROLLER)


@pytest.mark.adversarial
def test_another_policys_refusal_in_the_approvals_words_is_no_denial(backend):
    firm, run_id = _firm(), f"r-{uuid4().hex[:8]}"
    gated = govern(approval(PROPOSER), _closed_as_unapproved, gate="credit", run_id=run_id)
    program = resolve(SLA_CAP, critic=True, round=firm.round)
    backend.register(run_id, lambda _: program(), firm.world.domain(), Fault(), (gated,))
    snapshot = backend.run_until_result(backend.spawn(run_id, run_id))
    assert (snapshot.state, failure_kind(snapshot)) == ("failed", "Refused"), snapshot
    assert _memos(firm) == []


@pytest.mark.adversarial
def test_a_denial_names_its_ruler(backend):
    firm = _firm()
    run = resolved(backend, firm, manifest(FOUNDING, firm.round), decision="deny")
    assert run.result == {"reason": NOT_APPROVED, "ruler": CONTROLLER}


@pytest.mark.adversarial
def test_an_approval_naming_no_effect_is_refused(backend):
    firm = _firm()
    waiting = resolved(backend, firm, manifest(FOUNDING, firm.round), approvers=())
    (park,) = backend.parked(waiting.task_id)
    answer(backend.engine, park, UNBOUND, by=CONTROLLER)
    _refused_by_the_approval(backend.run_until_result(waiting.task_id))
    assert (_memos(firm), firm.told) == ([], [])


@pytest.mark.adversarial
def test_a_continuation_under_another_plan_is_refused_before_its_leg(backend, monkeypatch):
    """A collection out of attempts leaves the settlement unlanded; continued under a plan whose
    collection asks for another amount, the continuation is refused before it asks again."""
    firm = _firm(remind=lambda: {"ok": False, "reason": "AP is closed today"})
    run = resolved(backend, firm, manifest(FOUNDING, firm.round))
    assert Unlanded.model_validate(run.result).leg == REMIND
    told = len(firm.told)
    planned = ar.settlement
    monkeypatch.setattr(
        ar, "settlement", lambda *args: _sending(REMIND, owed="1.00")(planned(*args))
    )
    run_id = f"c-{uuid4().hex[:8]}"
    backend.register(
        run_id, lambda _: ar.resume(SLA_CAP, firm.round)(), firm.world.domain(), Fault(), ()
    )
    snapshot = backend.run_until_result(backend.spawn(run_id, run_id))
    assert (snapshot.state, failure_kind(snapshot)) == ("failed", "Replanned"), snapshot
    assert len(firm.told) == told, firm.told


def test_a_step_sends_arguments_detached_from_its_plan():
    """A tool that changes what it was sent leaves the approved plan as it was."""
    sent = Sent(step="notify", tool="notify", args={"note": {"owed": "2000.00"}})
    args = sent.args_with({})
    args["note"]["owed"] = "999999.00"
    assert sent.args == {"note": {"owed": "2000.00"}}


@pytest.mark.parametrize("value", [b"mail", {"mail"}], ids=["bytes", "a-set"])
def test_a_step_takes_only_json_arguments(value):
    """An argument the digest reads as JSON is sent as JSON, so two plans digesting alike send the
    same values."""
    with pytest.raises(ValidationError):
        Sent(step="notify", tool="notify", args={"payload": value})
