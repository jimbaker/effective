"""The AR firm's resolution on the scripted world, approved through a park, on both engines.

| question | the pin                                                                           |
|----------|-----------------------------------------------------------------------------------|
| R1       | a resolution posts one credit memo and one payment, however a crash lands         |
| R2       | every ordering of every subset of the notes and tickets commits the same effects  |
| R3       | a critique present, absent or uncited parts the run at `engage` and nowhere before |
| R5       | the credit posts only on an approval, and never on one its proposer gave          |
| spine    | the attempt resumed after the approval asks no model                              |
| twice    | a case resolved twice settles once, in sequence or gathered, and asks each time   |
| admitted | a run reads the case its manifest admitted, on every attempt and every replay     |

The R-numbers are the questions in `wiki/concepts/agent-firm.md`.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal, Inexact
from itertools import combinations, permutations
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition, at_every_op
from _firm import CAPPED, CONTROLLER, GENESIS, PROPOSER, Run, manifest, replayed, resolved
from _odoo import ERPS, another_invoice, books, fresh_round
from _schedules import Turnstile, branch_of

from effective.api import gather
from effective.cost import Contract, Usage
from effective.govern import (
    Ask,
    GateState,
    Proceed,
    Verdict,
    answers_for,
    govern,
    park,
)
from effective.handlers.base import step_key
from effective.keys import Index, Key, Name, Tag, compose_key
from effective.layers import current_placement
from effective.lineage import canonical, compare
from effective.ops import AppendLedgerRow, LedgerRow, ReadLedgerRow, Step, WorkflowOp
from effective.parked import answer
from examples.agent_firm.ar import TERMS, Round, Superseded, settle, settlement
from examples.agent_firm.cases import CLEAN, DISCOUNT, SLA_CAP, Case, Document, Payment
from examples.agent_firm.critic import CLAIMED
from examples.agent_firm.engage import REPAIRS, WITHIN, CritiqueWire
from examples.agent_firm.erp.port import Apply, Erp, Kind, Unapply
from examples.agent_firm.manifest import MANIFEST, Manifest
from examples.agent_firm.resolve import (
    APPROVAL,
    ASKED,
    RULED,
    Stopped,
    announced,
    approval,
    resolve,
    terms_of,
)
from examples.agent_firm.saga import RETRIES, Compensated, Completed
from examples.agent_firm.world import World, scripted

UNCITED = CAPPED | {"cites": []}
OK = {"ok": True}


@dataclass
class Firm:
    """A scripted world, and the arguments each of its tools was called with."""

    world: World
    told: list[tuple[str, Mapping[str, Any]]] = field(default_factory=list)
    briefs: list[str] = field(default_factory=list)
    """The critic's brief on each ask, as text."""
    round: Round = field(default_factory=fresh_round)
    """The round of the firm's books its resolutions settle in."""


def _firm(
    critique: dict[str, Any] | None = CAPPED, proposal=CLAIMED, erp: Erp | None = None
) -> Firm:
    def critic(op: Any) -> tuple[Any, Usage]:
        firm.briefs.append(str(op.messages))
        return CritiqueWire.model_validate(critique), Usage()

    def tool(name: str):
        def told(args: Mapping[str, Any]) -> dict[str, Any]:
            firm.told.append((name, args))
            return OK

        return told

    world = World(
        erp=erp or scripted(),
        llm=lambda op: proposal,
        critic=critic,
        critiques=frozenset({CritiqueWire}),
        tools={"notify": tool("notify"), "collect": tool("collect")},
    )
    firm = Firm(world)
    return firm


def _resolved(
    backend, firm: Firm, case: Case = SLA_CAP, *, critic=True, round: Round | None = None, **driven
) -> Run:
    """Resolve `case` in `firm`'s round, or in `round`, through the shared driver."""
    admission = manifest(GENESIS, round or firm.round, critic=critic, case=case)
    return resolved(backend, firm, admission, **driven)


def _terms(case: Case, amount: str, deducted: str) -> dict[str, str]:
    """The ask's detail for a credit of `amount` on `case`'s deduction of `deducted`."""
    return {"invoice": case.invoice.id, "amount": amount, "deducted": deducted}


def _rulers(backend, run_id: str) -> list[str]:
    return [row["ruler"] for row in backend.ledger_payloads(run_id) if row["kind"] == RULED]


def _pairs(backend, run_id: str) -> list[tuple[str, str]]:
    """Each approval row's kind, and its id past its namespace: an ask and its ruling share it."""
    rows = backend.ledger_payloads(run_id)
    return [
        (row["kind"], row["event_id"].split(":", 1)[1])
        for row in rows
        if row["kind"] in (ASKED, RULED)
    ]


def test_r5_each_ruling_pairs_with_its_ask_across_a_re_park(backend):
    run = _resolved(backend, _firm(), approvers=(PROPOSER, CONTROLLER))
    assert _rulers(backend, run.run_id) == [PROPOSER, CONTROLLER]
    pairs = _pairs(backend, run.run_id)
    assert [kind for kind, _ in pairs] == [ASKED, RULED, ASKED, RULED]
    assert pairs[0][1] == pairs[1][1] != pairs[2][1] == pairs[3][1]


def test_the_ledger_names_the_terms_and_who_proposed_it_before_each_approval(backend):
    run = _resolved(backend, _firm(), approvers=(PROPOSER, CONTROLLER))
    terms = _terms(SLA_CAP, "1000.00", "3000.00")
    assert _asked(backend, run.run_id) == [[terms], [terms | {"proposer": PROPOSER}]]


def test_a_gather_of_two_settlements_asks_for_each_settlement_on_the_ledger(backend):
    firm, run_id = _firm(), f"r-{uuid4().hex[:8]}"
    credits = [(SLA_CAP, Decimal(1000)), (DISCOUNT, Decimal(400))]

    def program():
        return (
            yield from gather([settle(case, credit, "r", firm.round) for case, credit in credits])
        )

    gate = govern(approval(PROPOSER), gate="credit", run_id=run_id, announce=announced)
    backend.register(f"two-{run_id}", lambda _: program(), firm.world.domain(), Fault(), (gate,))
    task_id = backend.spawn(f"two-{run_id}", run_id)
    snapshot = backend.run_until_result(task_id)
    while parked := backend.parked(task_id):
        for waiting in parked:
            answer(
                backend.engine,
                waiting,
                {"answers": {APPROVAL: {"decision": "approve"}}},
                by=CONTROLLER,
            )
        snapshot = backend.run_until_result(task_id)
    assert snapshot.state == "completed", snapshot
    asked = [row["detail"] for row in backend.ledger_payloads(run_id) if row["kind"] == ASKED]
    terms = [_terms(SLA_CAP, "1000.00", "3000.00"), _terms(DISCOUNT, "400.00", "400.00")]
    assert sorted((detail for (detail,) in asked), key=str) == sorted(terms, key=str)


def _notice(op: WorkflowOp, state: GateState) -> Verdict:
    """A second policy on the credit gate, parking the notice the approval does not gate."""
    match op:
        case Step(name="notify") if not answers_for(state, "notice"):
            return park(Ask("notice", "send the customer's notice?"))
        case _:
            return Proceed()


def test_a_park_the_approval_did_not_ask_for_is_announced_too(backend):
    firm, run_id = _firm(), f"r-{uuid4().hex[:8]}"
    gate = govern(approval(PROPOSER), _notice, gate="credit", run_id=run_id, announce=announced)
    program = resolve(SLA_CAP, critic=True, round=firm.round)
    backend.register(f"two-{run_id}", lambda _: program(), firm.world.domain(), Fault(), (gate,))
    task_id = backend.spawn(f"two-{run_id}", run_id)
    snapshot = backend.run_until_result(task_id)
    answers = {APPROVAL: {"decision": "approve"}, "notice": {"decision": "send"}}
    while parked := backend.parked(task_id):
        for waiting in parked:
            answer(backend.engine, waiting, {"answers": answers}, by=CONTROLLER)
        snapshot = backend.run_until_result(task_id)
    assert snapshot.state == "completed", snapshot
    rows = [row for row in backend.ledger_payloads(run_id) if row["kind"] == ASKED]
    assert [row["ask"] for row in rows] == [
        "1. [approve-credit] approve this settlement?",
        "1. [notice] send the customer's notice?",
    ]


def _asked(backend, run_id: str) -> list[Any]:
    return [row["detail"] for row in backend.ledger_payloads(run_id) if row["kind"] == ASKED]


def _settled_once(firm: Firm) -> None:
    """The books and the customer show one settlement of `SLA_CAP` at the capped credit."""
    erp, invoice, payment = firm.world.erp, SLA_CAP.invoice.id, SLA_CAP.payment.id
    assert (_memos(firm), erp.keyed(invoice)) == ([Decimal(1000)], 1)
    assert (erp.residual(payment), erp.residual(invoice)) == (0, 2000)
    assert sorted(name for name, _ in firm.told) == ["collect", "notify"]


@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_a_case_resolved_twice_settles_once_and_the_second_asks_nobody(backend, arm):
    firm = _firm(erp=books(arm, [SLA_CAP])())
    first, second = (_resolved(backend, firm) for _ in range(2))
    assert "notify" in Completed.model_validate(first.result).landed
    _settled_once(firm)
    held = Superseded(credit=Decimal(1000), held=Decimal(1000))
    assert Superseded.model_validate(second.result) == held
    assert [len(_asked(backend, run.run_id)) for run in (first, second)] == [1, 0]


@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
@pytest.mark.parametrize("critique", [None, CAPPED | {"credit": "0"}], ids=["3000", "0"])
def test_a_second_resolution_with_another_credit_is_superseded_and_does_nothing(
    backend, arm, critique
):
    """Whether or not the second credit posts a memo, the round holds the first's, and the
    second run reads it before it asks: it asks nobody and performs nothing, no memo, no apply, no
    notice, no collection."""
    firm = _firm(erp=books(arm, [SLA_CAP])())
    _resolved(backend, firm)
    calls = dict(firm.world.calls)
    firm.world.critic = lambda op: (CritiqueWire.model_validate(critique or CAPPED), Usage())
    second = _resolved(backend, firm, critic=critique is not None)
    held = Superseded.model_validate(second.result)
    assert held.held == Decimal(1000) != held.credit
    assert (second.parks, _asked(backend, second.run_id)) == (0, [])
    assert {kind: firm.world.calls[kind] for kind in ("erp", "notify", "collect")} == {
        kind: calls.get(kind, 0) for kind in ("erp", "notify", "collect")
    }
    _settled_once(firm)


@pytest.mark.adversarial
@pytest.mark.parametrize(
    ("critique", "critic"), [(CAPPED, False), (CAPPED | {"credit": "0"}, True)], ids=["3000", "0"]
)
def test_a_denied_credit_leaves_its_round_to_a_corrected_resolution(backend, critique, critic):
    firm = _firm(critique)
    denied = _resolved(backend, firm, critic=critic, decision="deny")
    assert "the settlement was not approved" in str(denied.failure)
    firm.world.critic = lambda op: (CritiqueWire.model_validate(CAPPED), Usage())
    corrected = _resolved(backend, firm)
    assert "notify" in Completed.model_validate(corrected.result).landed
    _settled_once(firm)


@pytest.mark.adversarial
def test_an_unanswered_credit_holds_no_round_and_is_superseded_once_approved(backend):
    """A $3,000 credit parked for approval claims nothing, so the critiqued $1,000 settles in the
    round; approved later, the $3,000 reads the round's $1,000 and does nothing."""
    firm = _firm()
    waiting = _resolved(backend, firm, critic=False, approvers=())
    assert backend.parked(waiting.task_id)
    assert "notify" in Completed.model_validate(_resolved(backend, firm).result).landed
    (park,) = backend.parked(waiting.task_id)
    answer(backend.engine, park, {"answers": {APPROVAL: {"decision": "approve"}}}, by=CONTROLLER)
    late = backend.run_until_result(waiting.task_id)
    assert Superseded.model_validate(late.result) == Superseded(
        credit=Decimal(3000), held=Decimal(1000)
    )
    _settled_once(firm)


def _rejects_first(name: str, firm: Firm, times: int):
    """`name`'s tool, rejecting its first `times` calls and then answering OK."""
    answers = iter([{"ok": False, "reason": "unreachable"}] * times)

    def tool(args: Mapping[str, Any]) -> dict[str, Any]:
        firm.told.append((name, args))
        return next(answers, OK)

    return tool


@pytest.mark.adversarial
@pytest.mark.parametrize(("leg", "times"), [("notify", 1), ("collect", RETRIES)])
def test_a_case_reopened_after_an_unfinished_settlement_settles_in_its_next_round(
    backend, leg, times
):
    """A rejected notice compensates the settlement, and a collection out of attempts leaves it
    unlanded. Resolved again in the same round, it does not land the leg, and asks nothing new of
    it; in the next round it asks again."""
    firm = _firm()
    firm.world.tools = {**firm.world.tools, leg: _rejects_first(leg, firm, times)}
    for _ in range(2):
        unfinished = _resolved(backend, firm).result
        assert leg not in unfinished.get("landed", {}), unfinished
    reopened = _resolved(backend, firm, round=replace(firm.round, number=1))
    assert leg in Completed.model_validate(reopened.result).landed
    assert [name for name, _ in firm.told].count(leg) == times + 1


def _branch_ledger(op: WorkflowOp) -> str | None:
    """A settlement's read or claim of its round, by the gather branch it runs in."""
    match op, current_placement():
        case ReadLedgerRow(), placed if placed is not None and (
            frames := branch_of(placed.stored())
        ):
            return f"read in {frames[-1]}"
        case AppendLedgerRow(claim=True), placed if placed is not None and (
            frames := branch_of(placed.stored())
        ):
            return f"claim in {frames[-1]}"
        case _:
            return None


def _ledger_order(first: int) -> list[str]:
    """Both branches read their round, then claim it, branch `first` first."""
    frame = [compose_key(t"gather:{Index(0)},{Index(branch)}").stored() for branch in (0, 1)]
    claims = [f"claim in {frame[first]}", f"claim in {frame[1 - first]}"]
    return [f"read in {frame[0]}", f"read in {frame[1]}", *claims]


ERP_CALLS = {"lands": 3, "bounces": 6}
"""The ERP calls of one settlement: a memo, two applies, and on a bounced notice the two unapplies
and the reversal."""


@pytest.mark.adversarial
@pytest.mark.parametrize("notice", ERP_CALLS)
@pytest.mark.parametrize("first", [0, 1], ids=["a-claims-first", "b-claims-first"])
def test_two_settlements_of_one_round_gathered_leave_the_round_to_the_first_claim(
    backend, first, notice
):
    """Both read an empty round at one credit. The first claim holds the round and settles it;
    the other ends `Superseded` holding the same credit and makes no ERP call, so nothing it does
    can land between the owner's unapply of the memo and its reversal."""
    firm, run_id = _firm(), f"r-{uuid4().hex[:8]}"
    if notice == "bounces":
        bounced = {"ok": False, "reason": "the address bounced"}
        firm.world.tools = {**firm.world.tools, "notify": lambda args: bounced}
    turnstile = Turnstile(_ledger_order(first), _branch_ledger)

    def program():
        settlements = [settle(SLA_CAP, Decimal(1000), "r", firm.round) for _ in range(2)]
        return (yield from gather(settlements))

    layers = (turnstile.layer(),)
    backend.register(f"two-{run_id}", lambda _: program(), firm.world.domain(), Fault(), layers)
    snapshot = backend.run_until_result(backend.spawn(f"two-{run_id}", run_id))
    assert snapshot.state == "completed", snapshot
    turnstile.check()
    owner, other = snapshot.result[first], snapshot.result[1 - first]
    assert Superseded.model_validate(other) == Superseded(credit=Decimal(1000), held=Decimal(1000))
    assert firm.world.calls["erp"] == ERP_CALLS[notice]
    erp, invoice, payment = firm.world.erp, SLA_CAP.invoice.id, SLA_CAP.payment.id
    if notice == "lands":
        assert "notify" in Completed.model_validate(owner).landed
        _settled_once(firm)
    else:
        assert Compensated.model_validate(owner).undone == [
            "apply-memo",
            "apply-payment",
            "credit-memo",
        ]
        assert (erp.residual(payment), erp.residual(invoice)) == (97000, 100000)
        assert all(erp.residual(entry.id) == 0 for entry in erp.entries(invoice)[2:])


@pytest.mark.adversarial
@pytest.mark.parametrize("position", list(FaultPosition))
def test_two_credits_of_one_round_gathered_settle_one_whatever_a_crash_interrupts(
    backend, position
):
    """Both read an empty round; the claim settles one, and the other ends `Superseded` by it.

    Which credit claims first is the gather's schedule, and the two settlements make different
    numbers of ops, so a crash point is swept up to the longest run seen. A fault that does not
    fire aimed past the end of its own run, which completed under the same assertions."""

    def resolved(fault: Fault) -> None:
        firm, run_id = _firm(), f"r-{uuid4().hex[:8]}"

        def program():
            credits = (Decimal(1000), Decimal(3000))
            return (yield from gather([settle(SLA_CAP, c, "r", firm.round) for c in credits]))

        backend.register(f"two-{run_id}", lambda _: program(), firm.world.domain(), fault, ())
        snapshot = backend.run_until_result(backend.spawn(f"two-{run_id}", run_id))
        assert snapshot.state == "completed", (snapshot, fault)
        (held,) = [Superseded.model_validate(r).held for r in snapshot.result if "held" in r]
        assert len([r for r in snapshot.result if "landed" in r]) == 1, fault
        assert (_memos(firm), firm.world.erp.keyed(SLA_CAP.invoice.id)) == ([held], 1), fault

    unarmed = Fault(None, position=position)
    resolved(unarmed)
    longest, k = unarmed.count, 0
    while (k := k + 1) <= longest:
        fault = Fault(k, position=position)
        resolved(fault)
        assert fault.fired or fault.count < k, f"k={k}: the fault never fired in reach"
        longest = max(longest, fault.count if not fault.fired else longest)


def test_two_resolutions_of_one_case_gathered_settle_once_and_ask_twice(backend):
    firm, run_id = _firm(), f"r-{uuid4().hex[:8]}"

    def program():
        return (
            yield from gather([resolve(SLA_CAP, critic=True, round=firm.round) for _ in range(2)])
        )

    gate = govern(approval(PROPOSER), gate="credit", run_id=run_id, announce=announced)
    backend.register(f"two-{run_id}", lambda _: program(), firm.world.domain(), Fault(), (gate,))
    task_id = backend.spawn(f"two-{run_id}", run_id)
    snapshot = backend.run_until_result(task_id)
    while parked := backend.parked(task_id):
        for waiting in parked:
            approved = {"answers": {APPROVAL: {"decision": "approve"}}}
            answer(backend.engine, waiting, approved, by=CONTROLLER)
        snapshot = backend.run_until_result(task_id)
    assert snapshot.state == "completed", snapshot
    terms = _terms(SLA_CAP, "1000.00", "3000.00")
    assert _asked(backend, run_id) == [[terms], [terms]]
    _settled_once(firm)


def test_two_announcing_gates_on_one_op_write_a_row_each(backend):
    firm, run_id = _firm(), f"r-{uuid4().hex[:8]}"
    gates = tuple(
        govern(approval(PROPOSER), gate=name, run_id=run_id, announce=announced)
        for name in ("credit", "treasury")
    )
    program = resolve(SLA_CAP, critic=True, round=firm.round)
    backend.register(f"two-{run_id}", lambda _: program(), firm.world.domain(), Fault(), gates)
    task_id = backend.spawn(f"two-{run_id}", run_id)
    snapshot = backend.run_until_result(task_id)
    while parked := backend.parked(task_id):
        for waiting in parked:
            approved = {"answers": {APPROVAL: {"decision": "approve"}}}
            answer(backend.engine, waiting, approved, by=CONTROLLER)
        snapshot = backend.run_until_result(task_id)
    assert snapshot.state == "completed", snapshot
    terms = _terms(SLA_CAP, "1000.00", "3000.00")
    assert _asked(backend, run_id) == [[terms], [terms]]


@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_a_case_resolved_again_after_its_memo_was_reversed_lands_on_no_reversed_memo(backend, arm):
    """The first settlement's apply is refused, since the payment is spent on another invoice, so
    its memo is reversed. Once the payment is freed, a second resolution in the round finds it held
    by the first and performs nothing, so the books still hold no credit."""
    erp, other = another_invoice(arm, Decimal(97000))
    payment, invoice = SLA_CAP.payment.id, SLA_CAP.invoice.id
    erp.perform(Apply(entry=payment, invoice=other), uuid4().hex)
    firm = _firm(erp=erp)
    first = _resolved(backend, firm)
    assert Compensated.model_validate(first.result).undone == ["credit-memo"]
    erp.perform(Unapply(entry=payment, invoice=other), uuid4().hex)
    second = _resolved(backend, firm)
    held = Superseded(credit=Decimal(1000), held=Decimal(1000))
    assert Superseded.model_validate(second.result) == held
    assert not firm.told
    assert (erp.residual(payment), erp.residual(invoice)) == (97000, 100000)


@pytest.mark.parametrize("id", ["INV 1001", "INV/1001", "INV:1001", "INV;1", ""])
def test_an_invoice_id_outside_the_key_grammar_is_refused_when_the_case_is_built(id):
    with pytest.raises(ValueError, match="pattern"):
        SLA_CAP.invoice.model_validate(SLA_CAP.invoice.model_dump() | {"id": id})


FIXTURES = {Kind.INVOICE, Kind.PAYMENT}
"""The entries the world holds before the firm acts."""


def _entries(firm: Firm, case: Case = SLA_CAP) -> list[tuple[Kind, Any, str, str]]:
    """The entries the firm posted on the account of the case's customer."""
    entries = firm.world.erp.entries(case.invoice.id)
    return [(e.kind, e.amount, e.period, e.reason) for e in entries if e.kind not in FIXTURES]


def _memos(firm: Firm) -> list[Any]:
    return [amount for kind, amount, *_ in _entries(firm) if kind is Kind.CREDIT_MEMO]


@pytest.mark.adversarial
@pytest.mark.parametrize(
    ("approvers", "parks", "memos"),
    [
        ((PROPOSER,), 1, []),
        ((PROPOSER, PROPOSER), 2, []),
        ((PROPOSER, CONTROLLER), 2, [1000]),
        ((CONTROLLER,), 1, [1000]),
    ],
    ids=["proposer", "proposer-twice", "then-controller", "controller"],
)
def test_r5_an_approval_by_the_proposer_does_not_count(backend, approvers, parks, memos):
    firm = _firm()
    run = _resolved(backend, firm, approvers=approvers)
    assert _memos(firm) == memos
    assert run.parks == parks
    if memos:
        assert Completed.model_validate(run.result).landed.keys() >= {"credit-memo", "collect"}
        assert run.passed[-1] == approvers
    else:
        assert backend.parked(run.task_id), "the settlement waits for another approver"


def test_a_zero_credit_on_a_deduction_is_asked_with_what_the_customer_deducted(backend):
    run = _resolved(backend, _firm(CAPPED | {"credit": "0"}))
    assert run.parks == 1
    assert _asked(backend, run.run_id) == [[_terms(SLA_CAP, "0.00", "3000.00")]]


def test_a_claim_that_does_not_say_which_invariants_its_case_breaks_is_gated():
    terms = LedgerRow(
        event_id=compose_key(t"unknown-deduction:{Name('books')}"),
        kind=TERMS,
        invoice=SLA_CAP.invoice.id,
        credit="1000",
    )
    asked = {"invoice": SLA_CAP.invoice.id, "amount": "1000", "deducted": "unknown"}
    assert terms_of(AppendLedgerRow(row=terms, claim=True)) == asked


def test_another_customers_payment_in_full_is_approved_before_it_settles(backend):
    """No deduction, and still an exception: the payment is from another customer."""
    stray = Payment(
        id="PAY-9999",
        customer="Contoso Labs",
        amount=SLA_CAP.invoice.amount,
        received=SLA_CAP.payment.received,
        memo="",
    )
    case = SLA_CAP.model_copy(update={"payment": stray})
    firm = _firm(proposal=CLAIMED.model_copy(update={"credit": Decimal(0)}))
    run = _resolved(backend, firm, case, critic=False, approvers=())
    assert backend.parked(run.task_id), "the settlement waits for its approval"
    assert _asked(backend, run.run_id) == [[_terms(case, "0.00", "0.00")]]


@pytest.mark.parametrize(
    ("credit", "case"),
    [
        (Decimal("1000.005"), SLA_CAP),
        (
            Decimal(0),
            SLA_CAP.model_copy(
                update={
                    "payment": SLA_CAP.payment.model_copy(
                        update={"amount": SLA_CAP.invoice.amount - Decimal("0.004")}
                    )
                }
            ),
        ),
    ],
    ids=["credit", "deduction"],
)
def test_a_settlement_refuses_an_amount_finer_than_a_cent(credit, case):
    with pytest.raises(Inexact):
        settle(case, credit, "r", fresh_round())


def test_a_settlements_legs_refuse_a_credit_finer_than_a_cent_before_any_runs():
    with pytest.raises(Inexact):
        settlement(SLA_CAP, Decimal("1000.005"), "r", fresh_round())


def test_the_customer_hears_and_owes_in_dollars_and_cents(backend):
    firm = _firm()
    _resolved(backend, firm)
    told = {name: args.get("credit") or args.get("owed") for name, args in firm.told}
    assert told == {"notify": "1000.00", "collect": "2000.00"}


@pytest.mark.adversarial
def test_a_denied_settlement_ends_the_resolution_and_posts_nothing(backend):
    firm = _firm()
    run = _resolved(backend, firm, decision="deny")
    assert "the settlement was not approved" in str(run.failure)
    assert _entries(firm) == []


@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
@pytest.mark.parametrize("position", list(FaultPosition))
def test_r1_a_resolution_posts_once_whatever_a_crash_interrupts(backend, position, arm):
    erp_of = books(arm, [SLA_CAP])
    unarmed = Fault(None, position=position)
    _resolved(backend, _firm(erp=erp_of()), fault=unarmed)
    crashes = 0
    for k, fault in at_every_op(unarmed):
        firm = _firm(erp=erp_of())
        run = _resolved(backend, firm, fault=fault)
        assert Completed.model_validate(run.result), k
        assert [kind for kind, *_ in _entries(firm)] == [Kind.CREDIT_MEMO], k
        assert (_memos(firm), firm.world.erp.keyed(SLA_CAP.invoice.id)) == ([1000], 1), k
        assert firm.world.erp.residual(SLA_CAP.payment.id) == 0, k
        assert _rulers(backend, run.run_id) == [CONTROLLER], k
        crashes += 1
    assert crashes == {FaultPosition.BEFORE_OP: 21, FaultPosition.AFTER_THUNK: 13}[position]


@pytest.mark.adversarial
def test_the_attempt_resumed_after_the_approval_asks_no_model(backend):
    firm = _firm()
    run = _resolved(backend, firm)
    assert run.asked_at_park == {"Proposal": 1, "CritiqueWire": 1}
    assert firm.world.asked == run.asked_at_park


def test_a_matched_payment_settles_with_no_credit_and_asks_nobody(backend):
    firm = _firm()
    run = _resolved(backend, firm, CLEAN)
    assert run.asked_at_park is None
    assert _entries(firm, CLEAN) == []
    assert firm.world.erp.residual(CLEAN.invoice.id) == 0
    assert not firm.world.asked


@pytest.mark.adversarial
@pytest.mark.parametrize(
    ("critique", "memos", "owed"),
    [
        (CAPPED | {"verdict": "accept"}, [1000], "2000"),
        (CAPPED | {"credit": "0"}, [], "3000"),
        (CAPPED | {"credit": "3000"}, [3000], None),
    ],
    ids=["accept", "nothing", "the-gap"],
)
def test_the_critiques_credit_stands_whatever_its_verdict(backend, critique, memos, owed):
    firm = _firm(critique)
    run = _resolved(backend, firm)
    assert _memos(firm) == memos
    assert run.asked_at_park is not None, "every credit on a deduction is approved"
    collected = [Decimal(args["owed"]) for name, args in firm.told if name == "collect"]
    assert collected == ([Decimal(owed)] if owed else [])


@pytest.mark.adversarial
@pytest.mark.parametrize(
    ("critique", "proposed", "asked"),
    [
        (CAPPED | {"credit": "-500"}, CLAIMED, "CritiqueWire"),
        (CAPPED | {"credit": "3000.01"}, CLAIMED, "CritiqueWire"),
        (CAPPED | {"credit": "1000.005"}, CLAIMED, "CritiqueWire"),
        (CAPPED, CLAIMED.model_copy(update={"credit": -1}), "Proposal"),
        (CAPPED, CLAIMED.model_copy(update={"credit": 4000}), "Proposal"),
    ],
    ids=[
        "critic-negative",
        "critic-over-the-gap",
        "critic-past-the-cent",
        "proposal-negative",
        "proposal-over",
    ],
)
def test_a_credit_outside_the_shortfall_is_refused_and_posts_nothing(
    backend, critique, proposed, asked
):
    firm = _firm(critique, proposed)
    run = _resolved(backend, firm)
    assert WITHIN in Stopped.model_validate(run.result).reason
    assert firm.world.asked[asked] == 1 + REPAIRS
    assert firm.world.asked["CritiqueWire"] == (0 if asked == "Proposal" else 1 + REPAIRS)
    assert (_entries(firm), firm.told) == ([], [])


def _keys(backend, run: Run) -> tuple[str, ...]:
    return canonical(backend.checkpoint_keys(run.task_id), scrub=[run.run_id])


@pytest.mark.adversarial
def test_r3_the_critique_parts_the_run_at_engage_and_nowhere_before(backend):
    absent_firm, present_firm, uncited_firm = _firm(), _firm(CAPPED), _firm(UNCITED)
    books = absent_firm.round.books  # one case's arms read one chain of the firm's knowledge
    present_firm.round, uncited_firm.round = Round(books, 1), Round(books, 2)
    absent = _resolved(backend, absent_firm, critic=False)
    present = _resolved(backend, present_firm)
    uncited = _resolved(backend, uncited_firm)
    before = backend.checkpoint_states(absent.task_id)
    for arm in (present, uncited):
        keys = _keys(backend, arm)
        engaged = next(i for i, key in enumerate(keys) if "engage" in key)
        assert compare(_keys(backend, absent), keys).first_divergence == engaged, keys
        states = backend.checkpoint_states(arm.task_id)
        shared = [k for k in states if k.display() in keys[:engaged]]
        assert len(shared) == engaged
        assert {k: states[k] for k in shared} == {k: before[k] for k in shared}
    assert _memos(absent_firm) == [3000]
    assert _memos(present_firm) == [1000]
    read = [key for key in _keys(backend, present) if key.startswith("ledger-read;")]
    assert len(read) == 1, "the settlement reads its round once"
    assert isinstance(Stopped.model_validate(uncited.result), Stopped)
    assert (uncited.asked_at_park, _entries(uncited_firm)) == (None, [])


def _level(loop: str, name: str, level: int) -> Key:
    """Where `loop`'s step `name` runs at `level`."""
    step = step_key(name)
    # lint: terminal-hole — `step` is the `Key` `step_key` mints.
    return compose_key(t"{Tag(loop)}:{Name(name)};d:{Index(level)};{step:domain=identity}")


def test_each_repair_and_retry_is_a_level_of_its_own_scope(backend):
    firm = _firm()
    critiques = iter([UNCITED, CAPPED])
    firm.world.critic = lambda op: (CritiqueWire.model_validate(next(critiques)), Usage())
    run = _resolved(backend, firm)
    looped = [key for key in _keys(backend, run) if "engage" in key or "collect" in key]
    expected = [("repair", "engage", 0), ("repair", "engage", 1), ("retry", "collect", 0)]
    assert looped == [_level(*at).stored() for at in expected]


NOTES = [d for d in SLA_CAP.documents if d.kind in {"crm-note", "ticket"}]
ADDED = Document(id="CRM-9001", kind="crm-note", text="Called Northwind AP; no answer.")
(CONTRACT,) = [d for d in SLA_CAP.documents if d.kind == "contract"]


def _variants() -> list[Case]:
    """`SLA_CAP` with every ordering of every subset of its notes, its tickets and one new note,
    the contract placed at each position among them."""
    pool = [*NOTES, ADDED]
    return [
        SLA_CAP.model_copy(update={"documents": (*ordered[:at], CONTRACT, *ordered[at:])})
        for size in range(len(pool) + 1)
        for chosen in combinations(pool, size)
        for ordered in permutations(chosen)
        for at in range(size + 1)
    ]


def _committed(firm: Firm) -> tuple[Any, ...]:
    """What the firm did, without the idempotency keys, which name the run that did it."""
    told = [
        (name, {k: v for k, v in args.items() if k != "idempotency_key"})
        for name, args in firm.told
    ]
    return _entries(firm), told, dict(firm.world.calls)


@pytest.mark.property
@pytest.mark.adversarial
def test_r2_the_firms_effects_do_not_read_the_notes_and_tickets(backend):
    baseline = _firm()
    _resolved(backend, baseline)
    assert baseline.told
    assert _entries(baseline)
    variants = _variants()
    assert len(variants) == 261
    for case in variants:
        firm = _firm()
        _resolved(backend, firm, case)
        assert _committed(firm) == _committed(baseline), [d.id for d in case.documents]


CAPPED_AT_100 = SLA_CAP.documents[0].model_copy(
    update={
        "text": SLA_CAP.documents[0].text.replace("not to exceed $1,000", "not to exceed $100")
    }
)
"""The `sla-cap` contract with its clause 4.2 cap at $100, which no module case holds."""


def test_a_run_reads_the_case_its_manifest_admitted(backend):
    """The task's params hold the admitted case, and the critic is briefed on its contract."""
    case = SLA_CAP.model_copy(update={"documents": (CAPPED_AT_100, *SLA_CAP.documents[1:])})
    firm = _firm()
    run = _resolved(backend, firm, case)
    assert Manifest.admitted(run.params).case == case
    assert run.params.keys() == {"run_id", MANIFEST}
    (brief,) = firm.briefs
    assert ("not to exceed $100 " in brief, "not to exceed $1,000" in brief) == (True, False)


@pytest.mark.parametrize("contract", list(Contract), ids=str)
def test_a_replay_reads_the_admitted_case_after_the_module_changed(backend, monkeypatch, contract):
    """The replay is driven from the task's params, with every interpreter raising if called."""
    run = _resolved(backend, _firm(), contract=contract)
    monkeypatch.setattr(SLA_CAP, "documents", (CAPPED_AT_100, *SLA_CAP.documents[1:]))
    assert "not to exceed $1,000" in Manifest.admitted(run.params).case.documents[0].text
    assert replayed(backend, run) == run.result
