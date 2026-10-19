"""`resolve`: take one payment from its invoice to a settled ERP, with a critic and an approver.

| stage     | what runs                                                                |
|-----------|--------------------------------------------------------------------------|
| observe   | `reconcile` holds the case to `INVARIANTS`                               |
| explain   | the firm's model proposes a resolution of an exception                   |
| engage    | an outside critic reviews the proposal, when the firm has one            |
| decide    | the critique's credit stands over the proposal's; a refused answer stops |
| authorize | `approval` parks the credit for a principal other than the proposer      |
| act       | `settle` posts the credit and the payment by saga                        |

A matched case settles with no credit and asks nobody. Every model call is a step, so a resolution
resumed after its approval replays them all and asks none again.
"""

from collections.abc import Callable
from decimal import Decimal
from enum import StrEnum
from string.templatelib import Template
from typing import Any, Literal, assert_never

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from effective.api import Effect
from effective.channels import Field, Repair
from effective.govern import (
    Ask,
    GateState,
    Park,
    Policy,
    Proceed,
    answers_for,
    fuse_prompt,
    park,
    refuse,
    ruler_of,
)
from effective.govern import Verdict as Gating
from effective.keys import Index, Key, Name, Run, Segment, compose_key
from effective.ops import AppendLedgerRow, LedgerRow, WorkflowOp

from .ar import INVARIANTS, TERMS, Round, Settlement, settle
from .cases import Case
from .engage import Critique, Proposal, asked, engage, evidence, owed
from .knowledge import K0, Knowledge, load_knowledge
from .manifest import Manifest
from .reconcile import reconcile

APPROVAL = "approve-credit"
"""The policy an approver answers, and the name its answer comes back under."""

ASKED = "approval_asked"
"""The kind of the ledger row that carries a park's ask to whoever answers it."""

RULED = "approval_ruled"
"""The kind of the ledger row that records who answered a park, and what."""


class Stage(StrEnum):
    PROPOSAL = "proposal"
    CRITIQUE = "critique"


class Stopped(BaseModel):
    """A resolution that posted nothing, because the answer at `stage` was refused."""

    stage: Stage
    reason: str


type Resolved = Settlement | Stopped


def proposal_brief(case: Case, broken: tuple[str, ...], knowledge: Knowledge) -> Template:
    invoice, payment = case.invoice.model_dump_json(), case.payment.model_dump_json()
    documents = evidence(case, knowledge)
    credit, rationale, cites = owed(case), Field(str), Field(tuple[str, ...])
    return t"""This payment does not match its invoice; it breaks {", ".join(broken):data}.
Invoice: {invoice:data}
Payment: {payment:data}
Evidence:
{documents}Propose the credit memo in dollars that resolves the difference {credit}, why
{rationale}, and the ids of the documents you rely on {cites}"""


def explainer(
    knowledge: Knowledge,
) -> Callable[[Case, tuple[str, ...]], Effect[Proposal | Repair]]:
    """The firm's model, asked to explain an exception with the firm's rules in hand."""

    def explain(case: Case, broken: tuple[str, ...]) -> Effect[Proposal | Repair]:
        return asked("explain", proposal_brief(case, broken, knowledge), Proposal, Proposal)

    return explain


def decided(proposal: Proposal | Repair, critique: Critique | Repair | None) -> Proposal | Stopped:
    """The resolution the firm acts on: the critique's credit whatever its verdict, since the
    critique is the evidence's reading, and the proposal's when there is no critique."""
    match proposal, critique:
        case Repair(reason=reason), _:
            return Stopped(stage=Stage.PROPOSAL, reason=reason)
        case _, Repair(reason=reason):
            return Stopped(stage=Stage.CRITIQUE, reason=reason)
        case Proposal(), Critique(credit=credit, cites=cites, reason=reason):
            return Proposal(credit=credit, rationale=reason, cites=tuple(cites))
        case Proposal(), None:
            return proposal
        case unreachable:
            assert_never(unreachable)


def resolve(
    case: Case, *, critic: bool, round: Round, knowledge: Knowledge = K0
) -> Callable[[], Effect[Resolved]]:
    """The program that resolves `case` in `round` with the firm's `knowledge`, engaging the
    firm's critic when it has one."""

    def program() -> Effect[Resolved]:
        reconciled = yield from reconcile([case], INVARIANTS, explainer(knowledge))
        if not reconciled.exceptions:
            return (yield from settle(case, Decimal(0), "matched", round)())
        (exception,) = reconciled.exceptions
        proposal = exception.explanation
        match proposal:
            case Proposal() if critic:
                critique = yield from engage(case, proposal, knowledge)
            case _:
                critique = None
        match decided(proposal, critique):
            case Stopped() as stopped:
                return stopped
            case Proposal(credit=credit, rationale=rationale):
                return (yield from settle(case, credit, rationale, round)())
            case unreachable:
                assert_never(unreachable)

    return program


def admitted(manifest: Manifest) -> Callable[[], Effect[Resolved]]:
    """The program that resolves what `manifest` admitted, with the snapshot of the firm's
    knowledge it pins."""
    case, critic, round, pinned = (
        manifest.case,
        manifest.critic,
        manifest.round,
        manifest.knowledge,
    )

    def program() -> Effect[Resolved]:
        knowledge = yield from load_knowledge(round.books, pinned)
        return (yield from resolve(case, critic=critic, round=round, knowledge=knowledge)())

    return program


def terms_of(op: WorkflowOp) -> dict[str, str] | None:
    """The terms `op` would commit the firm to on an exception: the invoice, the credit and what
    the customer deducted, a $0 credit included. `None` for the claim of a case that breaks no
    invariant, which no approval gates. A settlement commits to its credit when it claims its
    round, so the claim is what an approval gates, and a credit nobody approved holds no round. A
    claim that does not say which invariants its case breaks is gated as one that breaks some."""
    match op:
        case AppendLedgerRow(row=row) if row.kind == TERMS and row.get("broken") != []:
            return {
                "invoice": row.get("invoice", ""),
                "amount": row.get("credit", ""),
                "deducted": row.get("deducted", "unknown"),
            }
        case _:
            return None


def approval(proposer: Segment) -> Policy:
    """A policy that parks a settlement's claim of its credit for approval. An approval
    `proposer` gave does not count, so the claim parks again for another approver; an answer that
    does not approve is refused, which ends the resolution with its round unclaimed."""

    def approve(op: WorkflowOp, state: GateState) -> Gating:
        if (terms := terms_of(op)) is None:
            return Proceed()
        match answers_for(state, APPROVAL), ruler_of(state, APPROVAL):
            case [], _:
                return park(Ask(APPROVAL, "approve this settlement?", terms))
            case _, ruler if ruler == proposer:
                prompt = "approve this settlement? its proposer's approval does not count"
                return park(Ask(APPROVAL, prompt, terms | {"proposer": proposer}))
            case [*_, {"decision": "approve"}], _:
                return Proceed()
            case _:
                return refuse("the settlement was not approved")

    return approve


def _approval(row: Literal["asked", "ruled"], state: GateState) -> Key:
    """Where an approval's rows live: the gate, the run, its generation, the pass, and where the
    gated op runs, its gather branch included; a gate no walk places names it by the park."""
    gate, run, generation = Name(state.gate), Run(state.run_id), Index(state.generation)
    passed = Index(state.pass_n)
    where = state.park_name if state.placement is None else state.placement
    match row:
        case "asked":
            return compose_key(
                t"approval-asked:{gate},{run},generation={generation:default=0},"
                t"pass-n={passed:default=0};{where:domain=any}"
            )
        case "ruled":
            return compose_key(
                t"approval-ruled:{gate},{run},generation={generation:default=0},"
                t"pass-n={passed:default=0};{where:domain=any}"
            )
        case unreachable:
            assert_never(unreachable)


def announced(parked: Park, state: GateState) -> WorkflowOp:
    """The park's ask as a ledger row, written before the run parks: the prompt, and each ask's
    detail, so whoever answers reads the terms, the invoice, the credit and the deduction, before
    ruling on them.

    A ledger's ids are unique across runs, so the row is named by the gate, the run, its
    generation and the pass, and by where the gated op runs: two resolutions gathered in one run
    write two rows, and two gates on one op a row each."""
    return AppendLedgerRow(
        row=LedgerRow(
            event_id=_approval("asked", state),
            kind=ASKED,
            ask=fuse_prompt(parked),
            detail=[dict(ask.detail) for ask in parked.asks],
        )
    )


def ruled(parked: Park, state: GateState, by: Segment | None, answer: Any) -> WorkflowOp:
    """Who ruled on a park, and what, as a ledger row under its ask's coordinates."""
    return AppendLedgerRow(
        row=LedgerRow(
            event_id=_approval("ruled", state),
            kind=RULED,
            ruler=by,
            answer=to_jsonable_python(answer),
        )
    )
