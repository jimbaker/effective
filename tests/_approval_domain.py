"""A refund-request workflow with an approval suspend: the domain the durable tests drive.

`process_refund` fetches a request by tool, assesses it with one LLM call, records the
assessment, and parks for a human approval when the routing policy asks for review. Its ops,
in order:

| op               | key                       | when                          |
|------------------|---------------------------|-------------------------------|
| `call_tool`      | `tool:fetch_request`      | always                        |
| `store_artifact` | content-addressed         | always                        |
| `ask_llm`        | `assess_request`          | always                        |
| `append_ledger`  | `assessed:{digest}`       | always                        |
| `await_event`    | `review:{digest}`         | routing is review             |
| `append_ledger`  | `reviewed:{digest}`       | routing is review             |
| `append_ledger`  | `committed:{digest}`      | unless the review rejected it |

`process_refund_async` is the same workflow on the coroutine surface.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel
from sqlmodel import Session, col, select

from effective import (
    Effect,
    append_ledger,
    ask_llm,
    await_event,
    call_tool,
    store_artifact,
)
from effective import coroutine_api as co
from effective.domain import AskLLM, CallTool, DomainOp
from effective.handlers.base import digest_id
from effective.keys import Key, Subject, compose_key
from effective.layers import TransientError
from effective.ledger import LedgerEntry
from effective.ops import LedgerRow

FETCH = "fetch_request"
ASSESS = "assess_request"
PROCESSED_KIND = "request_processed"

AMOUNT_THRESHOLD = Decimal("100")
CONFIDENCE_THRESHOLD = 0.80


class RefundRequest(BaseModel):
    request_id: str
    requester: str
    subject: str
    body: str


class Assessment(BaseModel):
    product: str
    amount: Decimal
    confidence: float | None = None
    reasoning: str = ""


class ApprovalEvent(BaseModel):
    decision: Literal["approve", "reject"]
    actor: str
    rationale: str = ""


class RefundDecision(BaseModel):
    status: Literal["committed", "rejected"]
    assessment: Assessment
    approval: ApprovalEvent | None = None


@dataclass(frozen=True)
class Routing:
    decision: Literal["auto", "review"]
    reason: str


def route(assessment: Assessment) -> Routing:
    """Review a low-confidence or large refund; a missing confidence counts as low."""
    if assessment.confidence is None or assessment.confidence < CONFIDENCE_THRESHOLD:
        return Routing("review", "low_confidence")
    if assessment.amount > AMOUNT_THRESHOLD:
        return Routing("review", "amount_threshold")
    return Routing("auto", "ok")


def assessed_id(request_id: str) -> Key:
    return compose_key(t"assessed:{Subject(digest_id(request_id))}")


def review_name(request_id: str) -> Key:
    """The address an approval is emitted to: an await name, not a ledger id."""
    return compose_key(t"review:{Subject(digest_id(request_id))}")


def reviewed_id(request_id: str) -> Key:
    return compose_key(t"reviewed:{Subject(digest_id(request_id))}")


def committed_id(request_id: str) -> Key:
    return compose_key(t"committed:{Subject(digest_id(request_id))}")


def assessed_row(request_id: str, request: RefundRequest, a: Assessment, r: Routing) -> LedgerRow:
    return LedgerRow(
        event_id=assessed_id(request_id),
        kind="assessment",
        request_id=request_id,
        requester=request.requester,
        product=a.product,
        amount=str(a.amount),
        confidence=a.confidence,
        routing=r.decision,
        reason=r.reason,
    )


def reviewed_row(request_id: str, approval: ApprovalEvent) -> LedgerRow:
    return LedgerRow(
        event_id=reviewed_id(request_id),
        kind="review",
        decision=approval.decision,
        actor=approval.actor,
        rationale=approval.rationale,
    )


def committed_row(request_id: str, a: Assessment) -> LedgerRow:
    return LedgerRow(
        event_id=committed_id(request_id),
        kind="commitment",
        product=a.product,
        amount=str(a.amount),
    )


def processed_row(run: str, amount: str = "4.50") -> LedgerRow:
    """A one-row summary of a finished request, the row `rebuild_total` folds."""
    return LedgerRow(
        event_id=compose_key(t"processed:{Subject(run)}"),
        kind=PROCESSED_KIND,
        request_id=run,
        amount=amount,
    )


def _prompt(request: RefundRequest) -> list[dict[str, Any]]:
    return [{"role": "user", "content": request.body}]


def process_refund(request_id: str) -> Effect[RefundDecision]:
    request = yield from call_tool(FETCH, {"id": request_id}, RefundRequest)
    yield from store_artifact(request.body, "text/plain")
    assessment = yield from ask_llm(ASSESS, _prompt(request), Assessment)
    routing = route(assessment)
    yield from append_ledger(assessed_row(request_id, request, assessment, routing))

    if routing.decision == "review":
        approval = yield from await_event(review_name(request_id), ApprovalEvent)
        yield from append_ledger(reviewed_row(request_id, approval))
        if approval.decision == "reject":
            return RefundDecision(status="rejected", assessment=assessment, approval=approval)

    yield from append_ledger(committed_row(request_id, assessment))
    return RefundDecision(status="committed", assessment=assessment)


async def process_refund_async(request_id: str) -> RefundDecision:
    request = await co.call_tool(FETCH, {"id": request_id}, RefundRequest)
    await co.store_artifact(request.body, "text/plain")
    assessment = await co.ask_llm(ASSESS, _prompt(request), Assessment)
    routing = route(assessment)
    await co.append_ledger(assessed_row(request_id, request, assessment, routing))

    if routing.decision == "review":
        approval = await co.await_event(review_name(request_id), ApprovalEvent)
        await co.append_ledger(reviewed_row(request_id, approval))
        if approval.decision == "reject":
            return RefundDecision(status="rejected", assessment=assessment, approval=approval)

    await co.append_ledger(committed_row(request_id, assessment))
    return RefundDecision(status="committed", assessment=assessment)


def sample_request(request_id: str = "r1", body: str = "Widget, refund $42.00") -> RefundRequest:
    return RefundRequest(
        request_id=request_id, requester="submitter@example.com", subject="Refund", body=body
    )


class CannedDomain:
    """Typed canned values; records each thunk that actually runs, for exactly-once checks."""

    def __init__(self, amount: str = "42.00", flaky_assess: bool = False) -> None:
        self.amount = Decimal(amount)
        self.flaky_assess = flaky_assess  # raise TransientError on the FIRST assessment
        self.calls: list[str] = []

    def run(self, op: DomainOp) -> object:
        match op:
            case CallTool(name="fetch_request"):
                self.calls.append(FETCH)
                return sample_request("x")
            case AskLLM():
                self.calls.append(ASSESS)
                if self.flaky_assess and self.calls.count(ASSESS) == 1:
                    raise TransientError("flaky assessment (first attempt)")
                return Assessment(product="Widget", amount=self.amount, confidence=0.95)
        raise AssertionError(f"unexpected op {op!r}")


def rebuild_total(session: Session, workflow_run_id: str) -> Decimal | None:
    """Fold a run's canonical `request_processed` rows into a total; None for an empty lineage."""
    entries = session.exec(
        select(LedgerEntry).where(
            col(LedgerEntry.workflow_run_id) == workflow_run_id,
            col(LedgerEntry.kind) == PROCESSED_KIND,
            col(LedgerEntry.hypothetical).is_(False),
        )
    ).all()
    if not entries:
        return None
    return sum((Decimal(e.payload["amount"]) for e in entries), Decimal(0))
