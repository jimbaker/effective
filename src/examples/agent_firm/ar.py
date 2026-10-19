"""The AR firm's effects as keyed steps through the ERP port, and the saga that settles one
payment against its invoice.

| step            | ERP op or tool | reversed by                                    |
|-----------------|----------------|------------------------------------------------|
| `credit-memo`   | `PostMemo`     | `reverse-credit-memo`, a `Reverse` of the memo |
| `apply-payment` | `Apply`        | `unapply-payment`, an `Unapply`                |
| `apply-memo`    | `Apply`        | `unapply-memo`, an `Unapply`                   |
| `notify`        | `notify`       | nothing: it is the saga's pivot                |
| `collect`       | `collect`      | nothing: retried until it lands                |

The world already holds the payment; applying it matches it against the invoice, and applying the
memo matches the credit, so the invoice owes the gap less the credit. Compensation runs latest
first, so a memo is unapplied before it is reversed, which a matched entry requires. A settlement
reads, then claims, its round on the ledger, so one settlement owns the round and a second
performs nothing. Each step names the fact its effect is about: the case's settlement in its
round, the leg and the attempt, so a step resumed after a crash reaches the ERP or the customer
under the key it was first sent with. A case reopened after a settlement that did not complete is
resolved in the next round.
"""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Context, Decimal, Inexact
from functools import partial
from typing import Any, assert_never

from pydantic import BaseModel

from effective.api import Effect, claim_ledger, read_ledger, step
from effective.combinators import Grantor
from effective.domain import CallTool
from effective.keys import Index, Key, Name, Subject, compose_key
from effective.ops import LedgerRow, Minted, Named

from .cases import Case
from .erp.port import (
    RESULTS,
    Applied,
    Apply,
    ErpOp,
    ErpResult,
    Posted,
    PostMemo,
    Refused,
    Reverse,
    Unapplied,
    Unapply,
)
from .saga import Ended, Landed, Leg, LegKind, Outcome, Rejected, saga

APPLY, UNAPPLY = "apply-payment", "unapply-payment"
"""The step names of the payment's apply and of its undo."""

MEMO, APPLY_MEMO, UNAPPLY_MEMO = "credit-memo", "apply-memo", "unapply-memo"
"""The step names of the memo's post, of its apply, and of the apply's undo."""


TERMS = "settlement-terms"
"""The ledger kind of a round's claimed credit."""

CLAIMANT = "claimant"
"""The tool that names a settlement: it answers with the idempotency key minted for its call."""


def _dollars(amount: Decimal) -> str:
    """`amount` in dollars and cents, as the firm writes it down; an amount finer than a cent
    raises `Inexact`."""
    return str(amount.quantize(Decimal("0.01"), context=Context(traps=[Inexact])))


@dataclass(frozen=True)
class Round:
    """Which settlement of a case a resolution makes: the books it is kept in, and its number. A
    case reopened after a settlement that did not complete is resolved in the next round."""

    books: str
    number: int = 0

    def __post_init__(self) -> None:
        Name(self.books)


class Superseded(BaseModel):
    """A resolution whose round another settlement holds, at `held`: it performs nothing."""

    credit: Decimal
    held: Decimal


type Settlement = Ended | Superseded


def _settlement_of(case: Case, round: Round) -> Key:
    books, invoice, payment = Name(round.books), Subject(case.invoice.id), Subject(case.payment.id)
    number = Index(round.number)
    return compose_key(t"settle:{books},{invoice},{payment},round={number:default=0}")


def about(case: Case, round: Round, leg: str, attempt: int = 0) -> Named:
    """The fact a leg's effect is about: the settlement of `case` in `round`, the leg, and its
    attempt."""
    settlement, named, tried = _settlement_of(case, round), Name(leg), Index(attempt)
    return Named(compose_key(t"leg:{named},attempt={tried:default=0};{settlement:domain=address}"))


def performed(name: str, op: ErpOp, fact: Named) -> Effect[Outcome]:
    call = CallTool(name="erp", args=op.model_dump(mode="json"), result_schema=dict)
    result: ErpResult = RESULTS.validate_python(
        (yield from step(name, call, idempotency_key=fact))
    )
    match result:
        case Posted(entry=entry):
            return Landed(value=entry)
        case Applied() | Unapplied():
            return Landed()
        case Refused(reason=reason):
            return Rejected(reason=reason)
        case unreachable:
            assert_never(unreachable)


def _told(tool: str, args: dict[str, Any], fact: Named) -> Effect[Outcome]:
    """A tool that answers `{"ok": bool, "reason": str}`, read as an outcome."""
    call = CallTool(name=tool, args=args, result_schema=dict)
    match (yield from step(tool, call, idempotency_key=fact)):
        case {"ok": True}:
            return Landed()
        case {"reason": str(reason)}:
            return Rejected(reason=reason)
        case _:
            # the answer itself is the step's checkpoint
            return Rejected(reason="the tool's answer is unreadable")


def settlement(
    case: Case, credit: Decimal, reason: str, round: Round, grantor: Grantor | None = None
) -> list[Leg]:
    """Credit `credit`, apply the payment and then the memo, tell the customer, and queue what is
    still owed, with more attempts at the collection as `grantor` grants them. The notice is the
    settlement's pivot, so a payment in full is told of a $0 credit too."""
    owed = case.gap - credit
    credited, still_owed = _dollars(credit), _dollars(owed)
    invoice, payment = case.invoice.id, case.payment.id
    fact = partial(about, case, round)
    memo = PostMemo(invoice=invoice, amount=credit, date=case.payment.received, reason=reason)
    post_memo = Leg(
        MEMO,
        LegKind.COMPENSABLE,
        lambda _attempt, _landed: performed(MEMO, memo, fact(MEMO)),
        lambda landed: performed(
            "reverse-credit-memo",
            Reverse(entry=landed[MEMO], reason="reversal"),
            fact("reverse-credit-memo"),
        ),
    )
    apply_payment = Leg(
        APPLY,
        LegKind.COMPENSABLE,
        lambda _attempt, _landed: performed(
            APPLY, Apply(entry=payment, invoice=invoice), fact(APPLY)
        ),
        lambda _landed: performed(UNAPPLY, Unapply(entry=payment, invoice=invoice), fact(UNAPPLY)),
    )
    apply_memo = Leg(
        APPLY_MEMO,
        LegKind.COMPENSABLE,
        lambda _attempt, landed: performed(
            APPLY_MEMO, Apply(entry=landed[MEMO], invoice=invoice), fact(APPLY_MEMO)
        ),
        lambda landed: performed(
            UNAPPLY_MEMO, Unapply(entry=landed[MEMO], invoice=invoice), fact(UNAPPLY_MEMO)
        ),
    )
    notify = Leg(
        "notify",
        LegKind.PIVOT,
        lambda _attempt, _landed: _told(
            "notify", {"invoice": invoice, "credit": credited}, fact("notify")
        ),
    )
    collect = Leg(
        "collect",
        LegKind.RETRIABLE,
        lambda attempt, _landed: _told(
            "collect", {"invoice": invoice, "owed": still_owed}, fact("collect", attempt)
        ),
        grantor=grantor,
    )
    credits = [post_memo, apply_payment, apply_memo] if credit else [apply_payment]
    return [*credits, notify, *([collect] if owed else [])]


def settle(
    case: Case, credit: Decimal, reason: str, round: Round
) -> Callable[[], Effect[Settlement]]:
    """The program that claims `round` for this settlement and settles it. The settlement names
    itself by the key minted for its claimant step, which a replay reads back, and the round's
    claim carries that name, so one settlement owns the round and recovers its own crashes. A
    settlement reads the round before it claims, so one that finds the round held by another ends
    `Superseded` before it claims or asks, whatever credit it holds. The claim decides: a round
    claimed after the read supersedes this settlement at the claim, before any leg runs. The claim
    commits the firm to the credit, so it is the op an approval of the credit gates."""
    run = saga(settlement(case, credit, reason, round))
    settled = _settlement_of(case, round)

    def program() -> Effect[Settlement]:
        call = CallTool(name=CLAIMANT, args={}, result_schema=str)
        claimant = yield from step(CLAIMANT, call, idempotency_key=Minted())
        terms = LedgerRow(
            event_id=settled,
            kind=TERMS,
            invoice=case.invoice.id,
            credit=_dollars(credit),
            deducted=_dollars(case.gap),
            broken=[name for name, holds in INVARIANTS.items() if not holds(case)],
            claimant=claimant,
        )

        def superseded(held: dict[str, Any]) -> Superseded | None:
            if held.get("claimant") == claimant:
                return None
            return Superseded(credit=credit, held=Decimal(held["credit"]))

        read = yield from read_ledger(settled)
        if read is not None and (ended := superseded(read)) is not None:
            return ended
        if (ended := superseded((yield from claim_ledger(terms)))) is not None:
            return ended
        return (yield from run())

    return program


INVARIANTS = {
    "same-customer": lambda case: case.payment.customer == case.invoice.customer,
    "paid-in-full": lambda case: case.payment.amount == case.invoice.amount,
}
"""What a payment matched to its invoice satisfies."""
