"""The AR firm's effects as keyed steps through the ERP port, and the saga that settles one
payment against its invoice.

| step            | ERP op or tool | reversed by                                    |
|-----------------|----------------|------------------------------------------------|
| `credit-memo`   | `PostMemo`     | `reverse-credit-memo`, a `Reverse` of the memo |
| `apply-payment` | `Apply`        | `unapply-payment`, an `Unapply`                |
| `apply-memo`    | `Apply`        | `unapply-memo`, an `Unapply`                   |
| `notify`        | `notify`       | nothing: it is the saga's pivot                |
| `remind`        | `remind`       | nothing: retried until it lands                |

The world already holds the payment; applying it matches it against the invoice, and applying the
memo matches the credit, so the invoice owes the gap less the credit. Compensation runs latest
first, so a memo is unapplied before it is reversed, which a matched entry requires. A settlement
reads, then claims, its round on the ledger, so one settlement owns the round and a second
performs nothing. Each step names the fact its effect is about: the case's settlement in its
round, the leg and the attempt, so a step resumed after a crash reaches the ERP or the customer
under the key it was first sent with. A settlement appends how it ended once its saga does, and a
round after the first starts only once the round before it has ended.
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
from effective.handlers.durable import named_idempotency_key
from effective.keys import Index, Key, Name, Subject, compose_key
from effective.ops import LedgerRow, Minted, Named

from .cases import WITHIN, Case, in_cents
from .erp.port import (
    RESULTS,
    Applied,
    Apply,
    ErpOp,
    ErpResult,
    Posted,
    PostMemo,
    Read,
    Refused,
    Reverse,
    Statement,
    Unapplied,
    Unapply,
)
from .saga import (
    Compensated,
    Completed,
    Ended,
    Landed,
    Leg,
    LegKind,
    Outcome,
    Rejected,
    Unlanded,
    resumed,
    saga,
)

APPLY, UNAPPLY = "apply-payment", "unapply-payment"
"""The step names of the payment's apply and of its undo."""

MEMO, APPLY_MEMO, UNAPPLY_MEMO = "credit-memo", "apply-memo", "unapply-memo"
"""The step names of the memo's post, of its apply, and of the apply's undo."""

REMIND = "remind"
"""The step, and the tool it calls, that sends the customer a collection notice for what is still
owed."""


TERMS = "settlement-terms"
"""The ledger kind of a round's claimed credit."""

ENDED = "settlement-ended"
"""The ledger kind of how a round's settlement ended, appended after its saga."""

CLAIMANT = "claimant"
"""The tool that names a settlement: it answers with the idempotency key minted for its call."""


def _dollars(amount: Decimal) -> str:
    """`amount` in dollars and cents, as the firm writes it down; an amount finer than a cent
    raises `Inexact`."""
    return str(amount.quantize(Decimal("0.01"), context=Context(traps=[Inexact])))


class Unabsorbed(ValueError):
    """A settlement's credit that its case's shortfall does not absorb."""


@dataclass(frozen=True)
class Round:
    """Which settlement of a case a resolution makes: the books it is kept in, and its number."""

    books: str
    number: int = 0

    def __post_init__(self) -> None:
        Name(self.books)


class Superseded(BaseModel):
    """A resolution whose round another settlement holds, at `held`: it performs nothing."""

    credit: Decimal
    held: Decimal


class Unopened(BaseModel):
    """A settlement whose round may not start, since the round before it, `earlier`, did not end
    compensated: it is still running when `ending` is `None`, and otherwise its case is settled or
    continues in that round. It performs nothing."""

    earlier: int
    ending: str | None


type Settlement = Ended | Superseded | Unopened


def claim_of(case: Case, round: Round) -> Key:
    """Where the ledger holds the claim of `round` for `case`: its terms."""
    books, invoice, payment = Name(round.books), Subject(case.invoice.id), Subject(case.payment.id)
    number = Index(round.number)
    return compose_key(t"settle:{books},{invoice},{payment},round={number:default=0}")


def _ended_of(case: Case, round: Round, continued: int = 0) -> Key:
    """Where the ledger holds how `round`'s settlement ended, or how its `continued`th
    continuation did."""
    books, invoice, payment = Name(round.books), Subject(case.invoice.id), Subject(case.payment.id)
    number, again = Index(round.number), Index(continued)
    return compose_key(
        t"settlement-ended:{books},{invoice},{payment},"
        t"round={number:default=0},continued={again:default=0}"
    )


def _ended_row(case: Case, round: Round, outcome: Ended, continued: int = 0) -> LedgerRow:
    result = outcome.model_dump(mode="json")
    held = _ended_of(case, round, continued)
    return LedgerRow(event_id=held, kind=ENDED, ending=ending(outcome), result=result)


def ending(ended: Ended) -> str:
    """How a settlement ended, as its ledger row names it."""
    match ended:
        case Completed():
            return "completed"
        case Compensated():
            return "compensated"
        case Unlanded():
            return "unlanded"
        case unreachable:
            assert_never(unreachable)


def about(case: Case, round: Round, leg: str, attempt: int = 0, continued: int = 0) -> Named:
    """The fact a leg's effect is about: the settlement of `case` in `round`, the leg, its
    attempt, and the continuation of the settlement that asked it."""
    settlement, named, tried = claim_of(case, round), Name(leg), Index(attempt)
    again = Index(continued)
    return Named(
        compose_key(
            t"leg:{named},attempt={tried:default=0},continued={again:default=0};"
            t"{settlement:domain=address}"
        )
    )


def key_of(fact: Named) -> str:
    """The idempotency key a step about `fact` reaches its tool under."""
    return named_idempotency_key(fact.fact).stored()


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
        case Statement() as statement:
            return Landed(value=statement)
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
    case: Case,
    credit: Decimal,
    reason: str,
    round: Round,
    grantor: Grantor | None = None,
    continued: int = 0,
) -> list[Leg]:
    """Credit `credit`, apply the payment and then the memo, tell the customer, and queue what is
    still owed, with more attempts at the collection as `grantor` grants them. The notice is the
    settlement's pivot, so a payment in full is told of a $0 credit too. A credit the case does not
    absorb raises `Unabsorbed` before any leg exists."""
    if not in_cents(case.gap):
        raise Unabsorbed(f"the shortfall {case.gap} is not in whole cents")
    if not case.absorbs(credit):
        raise Unabsorbed(f"{credit} is not {WITHIN}")
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
        lambda _landed: performed(
            UNAPPLY,
            Unapply(entry=payment, invoice=invoice, undoes=key_of(fact(APPLY))),
            fact(UNAPPLY),
        ),
    )
    apply_memo = Leg(
        APPLY_MEMO,
        LegKind.COMPENSABLE,
        lambda _attempt, landed: performed(
            APPLY_MEMO, Apply(entry=landed[MEMO], invoice=invoice), fact(APPLY_MEMO)
        ),
        lambda landed: performed(
            UNAPPLY_MEMO,
            Unapply(entry=landed[MEMO], invoice=invoice, undoes=key_of(fact(APPLY_MEMO))),
            fact(UNAPPLY_MEMO),
        ),
    )
    notify = Leg(
        "notify",
        LegKind.PIVOT,
        lambda _attempt, _landed: _told(
            "notify", {"invoice": invoice, "credit": credited}, fact("notify")
        ),
    )
    remind = Leg(
        REMIND,
        LegKind.RETRIABLE,
        lambda attempt, _landed: _told(
            REMIND, {"invoice": invoice, "owed": still_owed}, fact(REMIND, attempt, continued)
        ),
        grantor=grantor,
    )
    credits = [post_memo, apply_payment, apply_memo] if credit else [apply_payment]
    return [*credits, notify, *([remind] if owed else [])]


def settle(
    case: Case, credit: Decimal, reason: str, round: Round
) -> Callable[[], Effect[Settlement]]:
    """The program that claims `round` for this settlement, settles it, and appends how it ended.
    The claim carries the name minted for the claimant step, so one settlement owns the round, a
    settlement finding another's claim ends `Superseded`, and the claim is the op an approval of
    the credit gates. A round after the first opens only once the round before it ended
    compensated."""
    run = saga(settlement(case, credit, reason, round))
    settled = claim_of(case, round)
    earlier = Round(round.books, round.number - 1) if round.number else None

    def program() -> Effect[Settlement]:
        if earlier is not None:
            match (yield from _endings(case, earlier)):
                case [*_, {"ending": "compensated"}]:
                    pass
                case [*_, {"ending": str(ended)}]:
                    return Unopened(earlier=earlier.number, ending=ended)
                case _:
                    return Unopened(earlier=earlier.number, ending=None)
        call = CallTool(name=CLAIMANT, args={}, result_schema=str)
        claimant = yield from step(CLAIMANT, call, idempotency_key=Minted())
        terms = LedgerRow(
            event_id=settled,
            kind=TERMS,
            invoice=case.invoice.id,
            credit=_dollars(credit),
            deducted=_dollars(case.gap),
            reason=reason,
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
        outcome = yield from run()
        yield from claim_ledger(_ended_row(case, round, outcome))
        return outcome

    return program


class Uncontinued(BaseModel):
    """A continuation that ran nothing, since the round's latest ending, `ending`, is not
    unlanded, or nobody claimed the round when it is `None`."""

    ending: str | None


def resume(case: Case, round: Round) -> Callable[[], Effect[Ended | Uncontinued]]:
    """The program that continues `round`'s settlement of `case` at the leg that did not land,
    under the round's keys, numbered by the endings already on the ledger, and appends how that
    ended. It reads the round's claim, for the credit and its reason, and the round's endings.
    Only retriable legs remain, so a continuation posts nothing and undoes nothing."""

    def program() -> Effect[Ended | Uncontinued]:
        terms = yield from read_ledger(claim_of(case, round))
        endings = yield from _endings(case, round)
        match terms, endings:
            case {"credit": str(credit), "reason": str(reason)}, [
                *_,
                {"ending": "unlanded", "result": result},
            ]:
                unlanded, continued = Unlanded.model_validate(result), len(endings)
                legs = settlement(case, Decimal(credit), reason, round, continued=continued)
                outcome = yield from resumed(legs, unlanded.landed, unlanded.leg)()
                yield from claim_ledger(_ended_row(case, round, outcome, continued))
                return outcome
            case dict(), [*_, {"ending": str(ended)}]:
                return Uncontinued(ending=ended)
            case _:
                return Uncontinued(ending=None)

    return program


@dataclass(frozen=True)
class Unclaimed:
    """The first round of a case nobody has claimed."""

    round: Round


@dataclass(frozen=True)
class Claimed:
    """A round claimed and not ended."""

    round: Round


@dataclass(frozen=True)
class Done:
    """A round whose settlement, or its last continuation, completed."""

    round: Round


@dataclass(frozen=True)
class Unfinished:
    """A round whose settlement, or its last continuation, ended unlanded."""

    round: Round


type Standing = Unclaimed | Claimed | Done | Unfinished


def _endings(case: Case, round: Round) -> Effect[list[dict[str, Any]]]:
    """How `round`'s settlement ended and how each continuation of it did, in order."""
    endings: list[dict[str, Any]] = []
    while (row := (yield from read_ledger(_ended_of(case, round, len(endings))))) is not None:
        endings.append(row)
    return endings


def standing(case: Case, books: str) -> Effect[Standing]:
    """What the ledger says of `case` in `books`, read round by round: a round that ended
    compensated passes the case to the next."""
    number = 0
    while True:
        round = Round(books, number)
        if (yield from read_ledger(claim_of(case, round))) is None:
            return Unclaimed(round)
        match (yield from _endings(case, round)):
            case []:
                return Claimed(round)
            case [*_, {"ending": "compensated"}]:
                number += 1
            case [*_, {"ending": "completed"}]:
                return Done(round)
            case [*_, {"ending": "unlanded"}]:
                return Unfinished(round)
            case unreadable:
                raise ValueError(f"an ending the ledger cannot hold: {unreadable[-1]}")


def statement(invoice: str) -> Effect[Statement]:
    """The invoice's statement, read as a step, so a replay reads what the first attempt read."""
    call = CallTool(
        name="erp", args=Read(invoice=invoice).model_dump(mode="json"), result_schema=dict
    )
    match RESULTS.validate_python((yield from step("statement", call, idempotency_key=Minted()))):
        case Statement() as read:
            return read
        case unexpected:
            raise ValueError(f"the statement of {invoice} reads as {unexpected}")


INVARIANTS = {
    "same-customer": lambda case: case.payment.customer == case.invoice.customer,
    "paid-in-full": lambda case: case.payment.amount == case.invoice.amount,
}
"""What a payment matched to its invoice satisfies."""
