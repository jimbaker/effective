"""`engage`: ask an outside critic about a proposed resolution, and hear only a cited critique.

The critic answers through its own interpreter: `World` routes an ask whose response schema is
`CritiqueWire` to `World.critic`, so the model that proposed never answers its own review. A
critique must cite documents the case holds or rules the firm holds, and support a credit from $0
to the payment's shortfall; one that does not is refused and asked again with the reason, and
after the last re-prompt the refusal stands.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from functools import partial
from string.templatelib import Template
from typing import assert_never

from pydantic import BaseModel

from effective.api import Effect, ask_llm, scoped
from effective.channels import Field, Gated, Repair, render
from effective.combinators import Converged, Unconverged, fixpoint
from effective.govern import BudgetRefused
from effective.keys import Name, compose_key

from .cases import Case
from .knowledge import K0, Knowledge

REPAIRS = 2
"""Re-prompts after the first critique before a refusal stands."""


class Verdict(StrEnum):
    ACCEPT = "accept"
    REVISE = "revise"


class Proposal(BaseModel):
    """A resolution someone proposed: post a credit memo of `credit` and collect the rest."""

    credit: Decimal
    rationale: str
    cites: tuple[str, ...]


class Critique(BaseModel):
    verdict: Verdict
    credit: Decimal
    """The credit memo the evidence supports, whatever the proposal said."""
    cites: list[str]
    reason: str


class CritiqueWire(Critique):
    """A critique as the critic's model returns it, before the channels check its citations."""


def evidence(case: Case, knowledge: Knowledge) -> Template:
    """The case's documents, then the firm's rules, each by its id."""
    documents = t""
    for document in case.documents:
        documents += t"- {document.id} ({document.kind}):\n{document.text:data}\n"
    for rule in knowledge.rules:
        documents += t"- {rule.id} (firm rule):\n{rule.text:data}\n"
    return documents


def brief(case: Case, proposal: Proposal, knowledge: Knowledge = K0) -> Template:
    """The critic's brief: the invoice, the payment, every document and rule, and the proposal,
    as data."""
    invoice, payment = case.invoice.model_dump_json(), case.payment.model_dump_json()
    proposed = proposal.model_dump_json()
    documents = evidence(case, knowledge)
    verdict = Field(Verdict)
    credit = owed(case)
    held = partial(_held, case, knowledge)
    cites = Gated(list[str], held, "cite at least one document or rule the firm holds")
    reason = Field(str)
    return t"""You review a proposed resolution of an accounts-receivable exception, independently
of whoever proposed it. Check it against the evidence.
Invoice: {invoice:data}
Payment: {payment:data}
Evidence:
{documents}The proposal: {proposed:data}
Say whether to accept or revise the proposal {verdict}, the credit memo amount in dollars the
evidence supports {credit}, the ids of the documents your answer relies on {cites}, and why
{reason}"""


def _held(case: Case, knowledge: Knowledge, cites: list[str]) -> bool:
    return bool(cites) and all(case.cited(id) or knowledge.held(id) for id in cites)


WITHIN = "a credit in whole cents from $0 to the payment's shortfall"
"""What a credit memo's amount satisfies, as the refusal of one that does not states it."""


def owed(case: Case) -> Gated[Decimal]:
    """A credit memo amount the payment's shortfall can absorb."""
    return Gated(Decimal, partial(_within, case), WITHIN)


def _within(case: Case, credit: Decimal) -> bool:
    return 0 <= credit <= case.gap and credit == credit.quantize(Decimal("0.01"))


NOT_ASKED = Repair("not asked")
"""The answer of an ask not yet made."""


@dataclass(frozen=True)
class _Resolved[S]:
    """An answer the channels accepted."""

    value: S


@dataclass(frozen=True)
class _Asked[S]:
    """An ask's attempts so far, and the last answer: refused by its channels, resolved, or
    refused its spend by the budget, which the loop raises again once it ends."""

    asks: int
    answer: Repair | _Resolved[S] | BudgetRefused = NOT_ASKED


def _answered[S](_last: _Asked[S], asked: _Asked[S]) -> bool:
    """Does the ask that made `asked` end the loop: a resolved answer, or a refused spend?"""
    match asked.answer:
        case Repair():
            return False
        case _Resolved() | BudgetRefused():
            return True
        case unreachable:
            assert_never(unreachable)


def asked[S: BaseModel](
    name: str, template: Template, output: type[S], wire: type[BaseModel]
) -> Effect[S | Repair]:
    """Ask `template` as step `name`, re-prompting with the reason while its channels refuse the
    answer, `REPAIRS` times at most, each ask a level of its own under the `repair` scope. `wire`
    is the schema the model answers in, which is how `World` routes the ask. A refused spend
    ends the loop and is raised again."""

    def again(last: _Asked[S]) -> Effect[_Asked[S]]:
        match last:
            case _Asked(answer=Repair(reason=reason)) if last.asks:
                refused = t"\nThe last answer was refused: {reason:data}\nAnswer again."
                prompt = render(template + refused, output=output)
            case _:
                prompt = render(template, output=output)
        try:
            response = yield from ask_llm(name, prompt.messages, wire)
        except BudgetRefused as spent:
            return _Asked(last.asks + 1, spent)
        # lint: totality(total): `S | Repair`, so after the `Repair` arm the capture is `S`.
        match prompt.resolve(response.model_dump(mode="json")):
            case Repair() as repair:
                return _Asked(last.asks + 1, repair)
            case resolved:
                return _Asked(last.asks + 1, _Resolved(resolved))

    loop = partial(fixpoint, _Asked[S](0), again, budget=1 + REPAIRS, converged=_answered)
    match (yield from scoped(compose_key(t"repair:{Name(name)}"), loop)):
        case Converged(value=_Asked(answer=answer)) | Unconverged(value=_Asked(answer=answer)):
            pass
        case unreachable:
            assert_never(unreachable)
    # lint: totality(total): `_Asked.answer`'s three arms, as `_answered` reads them.
    match answer:
        case Repair() as repair:
            return repair
        case _Resolved() as resolved:
            return resolved.value
        case BudgetRefused() as spent:
            raise spent
        case unreachable:
            assert_never(unreachable)


def engage(case: Case, proposal: Proposal, knowledge: Knowledge = K0) -> Effect[Critique | Repair]:
    return asked("engage", brief(case, proposal, knowledge), Critique, CritiqueWire)
