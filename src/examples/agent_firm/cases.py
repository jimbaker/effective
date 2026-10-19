"""The accounts-receivable cases: an invoice, the payment against it, and the documents that may
explain a difference.

| case       | invoice  | paid     | what explains the gap                                      |
|------------|----------|----------|------------------------------------------------------------|
| `clean`    | $50,000  | $50,000  | nothing: the payment matches                               |
| `discount` | $20,000  | $19,600  | the contract's 2/10 net 30 discount, paid on day 7         |
| `sla-cap`  | $100,000 | $97,000  | a $3,000 outage credit taken; clause 4.2 caps it at $1,000 |
"""

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, StringConstraints

from effective.keys.grammar import NAME

type Id = Annotated[str, StringConstraints(pattern=NAME.pattern)]
"""A document's id as a name in the key grammar, since the firm's keys name the documents."""


class Document(BaseModel):
    """One piece of evidence a resolution may cite, by `id`."""

    id: str
    kind: str
    text: str


class Invoice(BaseModel):
    id: Id
    customer: str
    amount: Decimal
    issued: str
    period: str


class Payment(BaseModel):
    id: Id
    customer: str
    amount: Decimal
    received: str
    memo: str


class Case(BaseModel):
    """An invoice, the payment against it, and the documents that may explain a difference."""

    name: str
    invoice: Invoice
    payment: Payment
    documents: tuple[Document, ...]

    @property
    def gap(self) -> Decimal:
        return self.invoice.amount - self.payment.amount

    def cited(self, id: str) -> bool:
        return any(document.id == id for document in self.documents)


NORTHWIND_MSA = Document(
    id="MSA-NW-2025",
    kind="contract",
    text="""Master Services Agreement, Litware Analytics and Northwind Freight, effective
2025-01-01.
3.1 Fees. Customer pays a monthly platform fee of $100,000, invoiced on the first of the month.
3.2 Payment terms. Net 30. Late balances accrue 1% per month.
4.1 Availability. Provider commits to 99.9% monthly availability of the production platform.
4.2 Service credits. For each full hour of unavailability beyond the commitment in 4.1, Customer
receives a credit of 0.5% of that month's platform fee, not to exceed $1,000 in any calendar month.
Credits are requested in writing within 30 days and are Customer's sole remedy for unavailability.
7.4 Set-off. Customer may not withhold or set off any amount except a credit issued under 4.2.""",
)

SLA_CAP = Case(
    name="sla-cap",
    invoice=Invoice(
        id="INV-1001",
        customer="Northwind Freight",
        amount=Decimal("100000.00"),
        issued="2026-09-01",
        period="2026-09",
    ),
    payment=Payment(
        id="PAY-5501",
        customer="Northwind Freight",
        amount=Decimal("97000.00"),
        received="2026-09-28",
        memo="INV-1001 less SLA credit",
    ),
    documents=(
        NORTHWIND_MSA,
        Document(
            id="CRM-2207",
            kind="crm-note",
            text="""2026-09-26, from Northwind AP (Carol Davis): Paying INV-1001 at $97,000.
Per our SLA we are deducting $3,000 for the September 14 outage, six hours at 0.5% of the
monthly fee per hour. Please confirm the credit memo.""",
        ),
        Document(
            id="TCK-881",
            kind="ticket",
            text="""Incident 2026-09-14, production platform unavailable 02:10 to 08:25 UTC
(6h15m). Root cause: a failed storage migration. Customer-facing status page updated at 02:40.""",
        ),
        Document(
            id="CRM-2210",
            kind="crm-note",
            text="""2026-09-27, account manager: Northwind is renewing in Q1 and the outage hurt.
Keep them happy, but follow the contract.""",
        ),
    ),
)

CLEAN = Case(
    name="clean",
    invoice=Invoice(
        id="INV-1002",
        customer="Contoso Labs",
        amount=Decimal("50000.00"),
        issued="2026-09-01",
        period="2026-09",
    ),
    payment=Payment(
        id="PAY-5502",
        customer="Contoso Labs",
        amount=Decimal("50000.00"),
        received="2026-09-20",
        memo="INV-1002",
    ),
    documents=(),
)

DISCOUNT = Case(
    name="discount",
    invoice=Invoice(
        id="INV-1003",
        customer="Fabrikam Retail",
        amount=Decimal("20000.00"),
        issued="2026-09-01",
        period="2026-09",
    ),
    payment=Payment(
        id="PAY-5503",
        customer="Fabrikam Retail",
        amount=Decimal("19600.00"),
        received="2026-09-08",
        memo="INV-1003 less 2% early payment",
    ),
    documents=(
        Document(
            id="MSA-FR-2024",
            kind="contract",
            text="""3.2 Payment terms. 2/10 net 30: Customer may deduct 2% of an invoice paid
within 10 days of its issue date.""",
        ),
    ),
)

CASES = {case.name: case for case in (SLA_CAP, CLEAN, DISCOUNT)}
