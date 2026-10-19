"""The outcome, scored outside the firm: what the books show after a case's resolutions, against
the disposition each case's evidence supports.

An experiment's driver reads it after the runs; the firm's programs read only their own records.
A resolution's ending says what its run did, and the books say what the case's settlement is, so
a superseded run scores as the disposition its round holds.
"""

from decimal import Decimal

from pydantic import BaseModel

from .cases import Case
from .erp.port import Erp, Kind

GOLD = {"clean": Decimal(0), "discount": Decimal(400), "sla-cap": Decimal(1000)}
"""The credit each case's evidence supports: the right answer, known before any run, that the
books are scored against."""


class Disposition(BaseModel):
    credit: Decimal
    """The credit memos standing on the invoice's account, reversals netted out."""
    gold: Decimal

    @property
    def correct(self) -> bool:
        return self.credit == self.gold

    @property
    def error(self) -> Decimal:
        """Dollars between the credit standing and the gold one."""
        return abs(self.credit - self.gold)


def scored(case: Case, erp: Erp) -> Disposition:
    entries = erp.entries(case.invoice.id)
    reversed_ = {entry.reverses for entry in entries if entry.kind is Kind.REVERSAL}
    credit = sum(
        (e.amount for e in entries if e.kind is Kind.CREDIT_MEMO and e.id not in reversed_),
        Decimal(0),
    )
    return Disposition(credit=credit, gold=GOLD[case.name])
