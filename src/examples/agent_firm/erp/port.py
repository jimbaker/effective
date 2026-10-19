"""The ERP port: what the firm asks of a general ledger, whatever product keeps it.

| op         | what it asks                                                   | answered by |
|------------|----------------------------------------------------------------|-------------|
| `PostMemo` | post a credit memo to the invoice's customer                   | `Posted`    |
| `Reverse`  | post the entry that reverses an entry, matched against it      | `Posted`    |
| `Apply`    | match a payment or a memo against an invoice                   | `Applied`   |
| `Unapply`  | remove that match                                              | `Unapplied` |

A post or a reversal lands once per idempotency key: the key names the one entry it made, and a
replay answers that entry with `found` set to `EXISTING`. An apply or an unapply is read from the
match it makes or removes, so a replay answers `EXISTING` when the match is already as asked; its
key names nothing in the ERP. An op whose key names an entry it does not describe, or one since
reversed, is `Refused`: two resolutions of one case may disagree on what the key names, and the
first may have been undone. A key the ERP holds for a record that is no entry raises `Diverged`.
An adapter answers only what it read, and raises `Unknown` when it could not read whether a call
landed. A refusal is an answer, `Refused`, and the state is as it was.

An entry is named by the id its adapter gives it, or by the id of a document the world holds, such
as an invoice or a payment.
"""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, Field, TypeAdapter


class Found(StrEnum):
    CREATED = "created"
    EXISTING = "existing"


class PostMemo(BaseModel):
    op: Literal["post-memo"] = "post-memo"
    invoice: str
    amount: Decimal
    date: str
    reason: str


class Reverse(BaseModel):
    op: Literal["reverse"] = "reverse"
    entry: str
    reason: str


class Apply(BaseModel):
    op: Literal["apply"] = "apply"
    entry: str
    invoice: str


class Unapply(BaseModel):
    op: Literal["unapply"] = "unapply"
    entry: str
    invoice: str


type ErpOp = Annotated[PostMemo | Reverse | Apply | Unapply, Field(discriminator="op")]


class Posted(BaseModel):
    result: Literal["posted"] = "posted"
    entry: str
    amount: Decimal
    period: str
    found: Found


class Applied(BaseModel):
    """`residual` is what the entry has left to match, after this one."""

    result: Literal["applied"] = "applied"
    residual: Decimal
    found: Found


class Unapplied(BaseModel):
    result: Literal["unapplied"] = "unapplied"
    residual: Decimal
    found: Found


class Refused(BaseModel):
    """Why the op cannot land, and the values that sentence is about, by role."""

    result: Literal["refused"] = "refused"
    reason: str
    about: dict[str, str] = {}


type ErpResult = Annotated[Posted | Applied | Unapplied | Refused, Field(discriminator="result")]

OPS: TypeAdapter[ErpOp] = TypeAdapter(ErpOp)
RESULTS: TypeAdapter[ErpResult] = TypeAdapter(ErpResult)


class Unknown(Exception):
    """A call whose outcome could not be read: it may have landed."""


class Diverged(Exception):
    """The ERP holds a key for a record that is no entry."""


class Kind(StrEnum):
    INVOICE = "invoice"
    PAYMENT = "payment"
    CREDIT_MEMO = "credit-memo"
    REVERSAL = "reversal"
    OTHER = "other"
    """A move the port neither posts nor holds as a document, such as a vendor bill."""


class Entry(BaseModel):
    """A posted entry on a customer's account. A reversal names the entry it reverses."""

    id: str
    kind: Kind
    amount: Decimal
    period: str
    reason: str
    reverses: str | None = None


class Erp(Protocol):
    def perform(self, op: ErpOp, key: str) -> ErpResult: ...

    def entries(self, invoice: str) -> list[Entry]:
        """The posted entries on the account of the invoice's customer, oldest first."""
        ...

    def residual(self, document: str) -> Decimal:
        """What the invoice still owes across its installments, or what the payment or memo has
        left to match."""
        ...

    def keyed(self, invoice: str) -> int:
        """How many idempotency keys name a post or a reversal on the account of the invoice's
        customer."""
        ...
