"""The ERP port: what the firm asks of a general ledger, whatever product keeps it.

| op         | what it asks                                                   | answered by |
|------------|----------------------------------------------------------------|-------------|
| `PostMemo` | post a credit memo to the invoice's customer                   | `Posted`    |
| `Reverse`  | post the entry that reverses an entry, matched against it      | `Posted`    |
| `Apply`    | match a payment or a memo against an invoice                   | `Applied`   |
| `Unapply`  | release that match, removing it when no other apply holds it   | `Unapplied` |
| `Read`     | what the invoice owes, and the entries on its customer's account | `Statement` |

A post or a reversal lands once per idempotency key: the key names the one entry it made, and a
replay answers that entry with `found` set to `EXISTING`. An apply lands once per key and an
unapply once per apply it `undoes`: each claims the apply's key, committed with the match it makes
or removes, and an op that finds its claim answers from it and moves nothing. An apply refused
because the entry has nothing left or the invoice owes nothing claims the key as refused, so it is
refused again whatever moved since. An apply whose undo's claim is held is `Refused`, so an apply
an expired attempt sent lands nothing after the compensation that undid it. An apply that finds the
entry already matched to the invoice claims its key and holds that match without making it. An undo
releases its apply's holds and removes each match no other apply holds, so an undo naming one key
leaves a match another key holds, and the last undo removes the match whoever made it. An op whose
key names an entry it does not describe, or one since reversed, is `Refused`: two resolutions of
one case may disagree on what the key names, and the first resolution may have been undone. A key
the ERP holds for a record that is no entry raises `Diverged`. An adapter answers only what it
read, and raises `Unknown` when it could not read whether a call landed. A refusal is an answer,
`Refused`, and the books are as they were.

An entry is named by the id its adapter gives it, or by the id of a document the world holds, such
as an invoice or a payment.
"""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, Field, TypeAdapter


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
    undoes: str
    """The key of the apply whose holds this releases."""


class Read(BaseModel):
    """A read lands nothing, so the key it is asked under names nothing."""

    op: Literal["read"] = "read"
    invoice: str


type ErpOp = Annotated[PostMemo | Reverse | Apply | Unapply | Read, Field(discriminator="op")]


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
    """`removed` is whether the undo removed the match, which it does when the apply it names
    held the last claim on it; otherwise the match stands for the keys still holding it."""

    result: Literal["unapplied"] = "unapplied"
    residual: Decimal
    found: Found
    removed: bool


class Refused(BaseModel):
    """Why the op cannot land, and the values that sentence is about, by role."""

    result: Literal["refused"] = "refused"
    reason: str
    about: dict[str, str] = {}


class Statement(BaseModel):
    """What the invoice owes across its installments, and the posted entries on its customer's
    account, oldest first."""

    result: Literal["statement"] = "statement"
    owed: Decimal
    entries: list[Entry]


type ErpResult = Annotated[
    Posted | Applied | Unapplied | Statement | Refused, Field(discriminator="result")
]

OPS: TypeAdapter[ErpOp] = TypeAdapter(ErpOp)
RESULTS: TypeAdapter[ErpResult] = TypeAdapter(ErpResult)


class Unknown(Exception):
    """A call whose outcome could not be read: it may have landed."""


class Diverged(Exception):
    """The ERP holds a key for a record that is no entry."""


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
