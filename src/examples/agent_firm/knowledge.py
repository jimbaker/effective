"""The firm's knowledge K: a snapshot of its rules, read at a recorded step and advanced by a
recorded adoption.

A run reads the snapshot its manifest pins, and the step's recorded result is the snapshot's
contents, so a replay reads the rules the first attempt read whatever the store has gained since.
Adoptions are ledger rows by height in the firm's books: the first adopter to claim a height
holds it, and the head is the last height claimed.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from effective.api import Effect, claim_ledger, read_ledger, step
from effective.domain import CallTool
from effective.handlers.base import content_digest
from effective.keys import Index, Key, Name, compose_key
from effective.ops import LedgerRow, Unretryable

from .cases import Id

ADOPTED = "knowledge-adopted"
"""The ledger kind of an adoption."""


class Rule(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: Id
    text: str


class Knowledge(BaseModel):
    """The firm's rules, which every brief carries and a critique may cite."""

    model_config = ConfigDict(frozen=True)

    rules: tuple[Rule, ...] = ()

    @property
    def digest(self) -> str:
        return content_digest(self.model_dump(mode="json"))

    def held(self, id: str) -> bool:
        return any(rule.id == id for rule in self.rules)


SLA_CAP_RULE = Rule(
    id="POL-SLA-1",
    text="An SLA service credit is capped by the contract's service-credit clause: credit no more "
    "than the cap, whatever the customer deducted.",
)

K0 = Knowledge()
"""The firm's genesis: no rules beyond its contracts. Every chain of adoptions starts from its
digest, so a change to it unlinks every chain at height 1."""

K1 = Knowledge(rules=(SLA_CAP_RULE,))
"""K0 with the SLA rule adopted."""


class Pinned(BaseModel):
    """A snapshot by its height in the books' chain of adoptions, and its digest. A run checks
    the digest against what it loads; the height is the driver's word for where it folded."""

    model_config = ConfigDict(frozen=True)

    height: int
    digest: str


class Mismatched(Unretryable):
    """A snapshot whose contents do not hash to the digest the run was admitted with."""


def load_knowledge(books: str, pinned: Pinned) -> Effect[Knowledge]:
    """The snapshot `pinned` names, as a recorded step, refused when its contents do not match."""
    key = compose_key(t"knowledge:{Name(books)},height={Index(pinned.height):default=0}")
    call = CallTool(name="knowledge", args={"digest": pinned.digest}, result_schema=Knowledge)
    loaded = Knowledge.model_validate((yield from step(key.stored(), call)))
    if loaded.digest != pinned.digest:
        raise Mismatched(f"the snapshot read as {loaded.digest}, admitted as {pinned.digest}")
    return loaded


def adoption_of(books: str, height: int) -> Key:
    return compose_key(t"knowledge-adopted:{Name(books)},{Index(height)}")


class Adopted(BaseModel):
    height: int
    child: str


class Outrun(BaseModel):
    """An adoption whose height another adopter claimed first, with another child."""

    height: int
    held: str


class Unlinked(BaseModel):
    """An adoption whose parent is not the snapshot its books hold at the parent's height."""

    height: int
    parent: str


type Adoption = Adopted | Outrun | Unlinked


class Adopting(BaseModel):
    """What an adoption is admitted with: the books, the head it adopts from, and the child, which
    the driver also keeps in the store of snapshots."""

    model_config = ConfigDict(frozen=True)

    books: str
    parent: Pinned
    child: Knowledge

    def params(self) -> dict[str, Any]:
        return {ADOPTING: self.model_dump(mode="json")}

    @classmethod
    def admitted(cls, params: Mapping[str, Any]) -> Adopting:
        return cls.model_validate(params[ADOPTING])


ADOPTING = "adopting"
"""The spawn param an adoption's admission rides under."""


def _held_at(books: str, height: int) -> Effect[str | None]:
    """The digest of the snapshot `books` hold at `height`: the genesis at 0."""
    if height == 0:
        return K0.digest
    row = yield from read_ledger(adoption_of(books, height))
    return None if row is None else row["child"]


def adopted(adopting: Adopting, adopter: str) -> Effect[Adoption]:
    """Claim the height after the parent for the child, as `adopter`, once the books hold the
    parent at its height: the first adopter to claim a height holds it, and a later one with
    another child is outrun."""
    books, parent, child = adopting.books, adopting.parent, adopting.child
    height = parent.height + 1
    if (yield from _held_at(books, parent.height)) != parent.digest:
        return Unlinked(height=height, parent=parent.digest)
    row = LedgerRow(
        event_id=adoption_of(books, height),
        kind=ADOPTED,
        parent=parent.digest,
        child=child.digest,
        adopter=adopter,
    )
    held = yield from claim_ledger(row)
    if held["child"] != child.digest or held["parent"] != parent.digest:
        return Outrun(height=height, held=held["child"])
    return Adopted(height=height, child=child.digest)
