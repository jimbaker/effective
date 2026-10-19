"""The ERP port on Odoo Community, over its external API.

| op                    | Odoo's write                                | how a replay finds it     |
|-----------------------|---------------------------------------------|---------------------------|
| `PostMemo`, `Reverse` | `load` a draft; claim the key; post         | the key; the move's state |
| `Apply`, `Unapply`    | `FENCE`: claim the key; match or unlink     | the claim                 |

A key is claimed by creating its `ir.model.data` row, whose module and name are unique, so of two
attempts that overlap exactly one claims it: the other deletes its own draft, which never posted,
and posts the winner's. A draft carries its key's name in `narration`, so a replay after a crash
between the draft and the claim adopts that draft. Nothing loads into a move that exists, since a
second `load` appends its line to a draft. After a write raises, the adapter reads the state: Odoo
commits before it answers, and a method that returns nothing raises after it commits.

An apply and an unapply run as one server action, `FENCE`, so a claim and the match it records
commit together. Two attempts that overlap on one key both try to claim it; the second's create
violates the unique name and rolls its whole action back, which reaches the adapter as `Unknown`,
and the step's retry finds the first's claim.
"""

import hashlib
import os
import xmlrpc.client
from collections.abc import Collection, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, Protocol, assert_never
from urllib.parse import urljoin

from ..cases import Case
from .port import (
    Applied,
    Apply,
    Diverged,
    Entry,
    ErpOp,
    ErpResult,
    Found,
    Kind,
    Posted,
    PostMemo,
    Refused,
    Reverse,
    Unapplied,
    Unapply,
    Unknown,
)

MODULE = "effective_keys"
"""The `ir.model.data` module the keys live under. An update of a module collects that module's
rows, and this name is the port's alone, so its rows outlive every addon update."""


class Claim(StrEnum):
    """What a claim on an apply's key records, as the `ir.model.data` module it lives under. Its
    name is the key's, so an apply and the undo naming it find each other, and a count of keyed
    entries never sees a claim."""

    APPLIED = "effective_applied"
    REFUSED = "effective_refused"
    UNDONE = "effective_undone"


FENCE_NAME = "effective: apply or unapply once per claim"

FENCE = """
ask = env.context
data = env["ir.model.data"]
claims = ["effective_applied", "effective_refused", "effective_undone"]
held = data.search([("module", "in", claims), ("name", "=", ask["name"])]).mapped("module")
invoice = env["account.move"].browse(ask["invoice"])
line = env["account.move.line"].browse(ask["line"])
owed = invoice.line_ids.filtered(lambda row: row.account_id.account_type == "asset_receivable")
joins = (line.matched_debit_ids | line.matched_credit_ids).filtered(
    lambda join: join.debit_move_id in owed or join.credit_move_id in owed
)
mine = "effective_applied"
if "effective_undone" in held:
    action = {"found": "effective_undone"}
elif held and ask["op"] == "apply":
    action = {"found": held[0]}
elif ask["op"] == "unapply":
    mine = "effective_undone"
    joins.unlink()
    action = {"found": "created"}
elif joins:
    action = {"found": "effective_applied"}
elif invoice.amount_residual and line.amount_residual:
    invoice.js_assign_outstanding_line(line.id)
    action = {"found": "created"}
else:
    mine = "effective_refused"
    action = {"found": mine}
if action["found"] not in held:
    claim = {"module": mine, "name": ask["name"], "model": "account.move"}
    data.create(dict(claim, res_id=invoice.id))
"""
"""The server action an apply and an unapply run, in one transaction that reads the claims and
the entry line's matches against the invoice as it acts. An apply answers the claim its key
holds; holding none, it claims the key as applied, matching the entry unless a match already
stands, or as refused when the entry has nothing left or the invoice owes nothing. An unapply
answers its claim, or claims the key as undone and removes the matches. The adapter runs only an
action holding exactly this code, and creates one on first use; whoever administers Odoo's server
actions can still change what runs between the lookup and the run, and the port trusts them."""

CENT = Decimal("0.01")

FIELDS = [
    "move_type",
    "partner_id/.id",
    "invoice_date",
    "ref",
    "narration",
    "reversed_entry_id/.id",
    "invoice_line_ids/name",
    "invoice_line_ids/quantity",
    "invoice_line_ids/price_unit",
    "invoice_line_ids/tax_ids/.id",
]
"""The columns of the row a draft is loaded from."""

TRANSIENT = ("could not serialize", "deadlock detected")
"""What Odoo says, among a load's messages, when the database aborted it for a concurrent
write: the load did not run, and may run again."""

DESCRIBED = ("move_type", "partner_id", "amount_total", "reversed_entry_id")
"""The fields a keyed move is checked against the op by: a key names one move, and only it."""

MIRROR = {"out_refund": "out_invoice", "out_invoice": "out_refund"}
"""The move type of the entry that reverses a move of each type."""


class Transport(Protocol):
    def call(self, model: str, method: str, *args: Any, **kwargs: Any) -> Any:
        """`method` of `model` on the server, raising `Unknown` when no answer could be read."""
        ...


class XmlRpc:
    """Odoo's XML-RPC API, from the standard library."""

    def __init__(self, url: str, db: str, user: str, password: str) -> None:
        self.db, self.password = db, password
        self.uid = xmlrpc.client.ServerProxy(urljoin(url, "/xmlrpc/2/common")).authenticate(
            db, user, password, {}
        )
        self.models = xmlrpc.client.ServerProxy(urljoin(url, "/xmlrpc/2/object"), allow_none=True)

    @classmethod
    def from_env(cls) -> XmlRpc:
        """The server whose origin `EFFECTIVE_ODOO_URL` names, defaulting to the pod `just odoo-up`
        starts. Odoo serves its API at the origin's root, so a path on the URL is dropped."""
        env = os.environ.get
        return cls(
            env("EFFECTIVE_ODOO_URL", "http://127.0.0.1:18069"),
            env("EFFECTIVE_ODOO_DB", "spike"),
            env("EFFECTIVE_ODOO_USER", "admin"),
            env("EFFECTIVE_ODOO_PASSWORD", "admin"),
        )

    def call(self, model: str, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.models.execute_kw(
                self.db, self.uid, self.password, model, method, list(args), kwargs
            )
        except xmlrpc.client.Fault as fault:
            reason = (fault.faultString.strip().splitlines() or [""])[-1]
            raise Unknown(f"{model}.{method}: {reason}") from fault
        except (xmlrpc.client.Error, OSError) as error:
            raise Unknown(f"{model}.{method}: {error}") from error


@dataclass(frozen=True)
class Row:
    """A move to load: a memo, or with `reverses` set, the reversal of that move."""

    kind: str
    partner: int
    date: str
    reason: str
    amount: Decimal
    reverses: int | None = None

    def described(self, held: Mapping[str, Any]) -> Row:
        """The row that describes the move `held` reads as, with this row's date and reason."""
        reverses = _ids([held["reversed_entry_id"]])
        return replace(
            self,
            kind=held["move_type"],
            partner=held["partner_id"][0],
            amount=cents(held["amount_total"]),
            reverses=reverses[0] if reverses else None,
        )

    def values(self, name: str) -> list[str]:
        """The row's values in the order of `FIELDS`, carrying the key's name."""
        row = {
            "move_type": self.kind,
            "partner_id/.id": str(self.partner),
            "invoice_date": self.date,
            "ref": self.reason,
            "narration": name,
            "reversed_entry_id/.id": "" if self.reverses is None else str(self.reverses),
            "invoice_line_ids/name": self.reason,
            "invoice_line_ids/quantity": "1",
            "invoice_line_ids/price_unit": str(self.amount),
            "invoice_line_ids/tax_ids/.id": "",
        }
        return [row[column] for column in FIELDS]


def name_of(key: str) -> str:
    """The `ir.model.data` name an idempotency key is held under."""
    return f"k_{hashlib.sha256(key.encode()).hexdigest()[:32]}"


def cents(amount: Any) -> Decimal:
    return abs(Decimal(str(amount))).quantize(CENT)


@dataclass
class OdooErp:
    transport: Transport
    documents: Mapping[str, int]
    """The move of each invoice and payment the world holds, by the document's id."""

    def perform(self, op: ErpOp, key: str) -> ErpResult:
        match op:
            case PostMemo() | Reverse():
                return self._keyed(key, op)
            case Apply(entry=entry, invoice=invoice):
                match self._pair(entry, invoice):
                    case Refused() as refused:
                        return refused
                    case moved, owing:
                        return self._apply(moved, owing, key)
                    case unreachable:
                        assert_never(unreachable)
            case Unapply(entry=entry, invoice=invoice, undoes=undoes):
                match self._pair(entry, invoice):
                    case Refused() as refused:
                        return refused
                    case moved, owing:
                        return self._unapply(moved, owing, undoes)
                    case unreachable:
                        assert_never(unreachable)
            case unreachable:
                assert_never(unreachable)

    def entries(self, invoice: str) -> list[Entry]:
        fields = ["move_type", "amount_total", "date", "ref", "reversed_entry_id"]
        fields.append("origin_payment_id")
        domain = [("partner_id", "=", self._partner(invoice)), ("state", "=", "posted")]
        rows = self.transport.call("account.move", "search_read", domain, fields, order="id")
        return [self._entry(row) for row in rows]

    def residual(self, document: str) -> Decimal:
        return self._owed(self._receivables(self._known(document)))

    def keyed(self, invoice: str) -> int:
        domain = [("partner_id", "=", self._partner(invoice))]
        moves = self.transport.call("account.move", "search", domain)
        held = [("module", "=", MODULE), ("model", "=", "account.move"), ("res_id", "in", moves)]
        return self.transport.call("ir.model.data", "search_count", held)

    def _move(self, ref: str) -> int | None:
        """The move a document id or an entry id names."""
        held: int | None = self.documents.get(ref)
        match held:
            case int(move):
                return move
            case None:
                return int(ref) if ref.isdigit() else None
            case unreachable:
                assert_never(unreachable)

    def _pair(self, entry: str, invoice: str) -> tuple[int, int] | Refused:
        """The entry and the invoice, when the entry is a posted credit on the account of the
        posted invoice's customer."""
        moved, owing = self._move(entry), self._move(invoice)
        fields = ["move_type", "state", "partner_id", "origin_payment_id"]
        domain = [("id", "in", [moved, owing])]
        found = self.transport.call("account.move", "search_read", domain, fields)
        rows = {row["id"]: row for row in found}
        match rows.get(moved), rows.get(owing):
            case (
                {"id": int(credited), "state": "posted", "partner_id": [int(customer), *_]},
                {"id": int(owed), "state": "posted", "move_type": "out_invoice"} as invoiced,
            ) if customer == invoiced["partner_id"][0] and _credit(rows[credited]):
                return credited, owed
            case _:
                return Refused(
                    reason="the entry is not a posted credit of the posted invoice's customer",
                    about={"entry": entry, "invoice": invoice},
                )

    def _known(self, ref: str) -> int:
        match self._move(ref):
            case int(move):
                return move
            case None:
                raise LookupError(f"no document or entry {ref!r}")
            case unreachable:
                assert_never(unreachable)

    def _read(self, move: int, *fields: str) -> dict[str, Any]:
        match self.transport.call("account.move", "read", [move], list(fields)):
            case [dict() as row]:
                return row
            case unexpected:
                raise Unknown(f"move {move} reads as {unexpected!r}")

    def _partner(self, invoice: str) -> int:
        return self._read(self._known(invoice), "partner_id")["partner_id"][0]

    def _entry(self, row: Mapping[str, Any]) -> Entry:
        match row:
            case {"reversed_entry_id": [int(original), *_]}:
                kind, reverses = Kind.REVERSAL, str(original)
            case {"move_type": "out_refund"}:
                kind, reverses = Kind.CREDIT_MEMO, None
            case {"move_type": "out_invoice"}:
                kind, reverses = Kind.INVOICE, None
            case {"move_type": "entry", "origin_payment_id": [int(), *_]}:
                kind, reverses = Kind.PAYMENT, None
            case _:
                kind, reverses = Kind.OTHER, None
        return Entry(
            id=str(row["id"]),
            kind=kind,
            amount=cents(row["amount_total"]),
            period=row["date"][:7],
            reason=row["ref"] or "",
            reverses=reverses,
        )

    def _row(self, op: PostMemo | Reverse, fresh: bool) -> Row | Refused:
        """The move the op describes; a `fresh` op, not yet landed, is also checked to be one
        that can land."""
        match op:
            case PostMemo(amount=amount) if amount <= 0 or amount != amount.quantize(CENT):
                return Refused(
                    reason="a memo is a positive amount in cents", about={"amount": str(amount)}
                )
            case PostMemo(invoice=invoice) if (owing := self._move(invoice)) is not None:
                partner = self._read(owing, "partner_id")["partner_id"][0]
                return Row("out_refund", partner, op.date, op.reason, op.amount)
            case Reverse(entry=entry) if (move := self._move(entry)) is not None:
                match self._reversal(move, op.reason):
                    case Row() if fresh and self._matches(move):
                        return Refused(
                            reason="a matched entry is unapplied before it is reversed",
                            about={"entry": entry},
                        )
                    case described:
                        return described
            case PostMemo(invoice=ref) | Reverse(entry=ref):
                return Refused(reason="no document or entry is held", about={"ref": ref})

    def _reversal(self, move: int, reason: str) -> Row | Refused:
        fields = ("move_type", "partner_id", "invoice_date", "amount_total", "state")
        match self._read(move, *fields):
            case {"state": "posted", "move_type": str(kind)} as original if kind in MIRROR:
                partner, date = original["partner_id"][0], original["invoice_date"]
                amount = cents(original["amount_total"])
                return Row(MIRROR[kind], partner, date, reason, amount, reverses=move)
            case {"state": state, "move_type": kind}:
                return Refused(
                    reason="a posted invoice, memo or payment is reversed here",
                    about={"move": str(move), "state": str(state), "kind": str(kind)},
                )
            case unexpected:
                raise Unknown(f"move {move} reads as {unexpected!r}")

    def _resolved(self, name: str) -> int | None:
        domain = [("module", "=", MODULE), ("name", "=", name)]
        match self.transport.call("ir.model.data", "search_read", domain, ["model", "res_id"]):
            case []:
                return None
            case [{"model": "account.move", "res_id": int(move)}]:
                return move
            case unexpected:
                raise Diverged(f"key {name} names {unexpected!r}")

    def _keyed(self, key: str, op: PostMemo | Reverse) -> ErpResult:
        name = name_of(key)
        landed = self._resolved(name)
        match landed, self._row(op, fresh=landed is None):
            case None, Refused() if self._resolved(name) is not None:
                return self._keyed(key, op)
            case _, Refused() as refused:
                return refused
            case int(move), Row() as row:
                self._reap(name, move)
                return self._posted(move, row, Found.EXISTING)
            case None, Row() as row:
                match self._claimed(name, row):
                    case Refused() as refused:
                        return refused
                    case move, found:
                        return self._posted(move, row, found)
                    case unreachable:
                        assert_never(unreachable)

    def _claimed(self, name: str, row: Row) -> tuple[int, Found] | Refused:
        """The move the key names once this attempt has claimed it, or lost it to another."""
        match self._orphan(name):
            case int(draft):
                found = Found.EXISTING
            case None:
                match self._drafted(name, row):
                    case Refused() as refused:
                        return refused
                    case int(draft):
                        found = Found.CREATED
                    case unreachable:
                        assert_never(unreachable)
            case unreachable:
                assert_never(unreachable)
        claim = {"module": MODULE, "name": name, "model": "account.move", "res_id": draft}
        with suppress(Unknown):
            self.transport.call("ir.model.data", "create", claim)
        match self._resolved(name):
            case int(move) if move == draft:
                return move, found
            case int(move):
                self._reap(name, move)
                return move, Found.EXISTING
            case None:
                raise Unknown(f"key {name} was not claimed")

    def _reap(self, name: str, kept: int) -> None:
        """Delete every draft loaded under the key's name but the move the key names: a loser's
        draft, or one a crash left behind. None of them ever posted."""
        domain = [("narration", "ilike", name), ("state", "=", "draft"), ("id", "!=", kept)]
        for draft in self.transport.call("account.move", "search", domain, order="id"):
            with suppress(Unknown):
                self.transport.call("account.move", "unlink", [draft])

    def _matches(self, move: int) -> bool:
        """Whether any of the move's receivable lines is matched."""
        return any(self._partials(line)[0] for line in self._receivables(move))

    def _orphan(self, name: str) -> int | None:
        """A draft an earlier attempt loaded under the key's name and did not claim."""
        domain = [("narration", "ilike", name), ("state", "=", "draft")]
        match self.transport.call("account.move", "search", domain, order="id"):
            case [int(draft), *_]:
                return draft
            case _:
                return None

    def _drafted(self, name: str, row: Row) -> int | Refused:
        try:
            answer = self.transport.call("account.move", "load", FIELDS, [row.values(name)])
        except Unknown:
            match self._orphan(name):
                case int(draft):
                    return draft
                case None:
                    raise
                case unreachable:
                    assert_never(unreachable)
        match answer:
            case {"ids": [int(move)]}:
                return move
            case {"ids": False, "messages": [*messages]}:
                reasons = [str(m.get("message")) for m in messages]
                if any(transient in reason for reason in reasons for transient in TRANSIENT):
                    raise Unknown(f"the load was aborted: {reasons}")
                return Refused(reason="; ".join(reasons))
            case unexpected:
                raise Unknown(f"load answered {unexpected!r}")

    def _posted(self, move: int, row: Row, found: Found) -> Posted | Refused:
        fields = ("state", "date", *DESCRIBED)
        match self._read(move, *fields):
            case held if row.described(held) != row:
                return Refused(reason="the key names another move", about={"move": str(move)})
            case {"state": "draft"}:
                with suppress(Unknown):
                    self.transport.call("account.move", "action_post", [move])
            case {"state": "posted"}:
                pass
            case unexpected:
                raise Unknown(f"move {move} reads as {unexpected!r}")
        match self._read(move, *fields):
            case held if row.described(held) != row:
                return Refused(reason="the key names another move", about={"move": str(move)})
            case {"state": "posted"} if self._reversed(move):
                return Refused(reason="the key names a reversed move", about={"move": str(move)})
            case {"state": "posted", "date": str(date), "amount_total": total}:
                return Posted(entry=str(move), amount=cents(total), period=date[:7], found=found)
            case unexpected:
                raise Unknown(f"move {move} did not post: {unexpected!r}")

    def _reversed(self, move: int) -> bool:
        """Does a posted entry reverse `move`?"""
        domain = [("reversed_entry_id", "=", move), ("state", "=", "posted")]
        return bool(self.transport.call("account.move", "search", domain))

    def _receivables(self, move: int) -> list[int]:
        """The move's receivable lines: one for a memo or a payment, one per installment for an
        invoice."""
        domain = [("move_id", "=", move), ("account_id.account_type", "=", "asset_receivable")]
        return self.transport.call("account.move.line", "search", domain, order="id")

    def _partials(self, line: int) -> tuple[list[int], Decimal]:
        """Every match on `line`, and what `line` has left."""
        fields = ["amount_residual", "matched_debit_ids", "matched_credit_ids"]
        match self.transport.call("account.move.line", "read", [line], fields):
            case [{"amount_residual": residual, "matched_debit_ids": [*debits]} as row]:
                return [*debits, *row["matched_credit_ids"]], cents(residual)
            case unexpected:
                raise Unknown(f"line {line} reads as {unexpected!r}")

    def _matched(self, line: int, against: Collection[int]) -> tuple[list[int], Decimal]:
        """The matches joining `line` to any of the lines `against`, and what `line` has left."""
        partials, residual = self._partials(line)
        if not partials or not against:
            return [], cents(residual)
        sides = ["debit_move_id", "credit_move_id"]
        domain = [("id", "in", partials)]
        rows = self.transport.call("account.partial.reconcile", "search_read", domain, sides)
        joined = [p["id"] for p in rows if set(against) & set(_ids(p[side] for side in sides))]
        return joined, cents(residual)

    def _owed(self, lines: Iterable[int]) -> Decimal:
        return sum((self._matched(line, ())[1] for line in lines), Decimal(0))

    def _apply(self, entry: int, invoice: int, key: str) -> Applied | Refused:
        match self._receivables(entry), self._receivables(invoice):
            case [line], ([_, *_] as against):
                pass
            case lines, against:
                return _unmatchable(entry, lines, invoice, against)
        found = self._fenced("apply", key, invoice=invoice, line=line)
        match found, self._matched(line, against):
            case Claim.UNDONE, _:
                return _undone(entry, invoice)
            case Claim.REFUSED, _:
                return Refused(
                    reason="the entry had nothing left to apply, or the invoice owed nothing",
                    about=_moves(entry, invoice),
                )
            case Claim.APPLIED, (_, residual):
                return Applied(residual=residual, found=Found.EXISTING)
            case None, ([_, *_], residual):
                return Applied(residual=residual, found=Found.CREATED)
            case _:
                raise Unknown(f"move {entry} was not matched against {invoice}: {found}")

    def _unapply(self, entry: int, invoice: int, undoes: str) -> Unapplied | Refused:
        match self._receivables(entry), self._receivables(invoice):
            case [line], ([_, *_] as against):
                pass
            case lines, against:
                return _unmatchable(entry, lines, invoice, against)
        found = self._fenced("unapply", undoes, invoice=invoice, line=line)
        match found, self._matched(line, against):
            case Claim.UNDONE, (_, residual):
                return Unapplied(residual=residual, found=Found.EXISTING)
            case None, ([], residual):
                return Unapplied(residual=residual, found=Found.CREATED)
            case _:
                raise Unknown(f"move {entry} is still matched against {invoice}: {found}")

    def _fenced(self, op: Literal["apply", "unapply"], applied: str, **asked: Any) -> Claim | None:
        """`FENCE`'s answer for `op` on the apply keyed `applied`: the claim the key held or this
        call wrote, or `None` when this call made or removed the match."""
        context = {"op": op, "name": name_of(applied), **asked}
        match self.transport.call("ir.actions.server", "run", [self._fence()], context=context):
            case {"found": "created"}:
                return None
            case {"found": str(found)} if found in {claim.value for claim in Claim}:
                return Claim(found)
            case unexpected:
                raise Unknown(f"the fence answered {unexpected!r}")

    def _fence(self) -> int:
        """The id of a server action holding exactly `FENCE`, created when none does, so an
        action edited in Odoo, or one an adapter with other code created, is never run."""
        domain = [("name", "=", FENCE_NAME), ("code", "=", FENCE)]
        if held := self.transport.call("ir.actions.server", "search", domain):
            return min(held)
        (model,) = self.transport.call("ir.model", "search", [("model", "=", "account.move")])
        action = {"name": FENCE_NAME, "model_id": model, "state": "code", "code": FENCE}
        return self.transport.call("ir.actions.server", "create", action)


def _credit(move: Mapping[str, Any]) -> bool:
    """Whether a move credits a customer's account: a memo, or a payment's entry."""
    match move:
        case {"move_type": "out_refund"} | {
            "move_type": "entry",
            "origin_payment_id": [int(), *_],
        }:
            return True
        case _:
            return False


def _undone(entry: int, invoice: int) -> Refused:
    return Refused(reason="the key names an apply since undone", about=_moves(entry, invoice))


def _unmatchable(entry: int, lines: list[int], invoice: int, against: list[int]) -> Refused:
    return Refused(
        reason="an apply matches one receivable line against one or more",
        about=_moves(entry, invoice) | {"lines": str(len(lines)), "against": str(len(against))},
    )


def _moves(entry: int | None = None, invoice: int | None = None) -> dict[str, str]:
    """The `about` of a refusal naming an entry, an invoice, or both, by move id."""
    named = {"entry": entry, "invoice": invoice}
    return {role: str(move) for role, move in named.items() if move is not None}


def _ids(values: Iterable[Any]) -> list[int]:
    """The ids of many-to-one values, which Odoo reads as `[id, name]`."""
    return [value[0] for value in values if value]


def seed(transport: Transport, cases: Iterable[Case], tag: str) -> dict[str, int]:
    """Open each case's books: its customer, with `tag` as its reference, its posted invoice, and
    its payment, posted; returns the move of each invoice and payment by the
    document's id."""
    (bank,) = transport.call("account.journal", "search", [("type", "=", "bank")], limit=1)
    (outstanding,) = transport.call(
        "account.account", "search", [("name", "=", "Outstanding Receipts")]
    )
    documents = {}
    for case in cases:
        invoice, payment = case.invoice, case.payment
        customer = {"name": invoice.customer, "ref": tag}
        partner = transport.call("res.partner", "create", customer)
        line = {"name": "Platform fee", "quantity": 1, "price_unit": float(invoice.amount)}
        move = transport.call(
            "account.move",
            "create",
            {
                "move_type": "out_invoice",
                "partner_id": partner,
                "invoice_date": invoice.issued,
                "ref": invoice.id,
                "invoice_line_ids": [(0, 0, line | {"tax_ids": []})],
            },
        )
        transport.call("account.move", "action_post", [move])
        record = transport.call(
            "account.payment",
            "create",
            {
                "payment_type": "inbound",
                "partner_type": "customer",
                "partner_id": partner,
                "amount": float(payment.amount),
                "date": payment.received,
                "journal_id": bank,
                "memo": payment.id,
                "outstanding_account_id": outstanding,
            },
        )
        with suppress(Unknown):
            transport.call("account.payment", "action_post", [record])
        (paid,) = transport.call("account.payment", "read", [record], ["move_id"])
        documents[invoice.id], documents[payment.id] = move, paid["move_id"][0]
    return documents
