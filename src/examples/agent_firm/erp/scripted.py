"""A scripted Odoo: the calls the ERP port's Odoo adapter makes, answered in memory as Odoo 20
answers them.

| call                                     | what it does, as Odoo does                     |
|------------------------------------------|------------------------------------------------|
| `load` with no key                       | creates a draft                                |
| `load` under a new key                   | creates a draft and the key                    |
| `load` under a key whose move is a draft | appends the row's line to it                   |
| `load` under a key whose move is posted  | refuses, as data                               |
| `load` of a row that does not validate   | refuses, as data, and writes nothing           |
| `action_post` of a draft                 | posts it, on the first open day if locked      |
| `action_post` of a reversal              | also matches it against an open original       |
| `unlink` of a move                       | removes a draft, and refuses a posted move     |
| `create` of a key                        | holds it, and refuses a module and name held   |
| `create` of a server action              | holds its name and code                        |
| `run` of the fence action                | answers as `FENCE` does, in one transaction    |
| `run` of an action holding other code    | raises, since only `FENCE`'s semantics run     |

The fence's run is `FENCE`'s semantics written in Python; its text is run by the live Odoo only,
which the live arm exercises. One lock serializes every call, so a call is one transaction under
threads, as a database makes it, and a call's arguments marshal as XML-RPC's do. A move carries one
receivable line per installment of its terms, a debit for an invoice and a credit for a memo or a
payment; a search answers in id order.
"""

import xmlrpc.client
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from itertools import count
from threading import Lock
from typing import Any

from ..cases import Case
from .odoo import FENCE, MODULE, Claim, OdooErp
from .port import Unknown

DEBIT = {"out_invoice"}
"""The move types whose receivable line is a debit."""

LINES = 100
"""A receivable line's id is its move's id times this, plus its installment's index."""


@dataclass
class _Move:
    id: int
    kind: str
    partner: int
    date: str
    ref: str
    lines: list[Decimal]
    state: str = "draft"
    reverses: int | None = None
    narration: str = ""
    terms: tuple[Decimal, ...] = (Decimal(1),)
    """The share of the total each installment's receivable line carries."""

    @property
    def receivables(self) -> dict[int, Decimal]:
        """Each receivable line's amount by its id; the last installment takes the rounding."""
        total, ids = (
            sum(self.lines, Decimal(0)),
            [self.id * LINES + i for i in range(len(self.terms))],
        )
        shares = [(total * share).quantize(Decimal("0.01")) for share in self.terms[:-1]]
        return dict(zip(ids, [*shares, total - sum(shares, Decimal(0))], strict=True))

    @property
    def debit(self) -> bool:
        return self.kind in DEBIT


@dataclass
class _Match:
    id: int
    debit: int
    credit: int
    amount: Decimal


def _refused(message: str) -> dict[str, Any]:
    return {"ids": False, "messages": [{"type": "error", "message": message}], "nextrow": 0}


def _holds(row: Mapping[str, Any], domain: Sequence[tuple[str, str, Any]]) -> bool:
    def value(field: str) -> Any:
        match row.get(field):
            case [int(id), str()]:
                return id
            case other:
                return other

    def term(field: str, op: str, wanted: Any) -> bool:
        match op:
            case "=":
                return value(field) == wanted
            case "in":
                return value(field) in wanted
            case "!=":
                return value(field) != wanted
            case "ilike":
                return str(wanted).lower() in str(value(field)).lower()
            case _:
                raise Unknown(f"the scripted Odoo has no domain operator {op!r}")

    return all(term(*clause) for clause in domain)


@dataclass
class ScriptedOdoo:
    partners: dict[int, str] = field(default_factory=dict)
    moves: dict[int, _Move] = field(default_factory=dict)
    matches: dict[int, _Match] = field(default_factory=dict)
    keys: dict[str, int] = field(default_factory=dict)
    claims: dict[tuple[Claim, str], int] = field(default_factory=dict)
    """The invoice each claim on an apply's key was written for, by what it records and the
    key's name."""
    actions: dict[int, dict[str, Any]] = field(default_factory=dict)
    locked: str | None = None
    """The last day of the locked periods."""
    documents: dict[str, int] = field(default_factory=dict)
    ids: count[int] = field(default_factory=lambda: count(1))
    lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    @classmethod
    def seeded(cls, cases: Iterable[Case]) -> ScriptedOdoo:
        """Each case's books: its customer, its posted invoice, and its posted payment."""
        odoo = cls()
        for case in cases:
            invoice, payment = case.invoice, case.payment
            partner = next(odoo.ids)
            odoo.partners[partner] = invoice.customer
            move = odoo.posted("out_invoice", partner, invoice.issued, invoice.id, invoice.amount)
            paid = odoo.posted("entry", partner, payment.received, payment.id, payment.amount)
            odoo.documents |= {invoice.id: move, payment.id: paid}
        return odoo

    def erp(self) -> OdooErp:
        return OdooErp(self, self.documents)

    def posted(
        self,
        kind: str,
        partner: int,
        day: str,
        ref: str,
        amount: Decimal,
        terms: tuple[Decimal, ...] = (Decimal(1),),
    ) -> int:
        move = _Move(next(self.ids), kind, partner, day, ref, [amount], "posted", terms=terms)
        self.moves[move.id] = move
        return move.id

    def call(self, model: str, method: str, *args: Any, **kwargs: Any) -> Any:
        xmlrpc.client.dumps((list(args), kwargs), allow_none=True)
        with self.lock:
            match model, method, args:
                case "ir.actions.server", "run", ([int(action)],):
                    return self._run(action, kwargs.get("context", {}))
                case _, "search" | "search_read" | "search_count" | "read", _:
                    if set(kwargs) - {"order"} or kwargs.get("order", "id") != "id":
                        raise Unknown(f"the scripted Odoo answers in id order only, not {kwargs}")
                    return self._query(model, method, args)
                case _:
                    return self._write(model, method, args)

    def _run(self, action: int, ask: Mapping[str, Any]) -> dict[str, str]:
        if self.actions.get(action, {}).get("code") != FENCE:
            raise Unknown(
                f"the scripted Odoo runs no code but FENCE, and action {action} holds other"
            )
        name, invoice, line = ask["name"], ask["invoice"], ask["line"]
        owed = self.moves[invoice].receivables
        joins = [m.id for m in self.matches.values() if line in (m.debit, m.credit)]
        joins = [
            id for id in joins if {self.matches[id].debit, self.matches[id].credit} & set(owed)
        ]
        held = [claim for claim in Claim if (claim, name) in self.claims]
        mine: Claim = Claim.APPLIED
        found: str
        if Claim.UNDONE in held:
            found = Claim.UNDONE
        elif held and ask["op"] == "apply":
            found = held[0]
        elif ask["op"] == "unapply":
            mine, found = Claim.UNDONE, "created"
            for id in joins:
                del self.matches[id]
        elif joins:
            found = Claim.APPLIED
        elif self._owes(invoice) and self._residual(line):
            for receivable in owed:
                self._assign(receivable, line)
            found = "created"
        else:
            mine = found = Claim.REFUSED
        if found not in held:
            self.claims[mine, name] = invoice
        return {"found": str(found)}

    def _owes(self, invoice: int) -> Decimal:
        return sum((self._residual(line) for line in self.moves[invoice].receivables), Decimal(0))

    def _write(self, model: str, method: str, args: Sequence[Any]) -> Any:
        match model, method, args:
            case "account.move", "unlink", ([*moves],):
                return self._unlink(moves)
            case "ir.model.data", "create", ({"name": str(name), "res_id": int(move)},):
                return self._claim(name, move)
            case "ir.actions.server", "create", ({"name": str(), "code": str()} as action,):
                id = next(self.ids)
                self.actions[id] = {"id": id, **action}
                return id
            case "account.move", "load", (fields, [row]):
                return self._load(dict(zip(fields, row, strict=True)))
            case "account.move", "action_post", ([*moves],):
                for move in moves:
                    self._post(self.moves[move])
                return False
            case _:
                raise Unknown(f"the scripted Odoo does not answer {model}.{method}")

    def _unlink(self, moves: Sequence[int]) -> bool:
        if missing := [move for move in moves if move not in self.moves]:
            raise Unknown(f"Record does not exist or has been deleted: {missing}")
        if any(self.moves[move].state != "draft" for move in moves):
            raise Unknown("You cannot delete an entry which has been posted once.")
        for move in moves:
            del self.moves[move]
        return True

    def _claim(self, name: str, move: int) -> int:
        if name in self.keys:
            raise Unknown(f"Key (module, name)=({MODULE}, {name}) already exists.")
        self.keys[name] = move
        return move

    def _query(self, model: str, method: str, args: Sequence[Any]) -> Any:
        rows = self._rows(model)
        match method, args:
            case "search", (domain,):
                return [row["id"] for row in rows if _holds(row, domain)]
            case "search_read", (domain, *_):
                return [row for row in rows if _holds(row, domain)]
            case "search_count", (domain,):
                return sum(1 for row in rows if _holds(row, domain))
            case "read", (ids, *_):
                by_id = {row["id"]: row for row in rows}
                if missing := [id for id in ids if id not in by_id]:
                    raise Unknown(f"{model} {missing} does not exist")
                return [by_id[id] for id in ids]
            case _:
                raise Unknown(f"the scripted Odoo does not answer {model}.{method}{args}")

    def _rows(self, model: str) -> list[dict[str, Any]]:
        match model:
            case "account.move":
                return [self._move_row(move) for move in self.moves.values()]
            case "account.move.line":
                return [
                    self._line_row(move, line)
                    for move in self.moves.values()
                    for line in move.receivables
                ]
            case "account.partial.reconcile":
                return [
                    {
                        "id": match.id,
                        "debit_move_id": [match.debit, ""],
                        "credit_move_id": [match.credit, ""],
                    }
                    for match in self.matches.values()
                ]
            case "ir.model.data":
                held = [(MODULE, name, move) for name, move in self.keys.items()]
                held += [(str(kind), name, move) for (kind, name), move in self.claims.items()]
                return [
                    {"id": i, "module": module, "name": name, "model": "account.move"}
                    | {"res_id": move}
                    for i, (module, name, move) in enumerate(held, 1)
                ]
            case "ir.model":
                return [{"id": 1, "model": "account.move"}]
            case "ir.actions.server":
                return list(self.actions.values())
            case _:
                raise Unknown(f"the scripted Odoo has no model {model}")

    def _move_row(self, move: _Move) -> dict[str, Any]:
        return {
            "id": move.id,
            "move_type": move.kind,
            "partner_id": [move.partner, self.partners[move.partner]],
            "state": move.state,
            "date": move.date,
            "invoice_date": move.date,
            "ref": move.ref,
            "narration": move.narration,
            "amount_total": float(sum(move.lines)),
            "reversed_entry_id": [move.reverses, ""] if move.reverses else False,
            "origin_payment_id": [move.id, move.ref] if move.kind == "entry" else False,
        }

    def _line_row(self, move: _Move, line: int) -> dict[str, Any]:
        residual = self._residual(line)
        return {
            "id": line,
            "move_id": [move.id, move.ref],
            "account_id.account_type": "asset_receivable",
            "amount_residual": float(residual if move.debit else -residual),
            "reconciled": not residual,
            "matched_debit_ids": [m.id for m in self.matches.values() if m.credit == line],
            "matched_credit_ids": [m.id for m in self.matches.values() if m.debit == line],
        }

    def _residual(self, line: int) -> Decimal:
        matched = (m.amount for m in self.matches.values() if line in (m.debit, m.credit))
        return self.moves[line // LINES].receivables[line] - sum(matched, Decimal(0))

    def _load(self, row: Mapping[str, str]) -> dict[str, Any]:
        name = row["id"].split(".", 1)[1] if row.get("id") else None
        try:
            date.fromisoformat(row["invoice_date"])
            amount = Decimal(row["invoice_line_ids/price_unit"])
        except ValueError, InvalidOperation:
            return _refused("the date or the amount does not validate")
        partner, reverses = int(row["partner_id/.id"]), row["reversed_entry_id/.id"]
        if partner not in self.partners or (reverses and int(reverses) not in self.moves):
            return _refused("No matching record found for database id")
        if name is not None and (held := self.keys.get(name)) is not None:
            move = self.moves[held]
            if move.state == "posted":
                return _refused(
                    "You cannot modify the following readonly fields on the posted move"
                )
            move.lines.append(amount)
            return {"ids": [held], "messages": [], "nextrow": 0}
        move = _Move(
            next(self.ids),
            row["move_type"],
            partner,
            row["invoice_date"],
            row["ref"],
            [amount],
            reverses=int(reverses) if reverses else None,
            narration=row["narration"],
        )
        self.moves[move.id] = move
        if name is not None:
            self.keys[name] = move.id
        return {"ids": [move.id], "messages": [], "nextrow": 0}

    def _post(self, move: _Move) -> None:
        if move.state != "draft":
            raise Unknown(f"The entry (id {move.id}) must be in draft.")
        if self.locked is not None and move.date <= self.locked:
            move.date = (date.fromisoformat(self.locked) + timedelta(days=1)).isoformat()
        move.state = "posted"
        if move.reverses is not None:
            for original in self.moves[move.reverses].receivables:
                for line in move.receivables:
                    self._assign(original, line)

    def _assign(self, one: int, other: int) -> None:
        """Match what the lines `one` and `other` have left, when one is a debit and one a
        credit."""
        if self.moves[one // LINES].debit == self.moves[other // LINES].debit:
            return
        debit, credit = (one, other) if self.moves[one // LINES].debit else (other, one)
        if amount := min(self._residual(debit), self._residual(credit)):
            match = _Match(next(self.ids), debit, credit, amount)
            self.matches[match.id] = match
