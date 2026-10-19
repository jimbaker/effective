"""The ERP port's contract on the scripted Odoo and a live one, with a fault at every call an op
makes, before the call or after the server committed it.

| pin                     | what holds                                                           |
|-------------------------|----------------------------------------------------------------------|
| once per key            | after a faulted op and its replay: one key, one match, to the cent   |
| a fault is unknown      | a faulted op answers only what landed                                |
| a replay finds it       | the replay of an op that landed answers `EXISTING`                   |
| a refusal is data       | a refused row, or an entry nobody holds, is `Refused`, unwritten     |
| a key names one move    | an op whose key names a different move is `Refused`                  |
| an apply reads its line | the second of two equal memos applies `CREATED`, on its own line     |
| an apply's claim        | an apply lands once per key, and nothing after the undo naming it    |
| an undo's claim         | an undo lands once per apply it names                                |
| the fence's code        | an action under the fence's name holding other code is never run     |
| matches move            | the fence reads the matches as it acts, not as the adapter read them |
| a shared match          | an undo removes a match at the last release only, and says which     |
| installments            | a payment applies across an invoice's installments, and unapplies    |
"""

import re
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from functools import partial
from typing import Any
from uuid import uuid4

import pytest
from _odoo import ERPS, another_invoice, opened
from _turns import Turns, schedules

from examples.agent_firm.cases import CLEAN, SLA_CAP
from examples.agent_firm.erp.odoo import (
    FENCE,
    FENCE_NAME,
    FIELDS,
    Claim,
    OdooErp,
    Row,
    Transport,
    XmlRpc,
    name_of,
)
from examples.agent_firm.erp.port import (
    Applied,
    Apply,
    ErpOp,
    ErpResult,
    Found,
    Kind,
    Posted,
    PostMemo,
    Read,
    Refused,
    Reverse,
    Statement,
    Unapplied,
    Unapply,
    Unknown,
)
from examples.agent_firm.erp.scripted import ScriptedOdoo

INVOICE, PAYMENT = SLA_CAP.invoice.id, SLA_CAP.payment.id
CLEAN_INVOICE = CLEAN.invoice.id
TIMEOUT = 10.0
"""How long a thread of a test waits on another before the test fails."""
MEMO = PostMemo(invoice=INVOICE, amount=Decimal(1000), date="2026-09-28", reason="clause 4.2")


def fresh(label: str) -> str:
    """A key no earlier run minted: a live Odoo keeps every key it was given."""
    return f"{label} {uuid4().hex}"


class Position(StrEnum):
    BEFORE_CALL = "before-call"
    AFTER_COMMIT = "after-commit"


@dataclass
class Faulting:
    """A transport that raises `Unknown` at its `at`th call, before it or after it committed."""

    inner: Transport
    at: int | None
    position: Position
    calls: int = 0

    def call(self, model: str, method: str, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == self.at and self.position is Position.BEFORE_CALL:
            raise Unknown(f"before {model}.{method}")
        answer = self.inner.call(model, method, *args, **kwargs)
        if self.calls == self.at and self.position is Position.AFTER_COMMIT:
            raise Unknown(f"after {model}.{method}")
        return answer


@dataclass(frozen=True)
class Scenario:
    """An op on fresh books, after `setup`, and what holds once it landed."""

    setup: Callable[[OdooErp], ErpOp]
    landed: Callable[[OdooErp, ErpOp], None]


def _memos(erp: OdooErp) -> list[Decimal]:
    return [e.amount for e in erp.entries(INVOICE) if e.kind is Kind.CREDIT_MEMO]


def _posted(erp: OdooErp, op: ErpOp) -> None:
    assert (_memos(erp), erp.keyed(INVOICE)) == ([Decimal(1000)], 1)


def _reversal(erp: OdooErp) -> ErpOp:
    match erp.perform(MEMO, fresh("the memo")):
        case Posted(entry=entry):
            return Reverse(entry=entry, reason="reversal")
        case unexpected:
            raise AssertionError(unexpected)


def _reversed(erp: OdooErp, op: ErpOp) -> None:
    *_, memo, reversal = erp.entries(INVOICE)
    assert (memo.kind, reversal.kind, reversal.reverses) == (
        Kind.CREDIT_MEMO,
        Kind.REVERSAL,
        memo.id,
    )
    assert (erp.keyed(INVOICE), erp.residual(memo.id), erp.residual(reversal.id)) == (2, 0, 0)


def _applied(erp: OdooErp, op: ErpOp) -> None:
    assert (erp.residual(PAYMENT), erp.residual(INVOICE)) == (0, 3000)


def _unapplying(erp: OdooErp) -> ErpOp:
    return _applied_under(erp, fresh("the apply"))


def _unapplied(erp: OdooErp, op: ErpOp) -> None:
    assert (erp.residual(PAYMENT), erp.residual(INVOICE)) == (97000, 100000)


OPS = {
    "post": Scenario(lambda erp: MEMO, _posted),
    "reverse": Scenario(_reversal, _reversed),
    "apply": Scenario(lambda erp: Apply(entry=PAYMENT, invoice=INVOICE), _applied),
    "unapply": Scenario(_unapplying, _unapplied),
}


def _found(answer: ErpResult) -> Found | None:
    """How an op that landed found its write, or nothing for a refusal."""
    match answer:
        case Posted(found=found) | Applied(found=found) | Unapplied(found=found):
            return found
        case Refused():
            return None


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
@pytest.mark.parametrize("op", OPS)
def test_an_op_lands_once_whatever_a_fault_interrupts(arm, op):
    open_, case = opened(arm, [SLA_CAP]), OPS[op]

    def run(at: int | None, position: Position) -> int:
        transport, documents = open_()
        erp = OdooErp(transport, documents)
        asked = case.setup(erp)
        faulting, key = Faulting(transport, at, position), fresh("the op")
        try:
            answer = OdooErp(faulting, documents).perform(asked, key)
        except Unknown:
            answer = None
        if answer is not None:
            assert _found(answer), (at, position, answer)
            case.landed(erp, asked)
        replayed = _found(erp.perform(asked, key))
        assert replayed, (at, position)
        if answer is not None:
            assert replayed is Found.EXISTING, (at, position)
        case.landed(erp, asked)
        return faulting.calls

    calls = run(None, Position.BEFORE_CALL)
    for position in Position:
        for at in range(1, calls + 1):
            run(at, position)


REFUSED = {
    "a row Odoo refuses": MEMO.model_copy(update={"date": "not-a-date"}),
    "an invoice nobody holds": MEMO.model_copy(update={"invoice": "INV-9999"}),
    "a negative memo": MEMO.model_copy(update={"amount": Decimal(-250)}),
    "a zero memo": MEMO.model_copy(update={"amount": Decimal(0)}),
    "a memo of a part of a cent": MEMO.model_copy(update={"amount": Decimal("0.004")}),
    "a reversal of nothing": Reverse(entry="PAY-9999", reason="reversal"),
    "a reversal of a payment": Reverse(entry=PAYMENT, reason="reversal"),
    "an apply of nothing": Apply(entry="PAY-9999", invoice=INVOICE),
    "an invoice applied to itself": Apply(entry=INVOICE, invoice=INVOICE),
    "a payment applied to another customer's invoice": Apply(
        entry=CLEAN.payment.id, invoice=INVOICE
    ),
    "an unapply of nothing": Unapply(entry=PAYMENT, invoice="INV-9999", undoes="an apply"),
    "a payment unapplied from itself": Unapply(entry=PAYMENT, invoice=PAYMENT, undoes="an apply"),
    "a read of an invoice nobody holds": Read(invoice="INV-9999"),
}


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
@pytest.mark.parametrize("op", REFUSED)
def test_a_refusal_is_an_answer_and_writes_nothing(arm, op):
    erp = OdooErp(*opened(arm, [SLA_CAP, CLEAN])())
    erp.perform(Apply(entry=PAYMENT, invoice=INVOICE), fresh("the apply"))
    match erp.perform(REFUSED[op], fresh("the op")):
        case Refused():
            pass
        case unexpected:
            raise AssertionError(unexpected)
    assert (_memos(erp), erp.keyed(INVOICE)) == ([], 0)
    assert (erp.residual(PAYMENT), erp.residual(INVOICE)) == (0, 3000)
    assert (erp.residual(CLEAN_INVOICE), erp.residual(CLEAN.payment.id)) == (50000, 50000)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_a_read_answers_the_books_as_they_stand_under_any_key(arm):
    """A read lands nothing, so one key asked again reads what moved since."""
    erp, key = OdooErp(*opened(arm, [SLA_CAP])()), fresh("the read")

    def read() -> tuple[Decimal, list[Kind]]:
        match erp.perform(Read(invoice=INVOICE), key):
            case Statement(owed=owed, entries=entries):
                return owed, [entry.kind for entry in entries]
            case unexpected:
                raise AssertionError(unexpected)

    assert read() == (100000, [Kind.INVOICE, Kind.PAYMENT])
    _applied_under(erp, fresh("the apply"))
    match erp.perform(MEMO, fresh("the memo")):
        case Posted(entry=memo):
            erp.perform(Apply(entry=memo, invoice=INVOICE), fresh("the memo's apply"))
        case unexpected:
            raise AssertionError(unexpected)
    assert read() == (2000, [Kind.INVOICE, Kind.PAYMENT, Kind.CREDIT_MEMO])
    assert erp.keyed(INVOICE) == 1


def _applied_under(erp: OdooErp, key: str) -> Unapply:
    """The payment applied under `key`, and the undo that names it."""
    assert _found(erp.perform(Apply(entry=PAYMENT, invoice=INVOICE), key)) is Found.CREATED
    return Unapply(entry=PAYMENT, invoice=INVOICE, undoes=key)


def _applied_under_key(erp: OdooErp, key: str) -> str:
    """The payment applied under `key`, and the key."""
    return _applied_under(erp, key).undoes


def _removed(answer: ErpResult) -> bool:
    match answer:
        case Unapplied(removed=bool(removed)):
            return removed
        case unexpected:
            raise AssertionError(unexpected)


def _matched_by_a_person(erp: OdooErp, invoice: str = INVOICE) -> None:
    """The payment matched against `invoice` in Odoo, as a person matches it, with no claim."""
    (line,) = erp._receivables(erp.documents[PAYMENT])
    owed = erp.residual(invoice)
    with suppress(Unknown):
        erp.transport.call(
            "account.move", "js_assign_outstanding_line", [erp._move(invoice)], line
        )
    assert erp.residual(invoice) < owed


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_an_apply_lands_nothing_after_the_undo_naming_it(arm):
    erp, key = OdooErp(*opened(arm, [SLA_CAP])()), fresh("the apply")
    erp.perform(_applied_under(erp, key), fresh("the unapply"))
    match erp.perform(Apply(entry=PAYMENT, invoice=INVOICE), key):
        case Refused():
            pass
        case unexpected:
            raise AssertionError(unexpected)
    assert (erp.keyed(INVOICE), erp.residual(INVOICE), erp.residual(PAYMENT)) == (0, 100000, 97000)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_an_undo_naming_an_apply_that_never_landed_removes_nothing(arm):
    """The match stands, and the key that made it, asked again, answers that it landed."""
    erp, key = OdooErp(*opened(arm, [SLA_CAP])()), fresh("the apply")
    _applied_under(erp, key)
    another = Unapply(entry=PAYMENT, invoice=INVOICE, undoes=fresh("another apply"))
    assert _removed(erp.perform(another, fresh("the unapply"))) is False
    found = _found(erp.perform(Apply(entry=PAYMENT, invoice=INVOICE), key))
    assert (found, erp.keyed(INVOICE), erp.residual(INVOICE)) == (Found.EXISTING, 0, 3000)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_an_undo_lands_once_per_apply_it_names(arm):
    """Asked again after a later apply under another key, the undo answers that it landed and
    leaves the later match standing."""
    erp, key = OdooErp(*opened(arm, [SLA_CAP])()), fresh("the apply")
    undo, under = _applied_under(erp, key), fresh("the unapply")
    assert _found(erp.perform(undo, under)) is Found.CREATED
    _applied_under(erp, fresh("a later apply"))
    assert _found(erp.perform(undo, under)) is Found.EXISTING
    assert (erp.keyed(INVOICE), erp.residual(INVOICE)) == (0, 3000)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_a_refused_apply_answers_once_per_key_whatever_moved_since(arm):
    """The payment is spent on another invoice, so the apply under the key is refused; once the
    payment is freed, the key asked again is still refused and matches nothing."""
    erp, other = another_invoice(arm, Decimal(97000))
    elsewhere, key = fresh("the apply elsewhere"), fresh("the apply")
    erp.perform(Apply(entry=PAYMENT, invoice=other), elsewhere)
    apply = Apply(entry=PAYMENT, invoice=INVOICE)
    assert isinstance(erp.perform(apply, key), Refused)
    erp.perform(Unapply(entry=PAYMENT, invoice=other, undoes=elsewhere), fresh("the unapply"))
    assert isinstance(erp.perform(apply, key), Refused)
    assert (erp.residual(INVOICE), erp.residual(PAYMENT)) == (100000, 97000)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_an_apply_that_found_another_keys_match_lands_nothing_after_its_undo(arm):
    """The apply under the key finds the match another key made and answers `EXISTING`; its undo
    releases that match, which the other key still holds, and the key asked again lands nothing."""
    erp, key = OdooErp(*opened(arm, [SLA_CAP])()), fresh("the apply")
    _applied_under(erp, fresh("another apply"))
    apply = Apply(entry=PAYMENT, invoice=INVOICE)
    assert _found(erp.perform(apply, key)) is Found.EXISTING
    erp.perform(Unapply(entry=PAYMENT, invoice=INVOICE, undoes=key), fresh("the unapply"))
    assert isinstance(erp.perform(apply, key), Refused)
    assert (erp.residual(INVOICE), erp.residual(PAYMENT)) == (3000, 0)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_an_unapply_says_whether_it_removed_the_match(arm):
    """Of two keys holding one match, the first undo releases it and the second removes it; each
    undo, asked again, answers as it did."""
    erp, first, second = OdooErp(*opened(arm, [SLA_CAP])()), fresh("the apply"), fresh("another")
    _applied_under(erp, first)
    assert _found(erp.perform(Apply(entry=PAYMENT, invoice=INVOICE), second)) is Found.EXISTING
    undos = [
        (Unapply(entry=PAYMENT, invoice=INVOICE, undoes=key), fresh("the unapply"))
        for key in (first, second)
    ]
    assert [_removed(erp.perform(*undo)) for undo in undos] == [False, True]
    assert [_removed(erp.perform(*undo)) for undo in undos] == [False, True]
    assert (erp.residual(INVOICE), erp.residual(PAYMENT)) == (100000, 97000)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_an_undo_answers_its_removal_beside_a_match_it_never_held(arm):
    """The payment holds the second installment, a person matches it to the first, and the
    payment's undo removes only the match it held."""
    erp, invoice = another_invoice(arm, Decimal(1000), installments=True)
    match erp.perform(MEMO.model_copy(update={"amount": Decimal(300)}), fresh("the memo")):
        case Posted(entry=memo):
            pass
        case unexpected:
            raise AssertionError(unexpected)
    first, second = fresh("the memo's apply"), fresh("the payment's apply")
    erp.perform(Apply(entry=memo, invoice=invoice), first)
    erp.perform(Apply(entry=PAYMENT, invoice=invoice), second)
    erp.perform(Unapply(entry=memo, invoice=invoice, undoes=first), fresh("the memo's undo"))
    assert erp.residual(invoice) == 300
    _matched_by_a_person(erp, invoice)
    undo = Unapply(entry=PAYMENT, invoice=invoice, undoes=second)
    match erp.perform(undo, fresh("the payment's undo")):
        case Unapplied(found=Found.CREATED, removed=True):
            pass
        case unexpected:
            raise AssertionError(unexpected)
    assert erp.residual(invoice) == 700


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_the_last_release_removes_a_match_a_person_made(arm):
    """A compensation undoes the books to what they were before the claims, and the ERP cannot
    tell an adopted match's history from its own."""
    transport, documents = opened(arm, [SLA_CAP])()
    erp, key = OdooErp(transport, documents), fresh("the apply")
    _matched_by_a_person(erp)
    assert _found(erp.perform(Apply(entry=PAYMENT, invoice=INVOICE), key)) is Found.EXISTING
    undo = Unapply(entry=PAYMENT, invoice=INVOICE, undoes=key)
    assert _removed(erp.perform(undo, fresh("the unapply"))) is True
    assert (erp.residual(INVOICE), erp.residual(PAYMENT)) == (100000, 97000)


@dataclass
class BeforeRun:
    """A transport that runs `hook` once, just before the first fence action it is asked to run."""

    inner: Transport
    hook: Callable[[], object] | None

    def call(self, model: str, method: str, *args: Any, **kwargs: Any) -> Any:
        if (model, method) == ("ir.actions.server", "run") and (hook := self.hook) is not None:
            self.hook = None
            hook()
        return self.inner.call(model, method, *args, **kwargs)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_an_apply_whose_found_match_is_undone_before_its_fence_runs_matches_itself(arm):
    transport, documents = opened(arm, [SLA_CAP])()
    erp, first, second = OdooErp(transport, documents), fresh("the apply"), fresh("a later apply")
    undo = Unapply(entry=PAYMENT, invoice=INVOICE, undoes=_applied_under_key(erp, first))
    later = OdooErp(
        BeforeRun(transport, lambda: erp.perform(undo, fresh("the unapply"))), documents
    )
    assert _found(later.perform(Apply(entry=PAYMENT, invoice=INVOICE), second)) is Found.CREATED
    assert erp.residual(INVOICE) == 3000


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_an_undo_removes_the_apply_that_lands_before_its_fence_runs(arm):
    transport, documents = opened(arm, [SLA_CAP])()
    erp, key = OdooErp(transport, documents), fresh("the apply")
    apply = Apply(entry=PAYMENT, invoice=INVOICE)
    earlier = OdooErp(BeforeRun(transport, lambda: erp.perform(apply, key)), documents)
    undo = Unapply(entry=PAYMENT, invoice=INVOICE, undoes=key)
    assert _found(earlier.perform(undo, fresh("the unapply"))) is Found.CREATED
    assert isinstance(erp.perform(apply, key), Refused)
    assert (erp.residual(INVOICE), erp.residual(PAYMENT)) == (100000, 97000)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_an_earlier_rounds_undo_leaves_the_match_a_later_round_adopted(arm):
    """Round 0 applies, round 1 finds that match and claims it, then round 0's undo arrives: it
    releases round 0's claim and leaves the match round 1 holds."""
    erp, earlier, later = OdooErp(*opened(arm, [SLA_CAP])()), fresh("round 0"), fresh("round 1")
    apply = Apply(entry=PAYMENT, invoice=INVOICE)
    erp.perform(apply, earlier)
    assert _found(erp.perform(apply, later)) is Found.EXISTING
    erp.perform(Unapply(entry=PAYMENT, invoice=INVOICE, undoes=earlier), fresh("the unapply"))
    assert _found(erp.perform(apply, later)) is Found.EXISTING
    assert erp.residual(INVOICE) == 3000


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_an_apply_that_finds_another_keys_match_claims_its_own_key(arm):
    transport, documents = opened(arm, [SLA_CAP])()
    erp, key = OdooErp(transport, documents), fresh("the apply")
    _applied_under(erp, fresh("another apply"))
    erp.perform(Apply(entry=PAYMENT, invoice=INVOICE), key)
    domain = [("module", "=", Claim.APPLIED.value), ("name", "=", name_of(key))]
    assert transport.call("ir.model.data", "search_count", domain) == 1


def _paused(transport: Transport, after: str) -> int:
    """A server action running `FENCE` with its transaction held open for two seconds after the
    first line that starts `after`. A sleep inside the server is the only order a test can impose
    between two transactions Odoo runs, since no event crosses its RPC boundary."""
    lines = FENCE.splitlines()
    at = next(i for i, line in enumerate(lines) if line.lstrip().startswith(after))
    indent = lines[at][: len(lines[at]) - len(lines[at].lstrip())]
    lines.insert(at + 1, f'{indent}env.cr.execute("SELECT pg_sleep(2)")')
    code = "\n".join(lines) + "\n"
    (model,) = transport.call("ir.model", "search", [("model", "=", "account.move")])
    held = {"name": f"paused {uuid4().hex}", "model_id": model, "state": "code", "code": code}
    return transport.call("ir.actions.server", "create", held)


PAUSED = """
env.cr.execute(
    "SELECT count(*) FROM pg_stat_activity"
    " WHERE state = 'active' AND query LIKE 'SELECT pg_sleep(2)%'"
)
action = {"paused": env.cr.fetchone()[0]}
"""
"""A server action answering how many fence runs are inside their pause."""


def _inside_its_pause(transport: Transport) -> None:
    """Return once a paused fence run's transaction is open, inside its pause."""
    (model,) = transport.call("ir.model", "search", [("model", "=", "account.move")])
    held = {"name": f"pause probe {uuid4().hex}", "model_id": model, "state": "code"}
    probe = transport.call("ir.actions.server", "create", held | {"code": PAUSED})
    deadline = time.monotonic() + TIMEOUT
    try:
        while transport.call("ir.actions.server", "run", [probe])["paused"] == 0:
            assert time.monotonic() < deadline, "no fence run reached its pause"
    finally:
        transport.call("ir.actions.server", "unlink", [probe])


def _fenced_by(action: int, op: str, key: str, documents: Mapping[str, int], line: int) -> Any:
    """`action` run on a connection of its own, as the adapter runs the fence."""
    context = {"op": op, "name": name_of(key), "invoice": documents[INVOICE], "line": line}
    return XmlRpc.from_env().call("ir.actions.server", "run", [action], context=context)


def _live() -> tuple[Transport, OdooErp, dict[str, int], int]:
    transport, documents = opened("odoo", [SLA_CAP])()
    erp = OdooErp(transport, documents)
    (line,) = erp._receivables(documents[PAYMENT])
    return transport, erp, documents, line


@pytest.mark.adversarial
def test_two_releases_of_one_match_at_once_remove_it():
    """Each release, held open after it counts the other's hold, would leave the match to the
    other; serialized, the later one is the last and removes it."""
    transport, erp, documents, line = _live()
    keys = [fresh("an apply") for _ in range(2)]
    for key in keys:
        erp.perform(Apply(entry=PAYMENT, invoice=INVOICE), key)
    slow = _paused(transport, "kept = holds(")
    try:
        with ThreadPoolExecutor(2) as pool:
            runs = [pool.submit(_fenced_by, slow, "unapply", key, documents, line) for key in keys]
            claims = sorted(run.result()["claim"] for run in runs)
    finally:
        transport.call("ir.actions.server", "unlink", [slow])
    assert claims == [Claim.RELEASED, Claim.UNDONE]
    assert erp.residual(INVOICE) == 100000


@pytest.mark.adversarial
def test_an_apply_and_its_undo_at_once_land_as_one_order_would():
    """The apply, held open after it read no claim, and its undo run at once: in either order the
    undo leaves the books unmatched, and the key asked again lands nothing."""
    transport, erp, documents, line = _live()
    key, slow = fresh("the apply"), _paused(transport, "joins = joined()")
    try:
        with ThreadPoolExecutor(1) as pool:
            apply = pool.submit(_fenced_by, slow, "apply", key, documents, line)
            _inside_its_pause(transport)
            erp.perform(Unapply(entry=PAYMENT, invoice=INVOICE, undoes=key), fresh("the undo"))
            apply.result()
    finally:
        transport.call("ir.actions.server", "unlink", [slow])
    assert erp.residual(INVOICE) == 100000
    assert isinstance(erp.perform(Apply(entry=PAYMENT, invoice=INVOICE), key), Refused)


@pytest.mark.adversarial
def test_an_adoption_beside_the_last_release_holds_a_match_that_stands():
    """An adoption, held open after it read the match, and the release of the match's only other
    claim run at once: the adopting key ends up holding a match that stands."""
    transport, erp, documents, line = _live()
    first, second = fresh("the first apply"), fresh("the adopting apply")
    erp.perform(Apply(entry=PAYMENT, invoice=INVOICE), first)
    slow = _paused(transport, "joins = joined()")
    try:
        with ThreadPoolExecutor(1) as pool:
            adoption = pool.submit(_fenced_by, slow, "apply", second, documents, line)
            _inside_its_pause(transport)
            undo = Unapply(entry=PAYMENT, invoice=INVOICE, undoes=first)
            erp.perform(undo, fresh("the undo"))
            adoption.result()
    finally:
        transport.call("ir.actions.server", "unlink", [slow])
    domain = [("module", "=", Claim.HELD.value), ("name", "=like", f"{name_of(second)}.%")]
    held = [
        row["res_id"] for row in transport.call("ir.model.data", "search_read", domain, ["res_id"])
    ]
    standing = transport.call("account.partial.reconcile", "search", [("id", "in", held)])
    assert held
    assert sorted(standing) == sorted(held)


@pytest.mark.adversarial
def test_the_scripted_fence_runs_no_code_but_the_fences():
    odoo = ScriptedOdoo.seeded([SLA_CAP])
    edited = {"name": FENCE_NAME, "state": "code", "code": FENCE.replace("unlink()", "unlink(1)")}
    action = odoo.call("ir.actions.server", "create", edited)
    context = {"op": "apply", "name": name_of(fresh("the apply")), "invoice": 1, "line": 200}
    with pytest.raises(Unknown):
        odoo.call("ir.actions.server", "run", [action], context=context)


@pytest.mark.adversarial
def test_the_fence_names_each_claim_the_adapter_reads():
    """The server action's text, which only the live arm runs, writes and answers the modules the
    adapter reads its claims under."""
    modules = set(re.findall(r'"(effective_[a-z]+)"', FENCE))
    assert modules == {claim.value for claim in Claim}


@pytest.mark.adversarial
def test_the_scripted_fence_runs_as_one_transaction(monkeypatch):
    """A second attempt of one apply under one key waits while the first is inside its match,
    then answers from the first's claim and match."""
    odoo, entered, go = ScriptedOdoo.seeded([SLA_CAP]), threading.Event(), threading.Event()
    assign = odoo._assign

    def paused(one: int, other: int) -> None:
        entered.set()
        assert go.wait(TIMEOUT)
        assign(one, other)

    monkeypatch.setattr(odoo, "_assign", paused)
    key, apply, answers = fresh("the apply"), Apply(entry=PAYMENT, invoice=INVOICE), {}
    first = threading.Thread(target=lambda: answers.update(first=odoo.erp().perform(apply, key)))
    first.start()
    assert entered.wait(TIMEOUT)
    second = threading.Thread(target=lambda: answers.update(second=odoo.erp().perform(apply, key)))
    second.start()
    second.join(0.5)
    waited = second.is_alive()
    go.set()
    first.join(TIMEOUT)
    second.join(TIMEOUT)
    assert waited, answers
    assert answers["first"] == Applied(residual=Decimal(0), found=Found.CREATED)
    assert answers["second"] == Applied(residual=Decimal(0), found=Found.EXISTING)


@pytest.mark.adversarial
def test_a_fence_action_holding_other_code_is_never_run():
    """An action under the fence's name whose code was edited, or written by an adapter with other
    code, is left alone, and the apply runs an action holding exactly `FENCE`."""
    odoo = ScriptedOdoo.seeded([SLA_CAP])
    edited = {"name": FENCE_NAME, "state": "code", "code": FENCE.replace("unlink()", "unlink(1)")}
    stale = odoo.call("ir.actions.server", "create", edited)
    applied = odoo.erp().perform(Apply(entry=PAYMENT, invoice=INVOICE), fresh("the apply"))
    assert _found(applied) is Found.CREATED
    codes = {row["id"]: row["code"] for row in odoo.call("ir.actions.server", "read", [stale])}
    assert codes == {stale: edited["code"]}
    assert [a["code"] for a in odoo.actions.values() if a["id"] != stale] == [FENCE]


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_a_payment_applies_across_an_invoices_installments(arm):
    """Two keys hold both installments' matches; the first undo releases them, the second
    removes them."""
    erp, invoice = another_invoice(arm, Decimal(1000), installments=True)
    apply, keys = Apply(entry=PAYMENT, invoice=invoice), [fresh("the apply") for _ in range(2)]
    applied = [_found(erp.perform(apply, key)) for key in keys]
    assert applied == [Found.CREATED, Found.EXISTING]
    assert (erp.residual(invoice), erp.residual(PAYMENT)) == (0, 96000)
    for key, books in zip(keys, [(0, 96000), (1000, 97000)], strict=True):
        unapply = Unapply(entry=PAYMENT, invoice=invoice, undoes=key)
        assert _found(erp.perform(unapply, fresh("the unapply"))) is Found.CREATED
        assert (erp.residual(invoice), erp.residual(PAYMENT)) == books


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_of_two_equal_memos_applying_the_second_is_not_the_first(arm):
    transport, documents = opened(arm, [SLA_CAP])()
    erp = OdooErp(transport, documents)
    first, second = (erp.perform(MEMO, fresh(key)) for key in ("the first", "the second"))
    assert isinstance(first, Posted)
    assert isinstance(second, Posted)
    applied = [
        erp.perform(Apply(entry=memo.entry, invoice=INVOICE), fresh("the apply"))
        for memo in (first, second)
    ]
    assert [answer.found for answer in applied if isinstance(answer, Applied)] == [
        Found.CREATED,
        Found.CREATED,
    ]
    assert (erp.residual(first.entry), erp.residual(second.entry)) == (0, 0)
    assert erp.residual(INVOICE) == 98000


DIVERGING: dict[str, Callable[[OdooErp], tuple[ErpOp, ErpOp]]] = {
    "another amount": lambda erp: (MEMO, MEMO.model_copy(update={"amount": Decimal(2000)})),
    "another customer": lambda erp: (MEMO, MEMO.model_copy(update={"invoice": CLEAN_INVOICE})),
    "another kind": lambda erp: (MEMO, _reversal(erp)),
    "another original": lambda erp: (_reversal(erp), _reversal(erp)),
}
"""Two ops given one key, the second describing a move the first did not make."""


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
@pytest.mark.parametrize("pair", DIVERGING)
def test_a_key_names_one_move_and_refuses_to_nameanother_invoice(arm, pair):
    erp, key = OdooErp(*opened(arm, [SLA_CAP, CLEAN])()), fresh("the op")
    first, second = DIVERGING[pair](erp)
    landed = erp.perform(first, key)
    assert _found(landed) is Found.CREATED
    match erp.perform(second, key):
        case Refused(reason="the key names another move"):
            pass
        case unexpected:
            raise AssertionError(unexpected)
    assert erp.perform(first, key) == landed.model_copy(update={"found": Found.EXISTING})
    assert erp.keyed(INVOICE) == len(_memos(erp)) + (pair == "another original")


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_an_unapply_leaves_the_entrys_match_on_another_invoice(arm):
    erp, small = another_invoice(arm, Decimal(500))
    keys = {invoice: fresh("the apply") for invoice in (small, INVOICE)}
    for invoice, key in keys.items():
        erp.perform(Apply(entry=PAYMENT, invoice=invoice), key)
    assert (erp.residual(small), erp.residual(INVOICE)) == (0, 3500)
    erp.perform(
        Unapply(entry=PAYMENT, invoice=INVOICE, undoes=keys[INVOICE]), fresh("the unapply")
    )
    assert (erp.residual(small), erp.residual(INVOICE), erp.residual(PAYMENT)) == (
        0,
        100000,
        96500,
    )


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_a_paid_invoice_takes_no_further_credit(arm):
    erp, small = another_invoice(arm, Decimal(500))
    erp.perform(Apply(entry=PAYMENT, invoice=small), fresh("the apply"))
    memo = erp.perform(MEMO, fresh("the memo"))
    assert isinstance(memo, Posted)
    match erp.perform(Apply(entry=memo.entry, invoice=small), fresh("the apply")):
        case Refused():
            pass
        case unexpected:
            raise AssertionError(unexpected)
    assert (erp.residual(small), erp.residual(memo.entry)) == (0, 1000)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_an_applied_memo_is_unapplied_before_it_is_reversed(arm):
    erp = OdooErp(*opened(arm, [SLA_CAP])())
    memo = erp.perform(MEMO, fresh("the memo"))
    assert isinstance(memo, Posted)
    applied = fresh("the apply")
    erp.perform(Apply(entry=memo.entry, invoice=INVOICE), applied)
    reverse = Reverse(entry=memo.entry, reason="reversal")
    match erp.perform(reverse, fresh("the reversal")):
        case Refused():
            pass
        case unexpected:
            raise AssertionError(unexpected)
    assert [e.kind for e in erp.entries(INVOICE)][-1] is Kind.CREDIT_MEMO
    erp.perform(Unapply(entry=memo.entry, invoice=INVOICE, undoes=applied), fresh("the unapply"))
    assert _found(erp.perform(reverse, fresh("the reversal"))) is Found.CREATED
    assert erp.residual(memo.entry) == 0


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_a_draft_is_neither_reversed_nor_applied(arm):
    transport, documents = opened(arm, [SLA_CAP])()
    erp = OdooErp(transport, documents)
    customer = transport.call("account.move", "read", [documents[INVOICE]], ["partner_id"])
    row = Row("out_refund", customer[0]["partner_id"][0], "2026-09-28", "draft", Decimal(1000))
    (draft,) = transport.call("account.move", "load", FIELDS, [row.values("a draft")])["ids"]
    for op in (
        Reverse(entry=str(draft), reason="reversal"),
        Apply(entry=str(draft), invoice=INVOICE),
    ):
        match erp.perform(op, fresh("the op")):
            case Refused():
                pass
            case unexpected:
                raise AssertionError(unexpected)
    assert erp.residual(INVOICE) == 100000


class Interleaving:
    """A transport that, before its `at`th call, runs `replay` once: a second attempt of the op
    overlapping the first."""

    def __init__(self, inner: Transport, at: int | None, replay: Callable[[], None]) -> None:
        self.inner, self.at, self.replay, self.calls = inner, at, replay, 0

    def call(self, model: str, method: str, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == self.at:
            self.replay()
        return self.inner.call(model, method, *args, **kwargs)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
@pytest.mark.parametrize("op", OPS)
def test_an_op_lands_once_when_its_replay_overlaps_it(arm, op):
    open_, case = opened(arm, [SLA_CAP]), OPS[op]

    def run(at: int | None) -> int:
        transport, documents = open_()
        erp = OdooErp(transport, documents)
        asked, key, answers = case.setup(erp), fresh("the op"), []

        def replay() -> None:
            answers.append(_found(erp.perform(asked, key)))

        interleaving = Interleaving(transport, at, replay)
        answers.append(_found(OdooErp(interleaving, documents).perform(asked, key)))
        assert all(answers), (at, answers)
        case.landed(erp, asked)
        return interleaving.calls

    calls = run(None)
    for at in range(1, calls + 1):
        run(at)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("op", ["post", "reverse"])
def test_two_attempts_at_once_land_one_entry_on_odoo(op):
    open_, case = opened("odoo", [SLA_CAP]), OPS[op]

    def trial() -> None:
        transport, documents = open_()
        erp = OdooErp(transport, documents)
        asked, key = case.setup(erp), fresh("the op")
        answers: list[Found | Unknown | None] = []

        def attempt() -> None:
            try:
                answers.append(_found(OdooErp(XmlRpc.from_env(), documents).perform(asked, key)))
            except Unknown as unknown:
                answers.append(unknown)

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(answers) == 2, answers
        assert all(answers), answers
        assert _found(erp.perform(asked, key)) is Found.EXISTING
        case.landed(erp, asked)

    for _ in range(4):
        trial()


def _turned(
    turns: Turns, name: str, transport: Transport, documents: Any, asked: ErpOp, key: str
) -> ErpResult:
    """One attempt of `asked`, taking a turn before each of its calls."""
    return OdooErp(turns.transport(name, transport), documents).perform(asked, key)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("op", OPS)
def test_two_attempts_land_once_in_every_schedule_of_their_calls(op):
    """On the scripted Odoo, two attempts of one op under one key, interleaved call by call in
    every order with at most three switches between them."""
    open_, case, explored = opened("scripted", [SLA_CAP]), OPS[op], 0
    for turns in schedules(["a", "b"], switches=3):
        transport, documents = open_()
        erp = OdooErp(transport, documents)
        asked, key = case.setup(erp), fresh("the op")
        answers = turns.run(
            {
                name: partial(_turned, turns, name, transport, documents, asked, key)
                for name in turns.names
            }
        )
        found = {name: _found(answer) for name, answer in answers.items()}
        assert all(found.values()), (turns.choices, answers)
        case.landed(erp, asked)
        explored += 1
    assert explored > 1


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_a_spent_payment_applies_nothing(arm):
    erp, other = another_invoice(arm, Decimal(97000))
    erp.perform(Apply(entry=PAYMENT, invoice=other), fresh("the apply"))
    match erp.perform(Apply(entry=PAYMENT, invoice=INVOICE), fresh("the apply")):
        case Refused():
            pass
        case unexpected:
            raise AssertionError(unexpected)
    assert (erp.residual(other), erp.residual(INVOICE)) == (0, 100000)


@dataclass
class Aborting:
    """A transport whose first `load` answers as Odoo does when Postgres aborted it for a
    concurrent write, without running it."""

    inner: Transport
    aborted: bool = False

    def call(self, model: str, method: str, *args: Any, **kwargs: Any) -> Any:
        if method == "load" and not self.aborted:
            self.aborted = True
            message = "could not serialize access due to concurrent update"
            return {"ids": False, "messages": [{"type": "error", "message": message}]}
        return self.inner.call(model, method, *args, **kwargs)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_a_load_the_database_aborted_is_unknown_and_lands_on_replay(arm):
    transport, documents = opened(arm, [SLA_CAP])()
    erp, key = OdooErp(transport, documents), fresh("the memo")
    with pytest.raises(Unknown):
        OdooErp(Aborting(transport), documents).perform(MEMO, key)
    assert _found(erp.perform(MEMO, key)) is Found.CREATED
    _posted(erp, MEMO)


def _drafts(transport: Transport, key: str) -> list[int]:
    domain = [("narration", "ilike", name_of(key)), ("state", "=", "draft")]
    return transport.call("account.move", "search", domain, order="id")


def _left(transport: Transport, documents: Any, key: str) -> None:
    """A draft of `MEMO` loaded under the key's name and never claimed, as a crash or a loser
    whose delete faulted leaves it."""
    (held,) = transport.call("account.move", "read", [documents[INVOICE]], ["partner_id"])
    row = Row("out_refund", held["partner_id"][0], MEMO.date, MEMO.reason, MEMO.amount)
    transport.call("account.move", "load", FIELDS, [row.values(name_of(key))])


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_a_replay_adopts_the_draft_a_crash_left_unclaimed(arm):
    transport, documents = opened(arm, [SLA_CAP])()
    erp, key = OdooErp(transport, documents), fresh("the memo")
    _left(transport, documents, key)
    (draft,) = _drafts(transport, key)
    match erp.perform(MEMO, key):
        case Posted(entry=entry, found=Found.EXISTING):
            assert entry == str(draft)
        case unexpected:
            raise AssertionError(unexpected)
    _posted(erp, MEMO)
    assert _drafts(transport, key) == []


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_a_replay_deletes_a_draft_its_key_does_not_name(arm):
    transport, documents = opened(arm, [SLA_CAP])()
    erp, key = OdooErp(transport, documents), fresh("the memo")
    erp.perform(MEMO, key)
    _left(transport, documents, key)
    assert _found(erp.perform(MEMO, key)) is Found.EXISTING
    _posted(erp, MEMO)
    assert _drafts(transport, key) == []


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_a_key_naming_a_reversed_move_is_refused(arm):
    """A second resolution of a case whose first was compensated lands on no undone memo."""
    erp, key = OdooErp(*opened(arm, [SLA_CAP])()), fresh("the memo")
    match erp.perform(MEMO, key):
        case Posted(entry=entry):
            erp.perform(Reverse(entry=entry, reason="reversal"), fresh("the reversal"))
        case unexpected:
            raise AssertionError(unexpected)
    match erp.perform(MEMO, key):
        case Refused(reason="the key names a reversed move"):
            pass
        case unexpected:
            raise AssertionError(unexpected)


@pytest.mark.conformance
@pytest.mark.adversarial
def test_two_divergent_memos_under_one_key_post_one_in_every_schedule():
    """On the scripted Odoo, a $1,000 and a $3,000 memo under one key, interleaved call by call
    in every order with at most three switches: one posts, the other is refused, no draft is
    left."""
    open_, explored = opened("scripted", [SLA_CAP]), 0
    other = MEMO.model_copy(update={"amount": Decimal(3000)})
    for turns in schedules(["a", "b"], switches=3):
        transport, documents = open_()
        erp, key, asked = OdooErp(transport, documents), fresh("the memo"), {"a": MEMO, "b": other}
        answers = turns.run(
            {
                name: partial(_turned, turns, name, transport, documents, asked[name], key)
                for name in turns.names
            }
        )
        kinds = sorted(type(answer).__name__ for answer in answers.values())
        assert kinds == ["Posted", "Refused"], (turns.choices, answers)
        assert len(_memos(erp)) == 1, turns.choices
        assert not transport.call("account.move", "search", [("state", "=", "draft")])
        explored += 1
    assert explored > 1
