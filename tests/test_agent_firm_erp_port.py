"""The ERP port's contract on the scripted Odoo and a live one, with a fault at every call an op
makes, before the call or after the server committed it.

| pin                     | what holds                                                       |
|-------------------------|------------------------------------------------------------------|
| once per key            | after a faulted op and its replay: one key, one match, to the cent |
| a fault is unknown      | a faulted op answers only what landed                            |
| a replay finds it       | the replay of an op that landed answers `EXISTING`               |
| a refusal is data       | a refused row, or an entry nobody holds, is `Refused`, unwritten |
| a key names one move    | an op whose key names a different move is `Refused`              |
| an apply reads its line | the second of two equal memos applies `CREATED`, on its own line |
| an apply's key          | an apply is read from its match, so its key names nothing        |
| installments            | a payment applies across an invoice's installments, and unapplies |
"""

import threading
from collections.abc import Callable
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
    FIELDS,
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
    Refused,
    Reverse,
    Unapplied,
    Unapply,
    Unknown,
)

INVOICE, PAYMENT = SLA_CAP.invoice.id, SLA_CAP.payment.id
CLEAN_INVOICE = CLEAN.invoice.id
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
    erp.perform(Apply(entry=PAYMENT, invoice=INVOICE), fresh("the apply"))
    return Unapply(entry=PAYMENT, invoice=INVOICE)


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
    "an unapply of nothing": Unapply(entry=PAYMENT, invoice="INV-9999"),
    "a payment unapplied from itself": Unapply(entry=PAYMENT, invoice=PAYMENT),
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
def test_an_apply_is_read_from_its_match_and_its_key_names_nothing(arm):
    erp, key = OdooErp(*opened(arm, [SLA_CAP])()), fresh("the apply")
    apply = Apply(entry=PAYMENT, invoice=INVOICE)
    answers = [
        erp.perform(apply, key),
        erp.perform(Unapply(entry=PAYMENT, invoice=INVOICE), fresh("the unapply")),
        erp.perform(apply, key),
    ]
    assert [_found(answer) for answer in answers] == [Found.CREATED] * 3
    assert (erp.keyed(INVOICE), erp.residual(INVOICE)) == (0, 3000)


@pytest.mark.conformance
@pytest.mark.adversarial
@pytest.mark.parametrize("arm", ERPS)
def test_a_payment_applies_across_an_invoices_installments(arm):
    erp, invoice = another_invoice(arm, Decimal(1000), installments=True)
    apply, unapply = Apply(entry=PAYMENT, invoice=invoice), Unapply(entry=PAYMENT, invoice=invoice)
    applied = [_found(erp.perform(apply, fresh("the apply"))) for _ in range(2)]
    assert applied == [Found.CREATED, Found.EXISTING]
    assert (erp.residual(invoice), erp.residual(PAYMENT)) == (0, 96000)
    assert _found(erp.perform(unapply, fresh("the unapply"))) is Found.CREATED
    assert (erp.residual(invoice), erp.residual(PAYMENT)) == (1000, 97000)


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
    for invoice in (small, INVOICE):
        erp.perform(Apply(entry=PAYMENT, invoice=invoice), fresh("the apply"))
    assert (erp.residual(small), erp.residual(INVOICE)) == (0, 3500)
    erp.perform(Unapply(entry=PAYMENT, invoice=INVOICE), fresh("the unapply"))
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
    erp.perform(Apply(entry=memo.entry, invoice=INVOICE), fresh("the apply"))
    reverse = Reverse(entry=memo.entry, reason="reversal")
    match erp.perform(reverse, fresh("the reversal")):
        case Refused():
            pass
        case unexpected:
            raise AssertionError(unexpected)
    assert [e.kind for e in erp.entries(INVOICE)][-1] is Kind.CREDIT_MEMO
    erp.perform(Unapply(entry=memo.entry, invoice=INVOICE), fresh("the unapply"))
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
