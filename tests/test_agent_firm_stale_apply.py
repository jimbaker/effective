"""A settlement whose first attempt's lease expired with an ERP op in flight, reclaimed by a
second worker, at every op the lease can expire at and every point the stale op can land, on the
scripted Odoo and both engines.

| control           | what holds                                                    |
|-------------------|---------------------------------------------------------------|
| each apply once   | an apply an expired attempt sent lands nothing after its undo |
| undo names apply  | each undo names the key of the apply it removes               |
| rejection unwinds | the run reports the compensation the books show               |
"""

from decimal import Decimal
from itertools import count
from uuid import uuid4

import _attempts
import psycopg
import pytest
from _attempts import Overlap, Watched, overlapped
from _conformance import PG_DSN, Fault
from _odoo import fresh_round

from examples.agent_firm.ar import REMIND, settle
from examples.agent_firm.cases import SLA_CAP
from examples.agent_firm.erp.port import Apply, Erp, ErpOp, Kind, Unapply
from examples.agent_firm.erp.scripted import ScriptedOdoo
from examples.agent_firm.saga import Compensated, Completed
from examples.agent_firm.world import World

INVOICE, PAYMENT = SLA_CAP.invoice.id, SLA_CAP.payment.id
NOTICES = {"lands": {"ok": True}, "bounces": {"ok": False, "reason": "bounced"}}
ERP_OPS = {"lands": 3, "bounces": 6}
"""The ERP ops one settlement sends. A lease expiring at op `at` leaves the reclaiming attempt
`sent - at + 1` calls, so its held op lands at `sent - at + 2` points; summed over `at`, that is
`sent * (sent + 3) / 2` cells."""


def _tools(notice: str) -> dict:
    return {"notify": lambda _: NOTICES[notice], REMIND: lambda _: {"ok": True}}


def _named_applies(performed: list[tuple[ErpOp, str]]) -> list[bool]:
    """For each undo, whether the key it names is an apply of the entry and invoice it removes."""
    applies: dict[str, tuple[str, str]] = {}
    undone: list[tuple[str, tuple[str, str]]] = []
    for op, key in performed:
        match op:
            case Apply(entry=entry, invoice=invoice):
                applies[key] = (entry, invoice)
            case Unapply(entry=entry, invoice=invoice, undoes=undoes):
                undone.append((undoes, (entry, invoice)))
    return [applies.get(key) == removed for key, removed in undone]


def _settle_once_overlapped(backend, notice: str, at: int, before: int):
    """The settlement run under one overlap, or `None` when its first attempt has no op `at`."""
    odoo = ScriptedOdoo.seeded([SLA_CAP])
    overlap = Overlap(odoo.erp(), at, before)
    attempt = overlap.attempt(lambda erp: World(erp=erp, tools=_tools(notice)).domain())
    program, name = settle(SLA_CAP, Decimal(1000), "r", fresh_round()), f"stale-{uuid4().hex[:8]}"
    domain = World(tools=_tools(notice)).domain()
    backend.register(name, lambda _: program(), domain, Fault(), (), fresh=attempt)
    task = backend.spawn(name, name)
    if not overlapped(backend, task, overlap):
        return None
    return overlap, backend.run_until_result(task), odoo.erp()


def _books_hold(notice: str, run, books: Erp, cell: tuple[int, int]) -> None:
    memos = [e for e in books.entries(INVOICE) if e.kind is Kind.CREDIT_MEMO]
    match notice:
        case "lands":
            assert "notify" in Completed.model_validate(run.result).landed
            assert len(memos) == 1, ("one memo", cell)
            owed = (books.residual(PAYMENT), books.residual(INVOICE))
            assert owed == (0, 2000), ("each apply once", cell)
        case _:
            undone = Compensated.model_validate(run.result).undone
            assert undone == ["apply-memo", "apply-payment", "credit-memo"], cell
            (memo,) = memos
            (reversal,) = [e for e in books.entries(INVOICE) if e.kind is Kind.REVERSAL]
            owed = [books.residual(d) for d in (INVOICE, PAYMENT, memo.id, reversal.id)]
            assert owed == [100000, 97000, 0, 0], ("each apply once", cell)


@pytest.mark.adversarial
@pytest.mark.parametrize("notice", NOTICES)
def test_an_op_sent_by_an_expired_attempt_lands_once_wherever_it_lands(backend, notice):
    """At every ERP op the first attempt's lease can expire at, the held op lands before each of
    the reclaiming attempt's calls and after its last, and the books show one settlement."""
    cells = 0
    for at in count(1):
        for before in count(0):
            ran = _settle_once_overlapped(backend, notice, at, before)
            if ran is None:
                break
            overlap, run, books = ran
            cell = (at, before)
            assert overlap.overlapping, ("the attempts overlapped", cell)
            assert run.state == "completed", (cell, run)
            assert all(_named_applies(overlap.performed)), ("undo names apply", cell)
            _books_hold(notice, run, books, cell)
            cells += 1
            if len(overlap.reclaiming.performed) <= before:
                break
        if ran is None and before == 0:
            break
    sent = ERP_OPS[notice]
    assert cells == sent * (sent + 3) // 2


@pytest.mark.adversarial
def test_each_undo_of_a_settlement_names_the_apply_it_removes(backend):
    erp = Watched(ScriptedOdoo.seeded([SLA_CAP]).erp())
    world = World(erp=erp, tools=_tools("bounces"))
    program = settle(SLA_CAP, Decimal(1000), "r", fresh_round())
    name = f"undo-{uuid4().hex[:8]}"
    backend.register(name, lambda _: program(), world.domain(), Fault(), ())
    run = backend.run_until_result(backend.spawn(name, name))
    assert Compensated.model_validate(run.result).undone[:2] == ["apply-memo", "apply-payment"]
    assert _named_applies(erp.performed) == [True, True]


@pytest.mark.adversarial
def test_a_retry_of_the_first_worker_holds_no_op(backend):
    """The first attempt crashes before its first op and the same worker retries the task. The
    retry runs over the reclaiming ERP, so it holds nothing, and the cell is no overlap."""
    overlap = Overlap(ScriptedOdoo.seeded([SLA_CAP]).erp(), 1, ERP_OPS["lands"] + 1)
    attempt = overlap.attempt(lambda erp: World(erp=erp, tools=_tools("lands")).domain())
    program, name = settle(SLA_CAP, Decimal(1000), "r", fresh_round()), f"retry-{uuid4().hex[:8]}"
    domain = World(tools=_tools("lands")).domain()
    backend.register(name, lambda _: program(), domain, Fault(1), (), fresh=attempt)
    assert not overlapped(backend, backend.spawn(name, name), overlap)
    assert (len(overlap.starts), overlap.stale.performed) == (2, [])


def _queued_ahead(backend, task) -> None:
    """Make `task` claimable ahead of everything the queue holds."""
    match backend.name:
        case "sqlite":
            with backend.app.write_lock:
                backend.app.conn.execute(
                    "UPDATE tasks SET available_at=0 WHERE task_id=?", (task,)
                )
        case _:
            with psycopg.connect(PG_DSN, autocommit=True) as conn:
                conn.execute(
                    t"UPDATE absurd.r_default SET available_at = now() - interval '1 day' "
                    t"WHERE task_id = {task}::uuid"
                )


@pytest.mark.adversarial
def test_a_task_queued_ahead_of_the_reclaim_does_not_stand_in_for_it(backend, monkeypatch):
    """A worker claims the oldest task first, so the reclaiming worker runs one queued ahead of the
    expired attempt and then the task itself, and the overlap is the task's own."""
    ran, name = [], f"ahead-{uuid4().hex[:8]}"
    backend.register_body(name, lambda params, ctx: ran.append(name))
    expire = _attempts.expire

    def queued_then_expired(backend, task) -> None:
        _queued_ahead(backend, backend.spawn(name, name))
        expire(backend, task)

    monkeypatch.setattr(_attempts, "expire", queued_then_expired)
    cell = _settle_once_overlapped(backend, "lands", 1, 0)
    assert cell is not None
    overlap, run, books = cell
    assert overlap.overlapping
    assert (ran, len(overlap.reclaiming.performed)) == ([name], ERP_OPS["lands"])
    _books_hold("lands", run, books, (1, 0))


def test_a_hold_that_waited_out_its_release_overlaps_nothing(monkeypatch):
    """A hold whose release never came raises before its op lands, so a second attempt starting
    after it is no overlap."""
    monkeypatch.setattr(_attempts, "TIMEOUT", 0)
    overlap = Overlap(ScriptedOdoo.seeded([SLA_CAP]).erp(), 1, 0)
    fresh = overlap.attempt(lambda erp: erp)
    fresh()
    with pytest.raises(AssertionError, match="nothing let the stale op 1 go"):
        overlap.stale.before(0)
    fresh()
    assert (overlap.holding, overlap.abandoned, overlap.overlapping) == (True, True, False)
