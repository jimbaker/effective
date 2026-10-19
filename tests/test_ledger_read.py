"""A read of the ledger: a workflow reads the row on record under an id, and a replay reads what
the first attempt read."""

from functools import partial
from typing import Any
from uuid import uuid4

import pytest
from _conformance import CountingDomain, Fault, FaultPosition, at_every_op

from effective.api import await_event, claim_ledger, read_ledger
from effective.counterfactual import ForkLedger, fork_scoped
from effective.engines.sqlite import SqliteApp, SqliteLedger
from effective.handlers.durable import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.keys import Key, Name, Segment, compose_key
from effective.ops import LedgerRow, ReadLedgerRow, UnreadLedger
from effective.sandbox import inspect_only


def _fact(fact: str) -> Key:
    return compose_key(t"terms:{Name(fact)}")


def _reads(fact: str, _run_id: str):
    return (yield from read_ledger(_fact(fact)))


def _run(backend, program) -> object:
    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, program, None, Fault(None), ())
    snapshot = backend.run_until_result(backend.spawn(run_id, run_id))
    assert snapshot.state == "completed", snapshot
    return snapshot.result


def test_a_read_answers_the_row_on_record_or_none(backend):
    fact = f"inv-{uuid4().hex}"
    assert _run(backend, partial(_reads, fact)) is None
    backend.ledger_append("writer", _fact(fact).stored(), "terms")
    assert _run(backend, partial(_reads, fact)) == {
        "event_id": _fact(fact).stored(),
        "kind": "terms",
    }


def _credit(payload: dict[str, Any] | None) -> str | None:
    return None if payload is None else payload["credit"]


def test_the_recorder_reads_the_row_it_holds():
    def appended_between():
        before = yield from read_ledger(_fact("inv"))
        yield from claim_ledger(LedgerRow(event_id=_fact("inv"), kind="terms", credit="1000"))
        return _credit(before), _credit((yield from read_ledger(_fact("inv"))))

    assert RecordingHandler(responses={}).run(appended_between) == (None, "1000")


def _read_then_waited(fact: str, run_id: str):
    read = yield from read_ledger(_fact(fact))
    yield from await_event(f"go:{run_id}", dict)
    return read


def test_a_replay_reads_what_the_first_attempt_read(backend):
    """The row lands while the run is parked; the attempt that resumes replays the read."""
    fact, run_id = f"inv-{uuid4().hex}", f"r-{uuid4().hex[:8]}"
    backend.register(run_id, partial(_read_then_waited, fact), None, Fault(None), ())
    task_id = backend.spawn(run_id, run_id)
    backend.run_until_result(task_id)
    backend.ledger_append("writer", _fact(fact).stored(), "terms")
    backend.emit_event(task_id, f"go:{run_id}", {})
    snapshot = backend.run_until_result(task_id)
    assert (snapshot.state, snapshot.result) == ("completed", None)


def _read_then_claimed(fact: str, _run_id: str):
    before = yield from read_ledger(_fact(fact))
    row = LedgerRow(event_id=_fact(fact), kind="terms", credit="1000")
    held = yield from claim_ledger(row)
    return before, held["credit"], _credit((yield from read_ledger(_fact(fact))))


@pytest.mark.parametrize("position", list(FaultPosition), ids=lambda p: p.name.lower())
def test_a_crash_at_every_op_of_a_read_and_a_claim_resumes_to_one_result(backend, position):
    unarmed = Fault(None, position=position)
    fact = f"inv-{uuid4().hex}"
    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, partial(_read_then_claimed, fact), None, unarmed, ())
    clean = backend.run_until_result(backend.spawn(run_id, run_id)).result
    assert clean == [None, "1000", "1000"]
    for k, fault in at_every_op(unarmed):
        fact, run_id = f"inv-{uuid4().hex}", f"r-{uuid4().hex[:8]}"
        backend.register(run_id, partial(_read_then_claimed, fact), None, fault, ())
        snapshot = backend.run_until_result(backend.spawn(run_id, run_id))
        assert snapshot.result == clean, f"k={k}: {snapshot}"


def test_a_fork_reads_its_own_row_and_not_the_canonical_one(tmp_path):
    app = SqliteApp(str(tmp_path / "ledger.db"))
    try:
        canonical = SqliteLedger(app.conn, "base", app.write_lock)
        hypothetical = SqliteLedger(app.conn, "child", app.write_lock, hypothetical=True)
        fork = ForkLedger(hypothetical, child_run_id=Segment("cf-a"))
        canonical.append(LedgerRow(event_id=_fact("inv"), kind="terms", credit="1000"))
        assert fork.read(_fact("inv")) is None
        fork.append(LedgerRow(event_id=_fact("inv"), kind="terms", credit="3000"))
        assert (fork.read(_fact("inv")), canonical.read(_fact("inv"))) == (
            {
                "event_id": fork_scoped(Segment("cf-a"), _fact("inv")).stored(),
                "kind": "terms",
                "credit": "3000",
            },
            {"event_id": _fact("inv").stored(), "kind": "terms", "credit": "1000"},
        )
    finally:
        app.close()


def test_a_read_with_no_ledger_is_refused(tmp_path):
    """A run with no ledger writer cannot say a row is absent from a ledger it does not hold."""
    app = SqliteApp(str(tmp_path / "unread.db"))
    try:
        app.register_task("reads")(
            lambda params, ctx: DurableHandler(ctx, CountingDomain(), ledger=None).run(
                partial(_reads, "inv", "r")
            )
        )
        done = app.run_until_result(app.spawn("reads", {"run_id": "r"}))
        assert done is not None
        assert done.state == "failed", done
        assert "no ledger is held" in str(done.failure)
    finally:
        app.close()


def test_a_counterfactual_in_process_refuses_a_read():
    with pytest.raises(UnreadLedger):
        inspect_only(ReadLedgerRow(event_id=_fact("inv")), {})


def test_each_lineage_reads_only_its_own_rows(tmp_path):
    app = SqliteApp(str(tmp_path / "lineages.db"))
    try:
        canonical = SqliteLedger(app.conn, "base", app.write_lock)
        hypothetical = SqliteLedger(app.conn, "arm", app.write_lock, hypothetical=True)
        hypothetical.append(LedgerRow(event_id=_fact("arm"), kind="terms"))
        canonical.append(LedgerRow(event_id=_fact("base"), kind="terms"))
        reads = [
            ledger.read(_fact(fact))
            for ledger in (canonical, hypothetical)
            for fact in ("base", "arm")
        ]
        assert [read is None for read in reads] == [False, True, True, False]
    finally:
        app.close()
