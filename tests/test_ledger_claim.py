"""A claim on the ledger: the first run to append an id holds it, and every run reads its row."""

from functools import partial
from uuid import uuid4

from _conformance import Fault

from effective.api import claim_ledger, gather
from effective.handlers.recording import RecordingHandler
from effective.keys import Name, compose_key
from effective.ops import LedgerRow


def _claims(fact: str, credit: str):
    row = LedgerRow(event_id=compose_key(t"terms:{Name(fact)}"), kind="terms", credit=credit)
    return (yield from claim_ledger(row))


def _claimed(fact: str, credit: str, _run_id: str):
    return _claims(fact, credit)


def test_two_runs_claiming_one_id_read_the_first_runs_row(backend):
    fact, held = f"inv-{uuid4().hex}", []
    for credit in ("1000", "3000"):
        run_id = f"r-{uuid4().hex[:8]}"
        backend.register(run_id, partial(_claimed, fact, credit), None, Fault(None), ())
        snapshot = backend.run_until_result(backend.spawn(run_id, run_id))
        assert snapshot.state == "completed", snapshot
        held.append(snapshot.result["credit"])
    assert held == ["1000", "1000"]


def test_the_recorder_answers_a_claim_with_the_row_it_holds():
    def twice():
        first = yield from _claims("inv", "1000")
        second = yield from _claims("inv", "3000")
        return first["credit"], second["credit"]

    assert RecordingHandler(responses={}).run(twice) == ("1000", "1000")


def _gathered(fact: str, _run_id: str):
    results = yield from gather([partial(_claims, fact, credit) for credit in ("1000", "3000")])
    return [held["credit"] for held in results]


def test_two_branches_of_one_run_claiming_one_id_read_one_row(backend):
    """A claim carries no placed writer, so two placements in one task may claim one id."""
    fact, run_id = f"inv-{uuid4().hex}", f"r-{uuid4().hex[:8]}"
    backend.register(run_id, partial(_gathered, fact), None, Fault(None), ())
    snapshot = backend.run_until_result(backend.spawn(run_id, run_id))
    assert snapshot.state == "completed", snapshot
    (credit,) = set(snapshot.result)
    assert credit in ("1000", "3000")
