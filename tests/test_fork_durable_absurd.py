"""The durable fork's ctx stack on ABSURD/Postgres: the 0<->N half of `test_fork_durable.py`.

The fork path is where the two engines diverge, because `run_fork` builds its wrapper stack from
the RAW task ctx. On SQLite the raw ctx already speaks our `TaskContext` protocol, so a wrapper
stack over it is complete. On Absurd it does NOT: the SDK's ctx is a foreign implementation, and
`SdkCtx` is the adapter that translates it (`Key` -> `str` for `step`, one-arg -> two-arg for
`sleep_until`). `DurableHandler` installs that adapter via `_adapt_ctx`, so a `SeedingCtx` wrapped
around the raw ctx must not hide the SDK ctx from it.

The SQLite pass
(`test_fork_durable.py::test_an_already_elapsed_sleep_in_the_replayed_PREFIX_passes_through`)
stays green with the adapter missing, so only this engine can fail on it.
"""

import uuid
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from _durable import DSN, absurd, pg_ready
from test_fork_sweep import new_message_id

from effective.api import append_ledger, ask_llm, await_event, sleep_until
from effective.bridge_absurd import read_absurd_task
from effective.cost import MeteredInterpreter, Usage
from effective.fork import fork_seed, run_fork
from effective.handlers.durable import DurableHandler, fork_event_name
from effective.keys import Key, Segment, compose_key
from effective.ledger import PostgresLedger
from effective.ops import LedgerRow

pytestmark = pytest.mark.skipif(not pg_ready(), reason="needs Postgres/Absurd (just pgt-up)")


def _domain():
    return MeteredInterpreter(
        llm=lambda _op: ({"amount": "5.00"}, Usage()), tools=lambda _op: "tool-done"
    )


@pytest.fixture
def conn():
    c = psycopg.connect(DSN, autocommit=True)
    yield c
    c.close()


def _rows(conn, run_id: str) -> list[str]:
    # lint: ledger-read-not-canonical: the fork's own hypothetical lineage is the subject here.
    return [
        r[0]
        for r in conn.execute(
            t"SELECT kind FROM ledger WHERE workflow_run_id = {run_id} ORDER BY seq"
        ).fetchall()
    ]


def sleeping_decision_wf(message_id: str):
    """The exemplar plus an already-elapsed PREFIX sleep — the one op whose arity differs
    between our protocol and the SDK's, placed where a fork must REPLAY it rather than refuse it.
    """
    yield from ask_llm("extract", [], dict)
    yield from sleep_until(datetime.now(UTC) - timedelta(days=1))
    yield from append_ledger(
        LedgerRow(event_id=compose_key(t"extracted:{Segment(message_id)}"), kind="extracted")
    )
    approval = yield from await_event(f"review:{message_id}", dict)
    yield from append_ledger(
        LedgerRow(
            event_id=compose_key(t"reviewed:{Segment(message_id)}"),
            kind="reviewed",
            decision=approval["decision"],
        )
    )
    return approval["decision"]


def test_an_elapsed_prefix_sleep_replays_on_the_DEPLOYED_engine(conn):
    """The `Seeding` arm, on Absurd.

    The refusal arm (`Live` -> `ForkedSleep`) never reaches the base ctx, so it is engine-blind.
    The pass-through arm delegates to the real ctx and is therefore the arm that proves the
    adapter is installed. Without the adapter this dies with
    `TypeError: TaskContext.sleep_until() missing 1 required positional argument: 'wake_at'`,
    retries to death, and fails the task.
    """
    app = absurd()
    base_rid = f"b{uuid.uuid4().hex[:8]}"
    fork_rid = f"f{uuid.uuid4().hex[:8]}"
    message_id = new_message_id()

    @app.register_task(f"base-{base_rid}")
    def base_task(params, ctx):
        ledger = PostgresLedger(DSN, workflow_run_id=params["run_id"])
        try:
            return DurableHandler(ctx, _domain(), ledger=ledger).run(
                lambda: sleeping_decision_wf(params["message_id"])
            )
        finally:
            ledger.close()

    base_id = app.spawn(
        f"base-{base_rid}",
        {"run_id": base_rid, "message_id": message_id},
    )
    app.run_until_result(base_id)
    app.emit_event(f"review:{message_id}", {"decision": "reject"})
    snap = app.run_until_result(base_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert snap.result == "reject"

    seed = fork_seed(read_absurd_task(conn, base_id), through=f"ledger;extracted:{message_id}")

    @app.register_task(f"child-{base_rid}", default_max_attempts=3)
    def child_task(params, ctx):
        hyp = PostgresLedger(DSN, workflow_run_id=params["run_id"], hypothetical=True)
        try:
            return run_fork(
                ctx,
                lambda: sleeping_decision_wf(message_id),
                child_run_id=params["run_id"],
                seed=seed,
                hypothetical_ledger=hyp,
                domain=_domain(),
                forked_from=base_rid,
                forked_at_event=f"extracted:{message_id}",
                fork_point=compose_key(t"review:{Segment(message_id)}"),
                delta={"decision": "approve"},
            )
        finally:
            hyp.close()

    fork_id = app.spawn(f"child-{base_rid}", {"run_id": fork_rid})
    app.run_until_result(fork_id)
    app.emit_event(
        fork_event_name(fork_rid, Key.parse(f"review:{message_id}")).stored(),
        {"decision": "approve"},
    )
    snap = app.run_until_result(fork_id)

    assert snap is not None, "the fork never completed"
    assert snap.state == "completed", f"the forked prefix sleep did not replay: {snap.failure}"
    assert snap.result == "approve"
    assert _rows(conn, fork_rid) == ["forked", "reviewed", "fork_sealed"]


def repeating_decision_wf(message_id: str):
    """A repeated step name in the forked TAIL — the `Live` phase, which delegates straight to
    the base ctx and so is the other consumer of the missing adapter."""
    yield from ask_llm("extract", [], dict)
    yield from append_ledger(
        LedgerRow(event_id=compose_key(t"extracted:{Segment(message_id)}"), kind="extracted")
    )
    approval = yield from await_event(f"review:{message_id}", dict)
    yield from ask_llm("probe", [], dict)
    yield from ask_llm("probe", [], dict)  # -> `probe#2`, composed by the SDK from the name
    yield from append_ledger(
        LedgerRow(
            event_id=compose_key(t"reviewed:{Segment(message_id)}"),
            kind="reviewed",
            decision=approval["decision"],
        )
    )
    return approval["decision"]


def test_a_repeated_tail_step_checkpoints_under_a_string_name_not_a_Key_repr(conn):
    """The same missing adapter, on `step` rather than `sleep_until`, and the reason the adapter
    belongs at the ctx stack's base rather than on one op.

    `SdkCtx.step` exists to call `name.stored()`, because the SDK composes `f"{name}#{count}"`
    for a repeated step name. Unadapted, the second occurrence is checkpointed as
    ``Key(_value='probe')#2``: a `Key` repr frozen into a durable checkpoint name, which is
    both unreadable and a fresh identity on every replay whose repr differs.
    """
    app = absurd()
    base_rid = f"b{uuid.uuid4().hex[:8]}"
    fork_rid = f"f{uuid.uuid4().hex[:8]}"
    message_id = new_message_id()

    @app.register_task(f"base-{base_rid}")
    def base_task(params, ctx):
        ledger = PostgresLedger(DSN, workflow_run_id=params["run_id"])
        try:
            return DurableHandler(ctx, _domain(), ledger=ledger).run(
                lambda: repeating_decision_wf(params["message_id"])
            )
        finally:
            ledger.close()

    base_id = app.spawn(
        f"base-{base_rid}",
        {"run_id": base_rid, "message_id": message_id},
    )
    app.run_until_result(base_id)
    app.emit_event(f"review:{message_id}", {"decision": "reject"})
    snap = app.run_until_result(base_id)
    assert snap is not None
    assert snap.state == "completed"

    seed = fork_seed(read_absurd_task(conn, base_id), through=f"ledger;extracted:{message_id}")

    @app.register_task(f"child-{base_rid}", default_max_attempts=3)
    def child_task(params, ctx):
        hyp = PostgresLedger(DSN, workflow_run_id=params["run_id"], hypothetical=True)
        try:
            return run_fork(
                ctx,
                lambda: repeating_decision_wf(message_id),
                child_run_id=params["run_id"],
                seed=seed,
                hypothetical_ledger=hyp,
                domain=_domain(),
                forked_from=base_rid,
                forked_at_event=f"extracted:{message_id}",
                fork_point=compose_key(t"review:{Segment(message_id)}"),
                delta={"decision": "approve"},
            )
        finally:
            hyp.close()

    fork_id = app.spawn(f"child-{base_rid}", {"run_id": fork_rid})
    app.run_until_result(fork_id)
    app.emit_event(
        fork_event_name(fork_rid, Key.parse(f"review:{message_id}")).stored(),
        {"decision": "approve"},
    )
    snap = app.run_until_result(fork_id)
    assert snap is not None
    assert snap.state == "completed", snap

    names = [
        r[0]
        for r in conn.execute(
            t"SELECT checkpoint_name FROM absurd.c_default WHERE task_id = {fork_id}::uuid "
            t"AND status = 'committed' ORDER BY checkpoint_name"
        ).fetchall()
    ]
    assert "step:probe#2" in names, f"the repeated tail step is not string-named: {names}"
    assert not any("Key(" in n for n in names), f"a Key repr reached a checkpoint name: {names}"
