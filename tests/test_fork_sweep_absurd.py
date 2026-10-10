"""The parallel marginal sweep on ABSURD/Postgres: the deployed engine's half of the sweep.

The 0<->N sibling of `test_fork_sweep.py`, kept in lockstep deliberately: a fork feature proven on
one engine is not thereby proven on the other, and this pair is where the two engines' genuinely
different mechanics show up in one shape:

- **spawn**: an independent connection (never the running task's ctx connection, which is
  reentrant), with the SDK's own `idempotency_key`;
- **events**: BROADCAST on both engines: the name is the whole address, so neither child needs a
  `reply_to`;
- **the seed**: `read_absurd_task` vs `read_sqlite_conn`, the reader pair whose parity lets one
  child body serve both.

Needs Postgres/Absurd (`just pgt-up`); auto-skips otherwise. Unique run ids **and unique message
ids** per test: the events table is global per queue and an answer is permanent, so a name reused
across runs is answered before it is asked (pinned below).
"""

import uuid
from contextlib import suppress
from functools import partial

import psycopg
import pytest
from _conformance import refusals_of
from _durable import DSN, IMMEDIATE_RETRY, absurd, pg_ready
from _spawning import absurd_spawner
from test_fork_sweep import decision_wf, new_message_id

from effective.api import append_ledger, await_event, call_tool
from effective.bridge_absurd import read_absurd_task
from effective.budget import BUDGET_DEPTH_PARAM
from effective.cost import MeteredInterpreter, Usage
from effective.domain import SPAWN_TOOL, SpawnArgs, SpawnResult
from effective.engines.absurd import ConcurrentAbsurdCtx
from effective.fork import ForkOutcome, join_fork, marginal_sweep, run_fork_as_task, spawn_fork
from effective.govern import Refused
from effective.handlers.absurd import DurableHandler
from effective.handlers.base import op_key
from effective.interpreters.tools import make_tool_runner, spawn_tool
from effective.keys import Key, Segment, compose_key
from effective.ledger import PostgresLedger
from effective.ops import AppendLedgerRow, LedgerRow
from effective.parked import read_absurd_parked
from effective.spawning import Returned

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


def _rows(conn, run_id: str) -> list[tuple[str, str, bool]]:
    # lint: ledger-read-not-canonical: a fork's own hypothetical lineage is exactly what this
    # asserts on — the canonical view is the subject of the base-untouched check below.
    return list(
        conn.execute(
            t"SELECT event_id, kind, hypothetical FROM ledger "
            t"WHERE workflow_run_id = {run_id} ORDER BY seq"
        ).fetchall()
    )


def _run_until(app, task_id: str, max_batches: int = 64):
    for _ in range(max_batches):
        snap = app.fetch_task_result(task_id)
        if snap is not None and snap.state in ("completed", "failed", "cancelled"):
            return snap
        app.work_batch()
    return app.fetch_task_result(task_id)


def test_a_sweep_returns_one_marginal_per_delta_on_the_deployed_engine(conn):
    app, spawner_app = absurd(), absurd()
    base_rid = f"b{uuid.uuid4().hex[:8]}"
    message_id = new_message_id()
    deltas = {f"{base_rid}-approve": "approve", f"{base_rid}-reject": "reject"}
    try:
        # --- the canonical base run: parks at review, decides `reject` ----------------------
        @app.register_task(f"base-{base_rid}")
        def base_task(params, ctx):
            ledger = PostgresLedger(DSN, workflow_run_id=params["run_id"])
            try:
                return DurableHandler(ConcurrentAbsurdCtx(ctx), _domain(), ledger=ledger).run(
                    lambda: decision_wf(params["message_id"])
                )
            finally:
                ledger.close()

        base_id = app.spawn(
            f"base-{base_rid}",
            {"run_id": base_rid, "message_id": message_id},
            retry_strategy=IMMEDIATE_RETRY,
        )["task_id"]
        _run_until(app, base_id)
        # It PARKED. Without this assertion the base is green whenever the
        # queue happens to hold an answer for its await name, which on this engine is forever.
        assert [
            p.wake_event for p in read_absurd_parked(conn) if p.task_name == f"base-{base_rid}"
        ] == [f"review:{message_id}"]
        app.emit_event(f"review:{message_id}", {"decision": "reject"})
        snap = _run_until(app, base_id)
        assert snap is not None
        assert snap.state == "completed", snap
        assert snap.result == "reject"
        base_rows = _rows(conn, base_rid)

        # --- the fork child: one task body, both engines (the reader is the seam) -----------
        app.register_task(f"child-{base_rid}", default_max_attempts=3)(
            partial(
                run_fork_as_task,
                # the child inherits the BASE's message id — see `decision_wf`'s naming rule
                workflow=lambda _child_run_id: decision_wf(message_id),
                domain=_domain(),
                hypothetical_ledger=lambda rid: PostgresLedger(
                    DSN, workflow_run_id=rid, hypothetical=True
                ),
                read_base=lambda tid: read_absurd_task(conn, tid),
            )
        )

        # An independent connection, never the running task's. Events are broadcast, so the
        # done-event name is the address and the child needs no `reply_to`.
        spawner = absurd_spawner(spawner_app, IMMEDIATE_RETRY)

        @app.register_task(f"sweep-{base_rid}", default_max_attempts=3)
        def sweep_task(params, ctx):
            def sweep():
                # the same `src/` combinator the SQLite sweep uses — one workflow, both engines
                return (
                    yield from marginal_sweep(
                        f"child-{base_rid}",
                        base_task_id=base_id,
                        through=f"ledger;extracted:{message_id}",
                        fork_point=compose_key(t"review:{Segment(message_id)}"),
                        forked_from=base_rid,
                        forked_at_event=f"extracted:{message_id}",
                        deltas={cid: {"decision": d} for cid, d in deltas.items()},
                    )
                )

            domain = MeteredInterpreter(
                llm=lambda _op: ("unused", Usage()),
                tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(spawner)}),
            )
            return DurableHandler(ConcurrentAbsurdCtx(ctx), domain).run(sweep)

        sweep_id = app.spawn(
            f"sweep-{base_rid}", {"run_id": f"s{base_rid}"}, retry_strategy=IMMEDIATE_RETRY
        )["task_id"]
        snap = _run_until(app, sweep_id, max_batches=128)

        assert snap is not None
        assert snap.state == "completed", snap
        outcomes = [ForkOutcome.model_validate(o) for o in snap.result]
        assert [o.answer for o in outcomes] == [
            Returned(value="approve"),
            Returned(value="reject"),
        ]

        for child_run_id, decision in deltas.items():
            kinds = [k for _, k, _ in _rows(conn, child_run_id)]
            assert kinds[0] == "forked"
            assert kinds[-1] == "fork_sealed"
            assert ("committed" in kinds) == (decision == "approve")
            assert all(hyp for _, _, hyp in _rows(conn, child_run_id))
        assert _rows(conn, base_rid) == base_rows  # the canonical lineage never moved

        # The CHILD TASKS must reach `completed` too, not merely answer. Without this the sweep
        # passed while every child died on the way out (a Pydantic return value the SDK cannot
        # serialize): the emit is checkpointed and happens BEFORE the return, so the parent got
        # its answer from a task that then failed and retried until it was out of attempts. The
        # only visible symptom was the shared test queue filling with sleeping children.
        for child_run_id in deltas:
            state = conn.execute(
                t"SELECT state FROM absurd.t_default WHERE task_name = {f'child-{base_rid}'} "
                t"AND params->>'child_run_id' = {child_run_id}"
            ).fetchone()
            assert state == ("completed",), f"{child_run_id}: {state}"
    finally:
        app.close()
        spawner_app.close()


def test_an_answer_is_forever_so_a_fresh_run_needs_a_fresh_message_id(conn):
    """A reproduction rather than a rule in a docstring: on this engine an answer is a permanent,
    queue-global fact, so a workflow whose await name repeats is answered before it is asked, and
    the run never parks at all.

    Three runs of one registered task, differing only in the subject each is handed:

    | run | message id | outcome                                                          |
    |-----|------------|------------------------------------------------------------------|
    | 1   | fresh      | **parks**, is answered `reject`, completes                       |
    | 2   | the SAME   | completes on run 1's answer, having registered no wake           |
    | 3   | fresh      | **parks**, which is what run 2 shows is not automatic            |

    Run 2 is the engine's own semantics (`absurd.sql`: exactly one NULL→payload transition per
    event name, ever), pinned here so the rule carries its reason with it. A workflow awaiting a
    module constant such as `review:m1` completes without parking on any container that has seen
    one suite run, and a sweep test over it stays green only because its emitted payload happens
    to match the stale one."""
    app = absurd()
    tag = f"m2{uuid.uuid4().hex[:8]}"
    name = f"base-{tag}"
    try:

        @app.register_task(name, default_max_attempts=3)
        def base_task(params, ctx):
            ledger = PostgresLedger(DSN, workflow_run_id=params["run_id"])
            try:
                return DurableHandler(ConcurrentAbsurdCtx(ctx), _domain(), ledger=ledger).run(
                    lambda: decision_wf(params["message_id"])
                )
            finally:
                ledger.close()

        def spawn_and_drain(run_id: str, message_id: str):
            task_id = app.spawn(
                name, {"run_id": run_id, "message_id": message_id}, retry_strategy=IMMEDIATE_RETRY
            )["task_id"]
            _run_until(app, task_id)
            parked = [p.wake_event for p in read_absurd_parked(conn) if p.task_id == task_id]
            return task_id, parked

        # 1 — a fresh subject parks, is answered, and reaches a TERMINAL state
        first = new_message_id()
        task_1, parked_1 = spawn_and_drain(f"{tag}-1", first)
        assert parked_1 == [f"review:{first}"]
        app.emit_event(f"review:{first}", {"decision": "reject"})
        snap = _run_until(app, task_1)
        assert snap is not None
        assert (snap.state, snap.result) == ("completed", "reject")

        # 2 — the same subject: no park, and the answer arrives from a run that already ended
        task_2, parked_2 = spawn_and_drain(f"{tag}-2", first)
        assert parked_2 == []  # it never registered a wake...
        snap = app.fetch_task_result(task_2)
        assert snap is not None
        assert (snap.state, snap.result) == ("completed", "reject")  # ...and decided anyway

        # 3 — a fresh subject parks again, which run 2 proves is a property of the NAME, not of
        # the engine having been restarted
        second = new_message_id()
        task_3, parked_3 = spawn_and_drain(f"{tag}-3", second)
        assert parked_3 == [f"review:{second}"]
        app.emit_event(f"review:{second}", {"decision": "approve"})
        snap = _run_until(app, task_3)  # drained to terminal: the shared queue is left clean
        assert snap is not None
        assert (snap.state, snap.result) == ("completed", "approve")
    finally:
        app.close()


def test_one_idempotency_key_enqueues_one_task_on_the_deployed_engine(conn):
    """Two spawns under one idempotency key enqueue one task on Absurd. This pins the engine's
    dedupe, which a spawn's crash window rests on; whether a spawn sends the same key when it
    retries is pinned where spawns are named, on both engines."""
    app, spawner_app = absurd(), absurd()
    rid = f"i{uuid.uuid4().hex[:8]}"
    seen: list[str] = []
    try:
        # A REGISTERED no-op body, and drained at the end: an unregistered spawn would leave a
        # claimable `pending` task in the shared `default` queue forever, and every later test's
        # `work_batch` would spend a claim on it (three govern-durable tests starved this way
        # before the handler existed — the one-queue hazard the justfile warns about).
        @app.register_task(f"noop-{rid}", default_max_attempts=1)
        def noop(params, ctx):
            return {"ok": True}

        spawner = absurd_spawner(spawner_app, IMMEDIATE_RETRY, seen)

        first = spawner(f"noop-{rid}", {"a": 1}, rid, "default")
        second = spawner(f"noop-{rid}", {"a": 1}, rid, "default")
        assert first == second  # same key -> one task
        assert len(set(seen)) == 1

        # psycopg takes the `Template` DIRECTLY — no `bind`, which is the sqlite3-side qmark
        # processor. Splatting one here passes `?` text plus a parameter tuple to a driver that
        # sees zero placeholders (measured: `ProgrammingError`). Two engines, two boundary forms.
        rows = conn.execute(
            t"SELECT count(*) FROM absurd.t_default WHERE task_id = {first}::uuid"
        ).fetchall()
        assert rows == [(1,)]
        _run_until(app, first)  # drain it so the queue is left clean for the next test
    finally:
        app.close()
        spawner_app.close()


def test_a_refused_child_answers_its_parent_on_the_DEPLOYED_engine(conn):
    """A refused fork reports back rather than hanging its parent — on Absurd, where the reply
    crosses the queue.

    **The SQLite twin (`test_fork_sweep.py`) does not cover this.** The refusal has to travel
    from a dying child to a waiting parent. Both engines broadcast events, so the hop is shared,
    but a relay proven on one engine says nothing about the other, and without this test the
    deployed engine pins only the happy path (every answer `Returned`), leaving the failure mode
    the mechanism exists for unmeasured in production.

    The refusal itself is engine-agnostic (`SeedingCtx.await_event` raising `ForkedPrefixAwait` is
    plain Python over a ctx wrapper) and is already pinned three times, so this deliberately does
    NOT re-assert it. What is asserted is only what the relay adds: the parent COMPLETES carrying
    the named reason, and the refused lineage is left unsealed."""
    app, spawner_app = absurd(), absurd()
    base_rid = f"r{uuid.uuid4().hex[:8]}"
    message_id = new_message_id()
    child_run_id = f"{base_rid}-bad"
    try:

        @app.register_task(f"base-{base_rid}")
        def base_task(params, ctx):
            ledger = PostgresLedger(DSN, workflow_run_id=params["run_id"])
            try:
                return DurableHandler(ConcurrentAbsurdCtx(ctx), _domain(), ledger=ledger).run(
                    lambda: decision_wf(params["message_id"])
                )
            finally:
                ledger.close()

        base_id = app.spawn(
            f"base-{base_rid}",
            {"run_id": base_rid, "message_id": message_id},
            retry_strategy=IMMEDIATE_RETRY,
        )["task_id"]
        _run_until(app, base_id)
        app.emit_event(f"review:{message_id}", {"decision": "reject"})
        snap = _run_until(app, base_id)
        assert snap is not None
        assert snap.state == "completed", snap

        app.register_task(f"child-{base_rid}", default_max_attempts=3)(
            partial(
                run_fork_as_task,
                workflow=lambda _child_run_id: decision_wf(message_id),
                domain=_domain(),
                hypothetical_ledger=lambda rid: PostgresLedger(
                    DSN, workflow_run_id=rid, hypothetical=True
                ),
                read_base=lambda tid: read_absurd_task(conn, tid),
            )
        )

        spawner = absurd_spawner(spawner_app, IMMEDIATE_RETRY)

        @app.register_task(f"bad-sweep-{base_rid}", default_max_attempts=3)
        def bad_sweep(params, ctx):
            def sweep():
                handle = yield from spawn_fork(
                    f"child-{base_rid}",
                    child_run_id=child_run_id,
                    base_task_id=base_id,
                    through=f"ledger;extracted:{message_id}",
                    fork_point=Key.parse("never-fires"),  # the workflow awaits review:{message_id}
                    forked_from=base_rid,
                    forked_at_event=f"extracted:{message_id}",
                    delta={"decision": "approve"},
                )
                return (yield from join_fork(handle))

            domain = MeteredInterpreter(
                llm=lambda _op: ("unused", Usage()),
                tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(spawner)}),
            )
            return DurableHandler(ConcurrentAbsurdCtx(ctx), domain).run(sweep)

        sweep_id = app.spawn(
            f"bad-sweep-{base_rid}", {"run_id": f"s{base_rid}"}, retry_strategy=IMMEDIATE_RETRY
        )["task_id"]
        snap = _run_until(app, sweep_id, max_batches=128)

        assert snap is not None
        assert snap.state == "completed", snap  # the PARENT completed — no hang
        [(kind, message)] = refusals_of(ForkOutcome.model_validate(snap.result))
        assert kind == "ForkedPrefixAwait"  # the reason survived the broadcast hop
        assert "never-fires" in message

        assert [k for _, k, _ in _rows(conn, child_run_id)] == ["forked"]  # unsealed

        # Terminal, not merely answered — the rule this file learned the hard way.
        state = conn.execute(
            t"SELECT state FROM absurd.t_default WHERE task_name = {f'child-{base_rid}'} "
            t"AND params->>'child_run_id' = {child_run_id}"
        ).fetchone()
        assert state == ("completed",), state
    finally:
        app.close()
        spawner_app.close()


def test_a_fork_of_a_base_whose_spawn_was_refused_replays_the_refusal_on_the_deployed_engine(
    conn,
):
    """Reddens if a base that routed around a refused spawn cannot be forked on Absurd, whose
    seed reader is its own: the refusal must reach the fork child as the step's recorded value."""
    app, spawner_app = absurd(), absurd()
    rid = f"f{uuid.uuid4().hex[:8]}"
    message_id = new_message_id()
    extracted = LedgerRow(
        event_id=compose_key(t"extracted:{Segment(message_id)}"), kind="extracted"
    )
    review = compose_key(t"review:{Segment(message_id)}")

    spawner = absurd_spawner(spawner_app, IMMEDIATE_RETRY)

    def spawning():
        return MeteredInterpreter(
            llm=lambda _op: ("unused", Usage()),
            tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(spawner)}),
        )

    def base_wf():
        leaf = SpawnArgs(task_name=f"leaf-{rid}", params={})
        with suppress(Refused):
            yield from call_tool(SPAWN_TOOL, leaf.model_dump(exclude_none=True), SpawnResult)
        yield from append_ledger(extracted)
        approval = yield from await_event(review, dict)
        return approval["decision"]

    try:

        @app.register_task(f"leaf-{rid}", default_max_attempts=1)
        def leaf(params, ctx):
            return "leaf"

        @app.register_task(f"base-{rid}")
        def base_task(params, ctx):
            ledger = PostgresLedger(DSN, workflow_run_id=rid)
            try:
                handler = DurableHandler(
                    ConcurrentAbsurdCtx(ctx), spawning(), ledger=ledger, params=params
                )
                return handler.run(base_wf)
            finally:
                ledger.close()

        base_id = app.spawn(
            f"base-{rid}", {BUDGET_DEPTH_PARAM: 0}, retry_strategy=IMMEDIATE_RETRY
        )["task_id"]
        _run_until(app, base_id)
        app.emit_event(review.stored(), {"decision": "approve"})
        snap = _run_until(app, base_id)
        assert snap is not None
        assert snap.state == "completed", snap

        app.register_task(f"child-{rid}", default_max_attempts=3)(
            partial(
                run_fork_as_task,
                workflow=lambda _child_run_id: base_wf(),
                domain=spawning(),
                hypothetical_ledger=lambda run: PostgresLedger(
                    DSN, workflow_run_id=run, hypothetical=True
                ),
                read_base=lambda task: read_absurd_task(conn, task),
            )
        )

        @app.register_task(f"sweep-{rid}", default_max_attempts=3)
        def sweep_task(params, ctx):
            def sweep():
                handle = yield from spawn_fork(
                    f"child-{rid}",
                    child_run_id=f"{rid}-reject",
                    base_task_id=base_id,
                    through=op_key(AppendLedgerRow(row=extracted)).stored(),
                    fork_point=review,
                    forked_from=rid,
                    forked_at_event=extracted.event_id.stored(),
                    delta={"decision": "reject"},
                )
                return (yield from join_fork(handle))

            return DurableHandler(ConcurrentAbsurdCtx(ctx), spawning()).run(sweep)

        sweep_id = app.spawn(f"sweep-{rid}", {}, retry_strategy=IMMEDIATE_RETRY)["task_id"]
        snap = _run_until(app, sweep_id, max_batches=128)

        assert snap is not None
        assert snap.state == "completed", snap
        assert ForkOutcome.model_validate(snap.result).answer == Returned(value="reject")
    finally:
        app.close()
        spawner_app.close()
