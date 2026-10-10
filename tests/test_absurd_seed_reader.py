"""The Absurd SEED reader (`effective.bridge_absurd.read_absurd_task`) on the real engine.

The 0↔N half of the fork's checkpoint reader — the piece whose absence meant a durable fork could
be proven on SQLite and remain unported on the DEPLOYED engine. Its cross-engine key-sequence
contract is pinned in `test_conformance.py` (parametrized over both engines); what lives here is
the part that has no SQLite counterpart and so cannot be a conformance case:

- the **filtered/unfiltered contrast** with `export_measured_prefix` on one real run — the reason
  a second reader had to exist rather than reusing the first;
- the seed it actually produces, fed to `fork_seed`;
- `AmbiguousCheckpointOrder` — the refusal that exists because `c_{queue}` has no ordinal, so
  `updated_at` is the whole of the commit-order evidence.

Needs Postgres/Absurd (`just pgt-up`); auto-skips otherwise. Unique run ids per test (the Absurd
events table is global per queue and persists across runs).
"""

import uuid
from uuid import UUID

import psycopg
import pytest
from _durable import DSN, absurd, pg_ready

from effective.api import append_ledger, call_tool, step
from effective.bridge_absurd import (
    AmbiguousCheckpointOrder,
    export_measured_prefix,
    read_absurd_task,
)
from effective.checkpoints import keys
from effective.cost import CONTRACT_PARAM, Contract, MeteredInterpreter, Usage
from effective.domain import AskLLM
from effective.engines.absurd import ConcurrentAbsurdCtx
from effective.fork import fork_seed
from effective.handlers.durable import DurableHandler
from effective.keys import Key
from effective.ledger import PostgresLedger
from effective.ops import LedgerRow

pytestmark = pytest.mark.skipif(not pg_ready(), reason="needs Postgres/Absurd (just pgt-up)")


def _domain():
    return MeteredInterpreter(
        llm=lambda _op: ("ans", Usage(prompt_tokens=10, completion_tokens=5, cost=0.001)),
        tools=lambda _op: 7,
    )


def _decision_wf(run_id: str):
    """A prefix (an ask, a tool, a ledger append) and a tail — the shape a decision fork cuts."""
    yield from step("extract", AskLLM(messages="m", response_schema=str))
    yield from call_tool("lookup", {}, int)
    yield from append_ledger(
        LedgerRow(event_id=Key.parse(f"{run_id}:extracted"), kind="extracted")
    )
    yield from append_ledger(LedgerRow(event_id=Key.parse(f"{run_id}:reviewed"), kind="reviewed"))
    return "done"


@pytest.fixture
def app():
    return absurd()


@pytest.fixture
def conn():
    c = psycopg.connect(DSN, autocommit=True)
    yield c
    c.close()


def _run(app, run_id: str) -> UUID:
    name = f"seed-{run_id}"

    @app.register_task(name, default_max_attempts=3)
    def task(params, ctx):
        rid = params["run_id"]
        ledger = PostgresLedger(DSN, workflow_run_id=rid)
        try:
            return DurableHandler(
                ConcurrentAbsurdCtx(ctx), _domain(), ledger=ledger, contract=Contract.V1
            ).run(lambda: _decision_wf(rid))
        finally:
            ledger.close()

    task_id = app.spawn(name, {"run_id": run_id, CONTRACT_PARAM: Contract.V1.value})
    # A generous drain budget on purpose: the Absurd `default` queue is SHARED across the durable
    # test lane, so `work_batch` may claim another test's task before ours (the justfile's
    # one-queue note). Budget for that rather than assume an empty queue.
    for _ in range(48):
        snap = app.fetch_task_result(task_id)
        if snap is not None and snap.state in ("completed", "failed"):
            break
        app.work_batch()
    snap = app.fetch_task_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    return task_id


def test_the_seed_reader_keeps_what_the_measured_prefix_reader_drops(app, conn):
    """Why two readers. `export_measured_prefix` answers "what did the METER see", so it drops
    every non-`Step` row — correct for replaying a budget trip, and fatal for a fork seed, whose
    prefix contains the ledger appends the child must not re-commit live."""
    rid = f"s{uuid.uuid4().hex[:8]}"
    task_id = _run(app, rid)

    seed_keys = keys(read_absurd_task(conn, task_id))
    measured, _ = export_measured_prefix(conn, task_id, rid)

    assert seed_keys == (
        "step:extract",
        "step;tool:lookup",
        f"ledger;{rid}:extracted",
        f"ledger;{rid}:reviewed",
    )
    assert [e.key.stored() for e in measured] == ["step:extract", "step;tool:lookup"]  # Steps only
    assert not any(k.stored().startswith("ledger;") for k in (e.key for e in measured))


def test_the_seed_reader_carries_raw_state_a_fork_can_seed_from(app, conn):
    """The reader's product, fed to its consumer: `fork_seed` cuts the prefix at `through` and the
    values are RAW checkpoint state — a v1 metered op's whole `{result, usage}` envelope, undecoded
    (the L1/F-2 placement: the driver decodes at the op, a reader that stripped usage would rob the
    meter). This is what makes a durable fork on the deployed engine possible at all."""
    rid = f"s{uuid.uuid4().hex[:8]}"
    task_id = _run(app, rid)

    checkpoints = read_absurd_task(conn, task_id)
    seed = fork_seed(checkpoints, through=f"ledger;{rid}:extracted")

    assert sorted(k.stored() for k in seed) == sorted(
        ["step:extract", "step;tool:lookup", f"ledger;{rid}:extracted"]
    )
    assert f"ledger;{rid}:reviewed" not in seed  # the tail is the fork's to re-decide
    envelope = seed[Key.parse("step:extract")]
    assert envelope["result"] == "ans"  # the value the workflow saw...
    assert envelope["usage"]["cost"] == 0.001  # ...still wrapped in the usage the meter needs
    assert (
        seed[Key.parse("step;tool:lookup")] == 7
    )  # a bare CallTool value: no envelope, by op class


def test_a_checkpoint_timestamp_tie_is_refused_not_guessed(conn):
    """`c_{queue}` has no ordinal, so two rows sharing `updated_at` are genuinely unordered. A
    guess is not a small error: `fork_seed` cuts at `through`, so a swapped pair around the cut
    seeds a tail op or drops a prefix one — silently. The written cause is a pinned clock
    (`absurd.fake_now`), and this reproduces it by writing the tie directly."""
    task_id = uuid.uuid4()  # a UUID object: psycopg adapts it, and the reader takes one
    conn.execute(
        t"INSERT INTO absurd.c_default (task_id, checkpoint_name, state, status, updated_at) "
        t"VALUES ({task_id}::uuid, 'a', '1'::jsonb, 'committed', '2026-07-25 12:00:00+00'), "
        t"       ({task_id}::uuid, 'b', '2'::jsonb, 'committed', '2026-07-25 12:00:00+00')"
    )
    try:
        with pytest.raises(AmbiguousCheckpointOrder) as caught:
            read_absurd_task(conn, task_id)
        assert "'a'" in str(caught.value)
        assert "'b'" in str(caught.value)
        assert "fake_now" in str(caught.value)  # the message names the usual culprit
    finally:
        conn.execute(t"DELETE FROM absurd.c_default WHERE task_id = {task_id}::uuid")


def test_distinct_timestamps_read_in_clock_order(conn):
    """The listing follows `updated_at`, whatever order the rows were inserted in, so the refusal
    above is about the tie."""
    task_id = uuid.uuid4()  # a UUID object: psycopg adapts it, and the reader takes one
    conn.execute(
        t"INSERT INTO absurd.c_default (task_id, checkpoint_name, state, status, updated_at) "
        t"VALUES ({task_id}::uuid, 'second', '2'::jsonb, 'committed', '2026-07-25 12:00:01+00'), "
        t"       ({task_id}::uuid, 'first', '1'::jsonb, 'committed', '2026-07-25 12:00:00+00')"
    )
    try:
        assert keys(read_absurd_task(conn, task_id)) == ("first", "second")
    finally:
        conn.execute(t"DELETE FROM absurd.c_default WHERE task_id = {task_id}::uuid")


def test_both_readers_refuse_an_ambiguous_commit_order(conn):
    """`c_{queue}` has no ordinal, so `updated_at` is the only commit-order proxy and a tie is
    genuinely unordered. `read_absurd_task` refused a tie; `export_measured_prefix` ordered by
    `updated_at` alone and returned whichever order the row-set happened to have — and
    `measured_drive` consumes that prefix POSITIONALLY, so a swapped pair replays one op's value
    at another op's position. One guard now, used by both."""
    import uuid as _uuid

    from effective.bridge_absurd import _refuse_ambiguous_order

    tid = _uuid.uuid4()
    tied = [("ask0", None, 1.0), ("ask1", None, 1.0)]
    ordered = [("ask0", None, 1.0), ("ask1", None, 2.0)]

    for stake in ("a fork seed cut between them", "the positionally-indexed prefix"):
        with pytest.raises(AmbiguousCheckpointOrder, match="unrecoverable"):
            _refuse_ambiguous_order(tied, tid, "c_default", stake)
        _refuse_ambiguous_order(ordered, tid, "c_default", stake)  # distinct stamps: fine


def test_the_measured_reader_actually_calls_the_guard(app, conn):
    """Through `export_measured_prefix`, not the helper — testing the helper alone left the
    WIRING unpinned, and deleting the call from this reader kept every test green. Flatten the
    timestamps of a real run's checkpoints and the reader must refuse."""
    rid = f"tie{uuid.uuid4().hex[:8]}"
    task_id = _run(app, rid)

    export_measured_prefix(conn, task_id, rid)  # distinct stamps: fine
    conn.execute(t"UPDATE absurd.c_default SET updated_at = now() WHERE task_id = {task_id}::uuid")
    with pytest.raises(AmbiguousCheckpointOrder, match="unrecoverable"):
        export_measured_prefix(conn, task_id, rid)


def test_a_clock_stepped_back_across_a_retry_keeps_the_attempts_in_order(conn):
    """Attempt 1 commits `first` at a later instant than attempt 2 commits `second`. The clock
    lists `second` first; the owning run's attempt puts `first` there, so a seed cut at `first`
    holds nothing the retry wrote."""
    from _conformance import AbsurdBackend, FaultInjected, private

    backend = AbsurdBackend()
    first, second = Key.parse("step:first"), Key.parse("step:second")
    stamps = {1: "2040-01-01T00:00:00Z", 2: "2039-01-01T00:00:00Z"}

    def body(params, ctx):
        stepped = ctx._ctx._conn
        stepped.execute(
            "SELECT set_config('absurd.fake_now', %s, false)", (stamps[ctx.attempt.number],)
        )
        try:
            ctx.step(first, lambda: 1)
            if ctx.attempt.number == 1:
                raise FaultInjected("after the first checkpoint")
            ctx.step(second, lambda: 2)
        finally:
            stepped.execute("SELECT set_config('absurd.fake_now', '', false)")
        return "done"

    name = private("clock-stepped-back")
    backend.register_body(name, body)
    task = backend.spawn(name, f"c{uuid.uuid4().hex[:8]}")
    try:
        assert backend.run_until_result(task).state == "completed"
        assert keys(read_absurd_task(conn, task)) == ("step:first", "step:second")
        assert list(fork_seed(read_absurd_task(conn, task), through="step:first")) == [first]
        measured, _ = export_measured_prefix(conn, task, "unused")
        assert [entry.key.stored() for entry in measured] == ["step:first", "step:second"]
    finally:
        backend.close()
