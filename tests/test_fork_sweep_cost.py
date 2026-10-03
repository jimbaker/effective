"""What a seeded prefix COSTS at sweep scale.

A fork's prefix is seeded by re-committing the base's recorded values through the child's
own `ctx.step`, which is what makes the counterfactual's tail durable by construction (a crash
mid-tail replays the prefix for free, from the CHILD, never re-reading the base). The bill for
that is storage: every child carries its own copy of the prefix.

This pins the SHAPE of that bill rather than a single number, because the shape is what a sweep
planner needs: checkpoint rows are **exactly `forks x prefix_len`**, and stored bytes are linear
in both. A regression that made seeding quadratic (say, by re-seeding a child from every sibling)
would break the row assertion immediately; a regression that fattened each seeded row would break
the bytes-per-row bound.

Measured here on SQLite with a 512-byte payload per prefix op (`just test-core`, 2026-07-25):

| prefix ops | forks | child ckpt rows | child state bytes | bytes/row |
|---|---|---|---|---|
| 2 | 1 | 2 | 1,084 | 542 |
| 2 | 4 | 8 | 4,336 | 542 |
| 4 | 4 | 16 | 8,672 | 542 |
| 8 | 4 | 32 | 17,344 | 542 |

So the rule of thumb for planning a sweep: **~540 bytes x prefix_len x forks** for a 512-byte
payload, i.e. the stored copy is the payload plus ~6% JSON overhead, times both factors. A
1,000-op prefix swept 20 ways is ~11 MB, which is nothing. The same sweep over a prefix carrying
megabyte tool payloads is not, and the lever is `through` (fork LATER, seed less), never the
fan-out width: the width is what the sweep is for.

**This file measures the bill; the correctness the bill BUYS is pinned elsewhere.** A seeded copy
that were lossy would be cheaper and wrong, so the numbers above only mean something alongside
"the copy is faithful", which is `test_a_seeded_prefix_value_survives_this_ENGINE_s_own_store` in
`test_conformance.py`. That pin runs on every engine because the lossy step happens in
SERIALIZATION and the engines serialize differently in kind (SQLite TEXT, Absurd `jsonb`). The
cost numbers here are SQLite-measured: they are a measurement of one engine's storage, not a
cross-engine property.
"""

from uuid import UUID

import pytest
from test_fork_sweep import new_message_id

from effective.api import append_ledger, ask_llm, await_event
from effective.checkpoints import read_sqlite_conn
from effective.fork import fork_seed, run_fork
from effective.handlers.absurd import DurableHandler, fork_event_name
from effective.keys import Key, Segment, compose_key
from effective.ops import LedgerRow
from effective.sql import bind
from effective.sqlite import SqliteApp, SqliteLedger

PAYLOAD = "x" * 512  # a stand-in for a real extraction result, so bytes mean something


def _wf(prefix_ops: int):
    """`decision_wf`'s shape with a tunable prefix — and the same naming rule: every authored
    name is scoped on the run's SUBJECT (fork-stable), never on its run id."""

    def workflow(message_id: str):
        for i in range(prefix_ops):
            yield from ask_llm(f"extract{i}", [], dict)
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

    return workflow


class _Dom:
    def run(self, op):
        return {"amount": "5.00", "blob": PAYLOAD}

    def run_metered(self, op):
        from effective.cost import Usage

        return self.run(op), Usage()


def _seeded_cost(app: SqliteApp, task_id: UUID) -> tuple[int, int]:
    """(rows, bytes of `state`) for one child's SEEDED prefix rows — the copy it pays for."""
    rows = app.conn.execute(
        *bind(
            t"SELECT state FROM checkpoints WHERE task_id={task_id} AND name LIKE 'step:extract%'"
        )
    ).fetchall()
    return len(rows), sum(len(state or "") for (state,) in rows)


def _run_sweep(app: SqliteApp, db: str, prefix_ops: int, forks: int) -> tuple[int, int]:
    workflow = _wf(prefix_ops)
    message_id = new_message_id()
    dom = _Dom()

    @app.register_task("base")
    def base_task(params, ctx):
        ledger = SqliteLedger(app.conn, params["run_id"], app.write_lock)
        return DurableHandler(ctx, dom, ledger=ledger).run(lambda: workflow(message_id))

    base_id = app.spawn("base", {"run_id": "r-base"})
    app.run_until_result(base_id)
    app.emit_event(f"review:{message_id}", {"decision": "reject"})
    app.run_until_result(base_id)

    # `through` is the LAST op before the fork point, not the last one you feel like copying:
    # everything between `through` and the fork point would run LIVE inside the free prefix, and
    # `SeedingCtx` refuses it. So the ledger row is seeded too: the lever moves the whole
    # boundary and skips no op inside it.
    seed = fork_seed(read_sqlite_conn(app.conn, base_id), through=f"ledger;extracted:{message_id}")
    assert len(seed) == prefix_ops + 1

    total_rows = total_bytes = 0
    for i in range(forks):
        rid = f"fk{i}"

        @app.register_task(f"fork-{i}")
        def fork_task(params, ctx, _rid=rid):
            hyp = SqliteLedger(app.conn, params["run_id"], app.write_lock, hypothetical=True)
            return run_fork(
                ctx,
                lambda: workflow(message_id),  # the child inherits the BASE's message id
                child_run_id=params["run_id"],
                seed=seed,
                hypothetical_ledger=hyp,
                domain=dom,
                forked_from="r-base",
                forked_at_event=f"extracted:{message_id}",
                fork_point=compose_key(t"review:{Segment(message_id)}"),
                delta={"decision": "approve"},
            )

        fid = app.spawn(f"fork-{i}", {"run_id": rid})
        app.run_until_result(fid)
        app.emit_event(
            fork_event_name(rid, Key.parse(f"review:{message_id}")).stored(),
            {"decision": "approve"},
        )
        snap = app.run_until_result(fid)
        assert snap is not None
        assert snap.result == "approve"

        rows, size = _seeded_cost(app, fid)
        assert rows == prefix_ops  # every child carries its OWN copy of the prefix
        total_rows += rows
        total_bytes += size
    return total_rows, total_bytes


@pytest.mark.parametrize(
    ("prefix_ops", "forks", "expected_rows"), [(2, 1, 2), (2, 4, 8), (4, 4, 16), (8, 4, 32)]
)
def test_seeded_prefix_storage_is_exactly_forks_times_prefix(
    tmp_path, prefix_ops, forks, expected_rows, sqlite_app
):
    """Rows are exact, not approximate: `forks x prefix_len`, one checkpoint per seeded op per
    child. Bytes track it linearly at a stable per-row size, so a sweep's storage is predictable
    from two numbers a planner already knows."""
    app = sqlite_app(str(tmp_path / f"cost-{prefix_ops}-{forks}.db"))
    rows, size = _run_sweep(app, str(tmp_path), prefix_ops, forks)

    assert rows == expected_rows
    per_row = size / rows
    assert 500 < per_row < 600, f"{per_row:.0f} bytes/row — the payload is {len(PAYLOAD)}"
    # linear in BOTH factors: doubling either doubles the bill (no cross-term, no re-seeding)
    assert size == pytest.approx(rows * per_row, rel=0.01)


def test_the_lever_is_through_not_the_fan_out_width(tmp_path, sqlite_app):
    """Where to spend the saving. Halving the seeded prefix halves every child's copy, while the
    fan-out width is what the sweep is FOR — so a sweep that costs too much should fork later
    (a longer shared prefix left un-seeded is not an option; `through` names where seeding stops,
    and everything before it must be seeded or `SeedingCtx` refuses)."""
    long_rows, long_bytes = _run_sweep(sqlite_app(str(tmp_path / "long.db")), str(tmp_path), 8, 2)
    short_rows, short_bytes = _run_sweep(
        sqlite_app(str(tmp_path / "short.db")), str(tmp_path), 4, 2
    )

    assert long_rows == 2 * short_rows
    assert long_bytes == pytest.approx(2 * short_bytes, rel=0.02)
