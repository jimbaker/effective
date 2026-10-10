"""Bridge: export a parked durable (Absurd/Postgres) measured run as `(entries, grants)`.

The Absurd-engine sibling of `effective.bridge_sqlite` — same `(entries, grants)` contract,
engine-specific reads. Carries RAW checkpoint state; envelope-ness is NOT decided here, because
the driver decodes at the op (`decode_checkpoint` → `metered_call`) where the op class is known.
A per-engine reader, not a widened `TaskContext` Protocol member.

Differences from SQLite (each verified against `absurd."*_default"`):
  - checkpoints: `c_{queue}` (`task_id`, `checkpoint_name`, `state` **jsonb**, `status`,
    `owner_run_id`, `updated_at`). Committed steps ordered by the owning run's attempt, then
    `updated_at`, the database clock, since there is no rowid/ordinal. `state` is jsonb, so
    `psycopg` returns a Python value (no `json.loads`).
  - grants: `e_{queue}` (`event_name`, `payload` jsonb) — events are GLOBAL per queue (no
    `task_id` column), so run-scoped by the run coordinate of the event name (the run_id in the
    name IS the scope — the same reason grant names embed the run id). SQL narrows to the
    namespace; `names_a_budget_grant` narrows to the run.
  - park name: the current run's `wake_event` (`r_{queue}`, via `t_{queue}.last_attempt_run`).
"""

from itertools import pairwise
from typing import Any
from uuid import UUID

import psycopg

from effective.budget import BUDGET_GRANT_LIKE, Grant, names_a_budget_grant
from effective.checkpoints import (
    ENGINE_INTERNAL,
    Checkpoint,
    is_engine_internal,
    is_step_checkpoint,
    positional_key,
)
from effective.fork import MeteredEntry
from effective.keys import Key

# Queries use PEP 750 t-strings (psycopg ≥ 3.3): `{table:i}` is a safely-quoted identifier,
# `{value}` a bound parameter — the whole query structure stays inline (no split format()+params).


def export_measured_prefix(
    conn: psycopg.Connection[Any], task_id: UUID, run_id: str, *, queue: str = "default"
) -> tuple[list[MeteredEntry], dict[Key, Grant]]:
    """Export a parked measured run's prefix + delivered grants. Feed both to `measured_drive`."""
    c_tbl = f"c_{queue}"
    entries: list[MeteredEntry] = []
    seen: dict[Key, int] = {}
    rows = _committed(conn, task_id, queue)
    steps = [(name, state, at) for name, state, at in rows if is_step_checkpoint(name)]
    _refuse_ambiguous_order(steps, task_id, c_tbl, "the positionally-indexed prefix")
    for name, state, _ in steps:
        # Carry RAW state (jsonb, already a Python value); the driver decodes it at the op
        # (`decode_checkpoint`). No `{result, usage}` shape sniff here: that placement is the
        # driver's, which holds the op class, and a content sniff here would disagree with it.
        # `Key.parse` is the READ boundary: the name arrives as raw text out of a store and
        # re-enters the typed world here, through the one named door for that (a cast accepted
        # until the writer side closes; see `Key.parse`).
        #
        # `split_occurrence` reads the `#k` suffix off the PARSE, so the strip and the render
        # cannot drift. `#0`, `#1` (which has no wire form) and `#02` are not occurrences: they
        # stay in the name, `Key.parse` refuses them, and a counting bug surfaces as a refusal
        # rather than being admitted as a Step.
        key = positional_key(name, seen)  # name#2 → name, refusing a gap
        entries.append(MeteredEntry(key=key, state=state))

    # `payload IS NOT NULL` selects only DELIVERED grants: an Absurd park writes a pending-await
    # MARKER row into e_{queue} with a null payload (SQLite does not — it uses waiting_event),
    # which would otherwise read back as an empty Grant.
    e_tbl = f"e_{queue}"
    grants: dict[Key, Grant] = {}
    grant_rows = conn.execute(
        t"SELECT event_name, payload FROM absurd.{e_tbl:i} "
        t"WHERE event_name LIKE {BUDGET_GRANT_LIKE} AND payload IS NOT NULL"
    ).fetchall()
    for name, payload in grant_rows:
        # `Key.parse` — this is the READ side, where a name arrives as a column value with no
        # `Key` to offer. Writers are typed; readers parse (`Key.__get_pydantic_core_schema__`).
        # The run is selected by PARSING, not by a glued `LIKE`: SQL narrows to the namespace,
        # the grammar narrows to the run.
        if not names_a_budget_grant(key := Key.parse(name), run_id):
            continue
        grants[key] = Grant.model_validate(payload or {})
    return entries, grants


def _committed(
    conn: psycopg.Connection[Any], task_id: UUID, queue: str
) -> list[tuple[str, Any, tuple[int, Any]]]:
    """A task's committed checkpoints as `(name, state, (attempt, updated_at))`, in that order.

    An attempt starts after the one before it ends, so the owning run's attempt orders commits
    across attempts whatever the clock did. Within one attempt the clock is all there is."""
    c_tbl, r_tbl = f"c_{queue}", f"r_{queue}"
    rows = conn.execute(
        t"SELECT c.checkpoint_name, c.state, coalesce(r.attempt, 0), c.updated_at "
        t"FROM absurd.{c_tbl:i} c LEFT JOIN absurd.{r_tbl:i} r ON r.run_id = c.owner_run_id "
        t"WHERE c.task_id = {task_id}::uuid AND c.status = 'committed' "
        t"ORDER BY coalesce(r.attempt, 0), c.updated_at, c.checkpoint_name"
    ).fetchall()
    return [(name, state, (attempt, at)) for name, state, attempt, at in rows]


class AmbiguousCheckpointOrder(RuntimeError):
    """Two of one attempt's checkpoints carry the SAME `updated_at`, so their commit order is not
    recoverable from the row — and on this engine the row is all there is.

    `c_{queue}` has no ordinal: `(task_id, checkpoint_name)` is the primary key and `updated_at`
    (a `clock_timestamp()` default) is the only commit-order proxy within an attempt. A tie is
    therefore genuinely unordered, and guessing is not a small error here — `fork_seed` cuts the
    prefix at `through`, so a swapped pair around that cut seeds a tail op or drops a prefix one,
    which is the exact silent-corruption class `SeedingCtx`'s four-arm match exists to make loud.

    The usual cause is a pinned clock: `absurd.fake_now` freezes `absurd.current_time()`, so every
    checkpoint in the run shares one timestamp. Read a run recorded on the real clock, or order the
    sequence from a source that has an ordinal (an in-process trace, the ledger's `seq`)."""


def _refuse_ambiguous_order(
    rows: list[tuple[str, Any, Any]], task_id: UUID, table: str, stake: str
) -> None:
    """Refuse a tie in `(attempt, updated_at)`, the only commit-order proxy this engine has.

    Both readers of `c_{queue}` need it and only one had it: `read_absurd_task` ordered with
    `checkpoint_name` as a tiebreak and raised on a genuine tie, while `export_measured_prefix`
    ordered by `updated_at` alone and returned whichever order the row-set happened to have.
    `measured_drive` consumes that prefix POSITIONALLY, so a swapped pair replays one op's
    recorded value at another op's position — the same silent-corruption class, one reader over.

    The usual cause is a pinned clock (`absurd.fake_now`), which gives every checkpoint in a run
    one timestamp."""
    for (left, _, at), (right, _, next_at) in pairwise(rows):
        if at == next_at:
            raise AmbiguousCheckpointOrder(
                f"checkpoints {left!r} and {right!r} of task {task_id} share "
                f"(attempt, updated_at)={at} in absurd.{table}, which has no ordinal — their "
                f"commit order is unrecoverable, and "
                f"{stake} would be silently wrong. Is `absurd.fake_now` set?"
            )


def read_absurd_task(
    conn: psycopg.Connection[Any],
    task_id: UUID,
    *,
    queue: str = "default",
    exclude: tuple[str, ...] = ENGINE_INTERNAL,
) -> tuple[Checkpoint, ...]:
    """Every committed checkpoint of ONE Absurd task, in the order of the store's clock — the 0↔N
    half of the reader pair (`effective.checkpoints.read_sqlite_task` is the 0↔1 half).

    **This is the seed reader, and it is deliberately NOT `export_measured_prefix`.** That one
    exists for a *measured prefix*, so it drops every non-`Step` row (`is_step_checkpoint`) and
    normalizes
    `name#2` → `name` because `measured_drive` keys positionally. Both moves are wrong for a fork
    seed: `fork_seed` needs the `ledger:` and `artifact:` keys (a fork's prefix contains ledger
    appends, and a missing seed key makes `SeedingCtx` refuse a correct fork), and `SeedingCtx` is
    occurrence-aware, so it looks up `name#2` verbatim. Its absence is why Stage 4 could not run on
    the deployed engine at all — SQLite-only would be a green pass, not a port.

    Engine differences, each verified against `absurd."*_default"` and pinned cross-engine:

    - **order.** SQLite has `rowid`; `c_{queue}` has no ordinal at all, so the listing is by the
      owning run's attempt, then `updated_at`, the database's `clock_timestamp()`. Across attempts
      that is commit order; within one it is commit order while the clock moves forward. A tie
      raises `AmbiguousCheckpointOrder`; a clock stepped back between two commits of one attempt
      lists them reversed, and nothing in the row can tell.
    - **status.** Only `committed` rows are the durable record; a `pending` row is a step in
      flight. SQLite writes only committed rows, so the filter has no SQLite counterpart.
    - **state.** `jsonb`, so `psycopg` hands back a Python value already — no `json.loads`, where
      SQLite's TEXT column needs one. Same `Checkpoint.state` contract from both.
    - **engine bookkeeping.** Absurd persists `$awaitEvent:` freezes and `sleep_until` wake times
      as checkpoints; SQLite has no such rows. `ENGINE_INTERNAL` excludes them on both sides, which
      is what makes the two key sequences comparable at all. `exclude` names which projection you
      want: the SEED's by default, `exclude=()` for a VIEW that must show the pending await.
    """
    c_tbl = f"c_{queue}"
    rows = _committed(conn, task_id, queue)
    kept = [(name, state, at) for name, state, at in rows if not is_engine_internal(name, exclude)]
    _refuse_ambiguous_order(kept, task_id, c_tbl, "a fork seed cut between them")
    # `Key.parse` — the Absurd read boundary, the twin of `read_sqlite_conn`'s.
    return tuple(Checkpoint(Key.parse(name), state) for name, state, _ in kept)


def end_attempts_at_this_run(
    conn: psycopg.Connection[Any], task_id: str, run_id: str, *, queue: str = "default"
) -> bool:
    """Lower a task's `max_attempts` to the attempt `run_id` runs as, so `fail_run` retries it no
    further, and say whether `run_id` is still the task's latest run. A run whose lease expired
    changes nothing: it must not spend the attempts of the successor that replaced it."""
    t_tbl = f"t_{queue}"
    return (
        conn.execute(
            t"UPDATE absurd.{t_tbl:i} SET max_attempts = LEAST(max_attempts, attempts) "
            t"WHERE task_id = {task_id}::uuid AND last_attempt_run = {run_id}::uuid "
            t"RETURNING task_id"
        ).fetchone()
        is not None
    )


def park_name(
    conn: psycopg.Connection[Any], task_id: UUID, *, queue: str = "default"
) -> str | None:
    """The pending measured park's event name — the current run's `wake_event` (`r_{queue}`),
    the Absurd analog of SQLite's `tasks.waiting_event`. `None` if the task is not parked.

    The single-task read; `effective.parked.read_absurd_parked` is the same join across the whole
    queue."""
    r_tbl, t_tbl = f"r_{queue}", f"t_{queue}"
    row = conn.execute(
        t"SELECT r.wake_event FROM absurd.{r_tbl:i} r "
        t"JOIN absurd.{t_tbl:i} t ON t.last_attempt_run = r.run_id "
        t"WHERE t.task_id = {task_id}::uuid"
    ).fetchone()
    return row[0] if row is not None else None
