"""Bridge: export a parked durable (SQLite) measured run as `(entries, grants)` for a fork.

Turns a real durable park into the inputs `measured_drive` replays. The SQLite sibling of
`effective.bridge_absurd`; `measured_drive` itself stays DB-free. The bridge carries RAW
checkpoint state and does NOT decide envelope-ness — the durable handler decides that by op class
and contract, and `measured_drive` re-decides the same way at the op.

Two reads:
  - `checkpoints`, ordered by `rowid` (commit order — there is no `recorded_at` column), filtered
    to Step rows, occurrence suffixes split off by the grammar, each with its raw state.
  - delivered `budget-grant:{run_id},{k}` rows from `events` → the `grants` map, without which a
    run past its first grant re-derives the wrong trip. Scoped to the run by PARSING the name, not
    by a `task_id` column — the same shape as the Absurd twin, on the same reasoning: the run
    coordinate lives in the name, which is the only scope a queue-global events table has.

`park_name(conn, task_id)` reads the pending park from `tasks.waiting_event` — the name the probe
grant must key.
"""

import json
import sqlite3
from uuid import UUID

from effective.budget import BUDGET_GRANT_LIKE, Grant, names_a_budget_grant
from effective.checkpoints import is_step_checkpoint
from effective.fork import MeteredEntry
from effective.keys import Key
from effective.keys.grammar import split_occurrence
from effective.parked import read_sqlite_parked_conn
from effective.sql import bind


def export_measured_prefix(
    conn: sqlite3.Connection, task_id: str, run_id: str
) -> tuple[list[MeteredEntry], dict[Key, Grant]]:
    """Export a parked measured run's prefix + delivered grants. Feed both to `measured_drive`.

    Exports the RAW checkpoint state per step, JSON-deserialized from storage and still
    envelope-encoded; `measured_drive` decodes it AT THE OP (`decode_checkpoint`). The bridge
    never sniffs a `{result, usage}` shape: that placement is the driver's, which holds the op
    class, and a content sniff here would disagree with it."""
    entries: list[MeteredEntry] = []
    rows = conn.execute(
        *bind(t"SELECT name, state FROM checkpoints WHERE task_id={task_id} ORDER BY rowid")
    ).fetchall()
    for name, state in rows:
        if not is_step_checkpoint(name):
            continue  # not a Step checkpoint
        raw = json.loads(state) if state is not None else None  # deserialize storage; don't decode
        # `Key.parse` is the READ boundary: the name arrives as raw text out of a store and
        # re-enters the typed world here, through the one named door for that (a cast accepted
        # until the writer side closes; see `Key.parse`).
        #
        # `split_occurrence` reads the `#k` suffix off the PARSE, so the strip and the render
        # cannot drift. `#0`, `#1` (which has no wire form) and `#02` are not occurrences: they
        # stay in the name, `Key.parse` refuses them, and a counting bug surfaces as a refusal
        # rather than being admitted as a Step.
        key = Key.parse(split_occurrence(name)[0])  # name#2 → name (fork keys by raw op_key)
        entries.append(MeteredEntry(key=key, state=raw))

    grants: dict[Key, Grant] = {}
    # `BUDGET_GRANT_LIKE` is in a VALUE position — it is the LIKE *pattern*, which reaches the
    # driver as one bound parameter. The structure (which table, which columns, where the holes
    # are) is the template's. `bind` refuses the other spelling — a hole inside the quoted literal.
    # The pattern carries the namespace and nothing else; the RUN is selected by parsing below.
    #
    # **NO `task_id` predicate, matching the Absurd twin** (`bridge_absurd.py:12-15`: *"the run_id
    # in the name IS the scope"*). This read narrowed by task until the delivery-parity change, and
    # that predicate was never what made it correct: `names_a_budget_grant` parses the run
    # coordinate out of the NAME, which is the narrowing that works on an engine whose events are
    # queue-global. Dropping it here first is deliberate — it is green against the addressed schema
    # too, so the port lands as its own commit rather than riding the delivery change.
    grant_rows = conn.execute(
        *bind(t"SELECT name, payload FROM events WHERE name LIKE {BUDGET_GRANT_LIKE}")
    ).fetchall()
    for name, payload in grant_rows:
        # `Key.parse` — the read side, the same cast the Absurd twin makes for the same reason.
        if not names_a_budget_grant(key := Key.parse(name), run_id):
            continue
        grants[key] = Grant.model_validate(json.loads(payload) if payload is not None else {})
    return entries, grants


def park_name(conn: sqlite3.Connection, task_id: UUID) -> str | None:
    """The pending measured park's event name — the name a probe grant keys, or `None`.

    **A grant is worth emitting, or the name is not worth returning**, which is why this reads
    the answerable-park relation rather than the column. `tasks.waiting_event` names the event
    a park registered under and outlives the park: the claim clears it for every row it takes out
    of `waiting`, so a sleep, a retry or a completion leaves it standing until then. And a waiter
    whose deadline has passed is claimable, which `_deliver` treats as past answering, so a grant
    keyed on one is a write nobody reads.

    Asked of one task where `read_sqlite_parked` asks it of all, so it PROJECTS that relation
    rather than restating its predicate. A second spelling is what went wrong: this reader had
    the state clause and not the deadline, and named parks the relation excludes.

    Reading the whole relation to answer about one task is O(N) in the parks outstanding, measured
    at 0.23 ms for a hundred and 2.4 ms for a thousand. Fine for the one-task question the fork
    probe asks, and a thing to change rather than to call per child across a fan-out."""
    wanted = str(task_id)  # both sides are a `UUID`, so the spellings cannot disagree
    answerable = read_sqlite_parked_conn(conn)
    return next((p.wake_event for p in answerable if str(p.task_id) == wanted), None)
