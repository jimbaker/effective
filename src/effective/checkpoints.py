"""Where ONE run has been: reading the checkpoint bookkeeper.

Sibling to `effective.parked`, which answers *where the fleet is waiting right now*; this module
answers *where a run has been*. Both are read-only projections over engine tables, and an
interactive surface composes both: a fleet list from `parked`, a run graph from
`keys(read_sqlite_task(...))` fed to `effective.graphview.from_keys`.

Sibling also to `effective.ledger`, the sharper pairing. Of the two durable bookkeepers,
*checkpoints* are disposable execution state, one row per committed op, and the *ledger* is the
canonical append-only record. One module each, so which bookkeeper a caller is reading is a fact
about the import.

The 0↔1 half of the task reader is here; the 0↔N half is `effective.bridge_absurd`. The two are
kept in lockstep by the conformance key-sequence case.

Reads recorded state, imports no domain package, and holds no policy about what any key
*means*.
"""

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from effective.engines.sqlite import connect
from effective.handlers.base import TraceEntry
from effective.keys import Key, unframed
from effective.keys.grammar import TAG_SEPARATOR, TERM_SEPARATOR, split_occurrence
from effective.sql import bind

# --- what a recorded run looks like, whatever engine wrote it ----------------------------


@dataclass(frozen=True)
class Checkpoint:
    """One committed checkpoint: its op key and its state, as a **Python value**.

    Two decodes, and only one belongs to a reader. JSON *deserialization* (the storage string ->
    a Python value) DOES belong here, so every source yields the same kind of thing: Absurd's jsonb
    is a Python value already, SQLite's `read_sqlite_task` `json.loads`es its TEXT column, and
    `from_trace` passes the live result through. Envelope *decoding* (splitting a metered op's
    `{result, usage}` value into `(result, usage)`) does NOT: that stays the driver's job at the
    op, which knows the op class (a reader that stripped usage here would rob the meter). So
    `state` is the full checkpoint value (for a metered op, the whole `{result, usage}` envelope
    as a dict), deserialized but not envelope-decoded. The keys projection ignores it; the fork's
    seed reader needs it, deserialized, since a raw JSON string would double-encode when
    re-committed through `ctx.step`."""

    key: Key
    state: Any = None


def keys(checkpoints: Iterable[Checkpoint]) -> tuple[str, ...]:
    """The op-key sequence — the alphabet both projections align over.

    **This is the PROJECTION exit, and it is deliberately the only one on this path.**
    `Checkpoint.key` is a `Key`; a projection (`graphview`, the SVG, a Mermaid label) works in
    text and does real string work on it — stripping branch prefixes, folding occurrence
    suffixes. Flattening once, here, in a function whose whole job is 'give me the alphabet',
    keeps that from becoming a `.display()` sprinkled through the projection layer. Every
    consumer downstream of this call is a projection by construction."""
    return tuple(c.key.display() for c in checkpoints)


ENGINE_INTERNAL: tuple[str, ...] = (
    "$awaitEvent:",
    "$awaitTaskResult:",
    "sleep:",
    ";wake-race:",
)
"""Checkpoint names an ENGINE writes for its own SUSPENSION bookkeeping, rather than to commit an
op's value through `ctx.step`. Excluded by every task reader here, on both engines.

The rule is structural — *not written by `ctx.step`* — and this is its enumeration on Absurd,
which is the only engine that has any: the SDK freezes a delivered payload as `$awaitEvent:{name}`
(and `$awaitTaskResult:{id}`), and every `sleep_until` writes a wake-time checkpoint under the name
its caller passes — `sleep:{n}` for a real `SleepUntil` (the ordinal the handler's walk assigns,
`keys.FramePosition`), and a `gather:…;wake-race:…` name for a `repark`, whose two shapes are
composed by `handlers.durable.wake_race_on_event` / `wake_race_at_time` and are stated THERE, so
a marker list here cannot drift apart from the producer it describes (see `is_engine_internal`).
**SQLite has none of these by construction**: its awaits live in `tasks.waiting_event` + the
`events` table, and its `sleep_until`/`repark` write nothing at all.

Two things depend on the exclusion. **Parity:** without it the same workflow projects to a
DIFFERENT key sequence per engine, so a cross-engine alignment (`~_H`) or a fork seeded on one
engine and compared on the other is measuring the engine, not the run. **Forkability:** none of
these rows is ever re-yielded through `ctx.step`, so a seeded one could never be consumed —
`SeedingCtx.unconsumed()` would be non-empty and `run_fork` would refuse a fork that is in fact
correct.

Enumerations drift; this one is held by `test_conformance.py`'s key-sequence parity case, which
fails loudly the moment an engine grows a checkpoint shape its sibling has no counterpart for."""


def is_engine_internal(name: str, markers: tuple[str, ...] = ENGINE_INTERNAL) -> bool:
    """Whether a checkpoint name is engine suspension bookkeeping (see `ENGINE_INTERNAL`).

    Prefix OR infix: the SDK's own names are prefixes, but `repark`'s wake-time checkpoint is a
    `gather:`-prefixed name whose *branch step* siblings must stay in — so the wake-race marker is
    matched where it actually sits, mid-name.

    **`unframed` first, for the prefix markers, exactly as `is_step_checkpoint` does.** A sleep
    inside a `scoped(...)` reads `rec:0;sleep:0`, which does not START with its tag — so a bare
    `startswith` would call it NOT engine-internal while calling its unscoped twin `sleep:0`
    internal. A frame-blind predicate answers differently for the same op depending on where it
    was placed, which is the one thing a classifier here may not do.

    Three consumers would have taken the leak: checkpoint reads; cross-engine key-sequence parity,
    where an unfiltered scoped sleep makes the two engines look divergent; and a fork seed, where
    `SeedingCtx.unconsumed()` would be non-empty and `run_fork` would refuse a correct fork.

    The infix marker (`;wake-race:`) stays on the RAW name: it is already matched mid-string, so
    a frame in front of it changes nothing, and unframing first would only narrow what it sees.

    **The partition asks whether a marker OPENS WITH A SEPARATOR — any of the grammar's own —
    and not whether it opens with one particular character.** A test against a single literal is a
    guess about the marker's KIND made from its punctuation, and it misclassifies the moment a
    separator moves: an infix read as a PREFIX is never matched by `startswith`, so the marker
    silently stops being engine-internal and leaks into all three consumers above, `run_fork`
    refusing a correct fork among them. Asking about the separators is at least a question the
    grammar can answer."""
    infixes = tuple(m for m in markers if m.startswith((TERM_SEPARATOR, TAG_SEPARATOR)))
    prefixes = tuple(m for m in markers if m not in infixes)
    canonical = unframed(name, tags=prefixes) if prefixes else name
    return canonical.startswith(prefixes) or any(m in name for m in infixes)


NON_STEP = ("ledger;", "artifact:", "event;", "sleep:", "gather:", "$awaitEvent:", "monitor:")
"""The namespaces whose checkpoints do NOT commit a `Step`'s value — the key-prefix twin of
`to_step_index`'s op-type projection.

**Its overlap with `ENGINE_INTERNAL` is not redundancy:** that one asks *did the engine write this
row for its own suspension bookkeeping* (engine-specific — SQLite writes none of them), this one
asks *does this row commit a Step's value* (engine-independent, and true of `ledger:` rows that
are emphatically not bookkeeping). `sleep:` and `$awaitEvent:` satisfy both; neither list subsumes
the other."""


class SparsePrefix(ValueError):
    """A step prefix holds an occurrence whose earlier occurrences it lacks: a layer refused the
    earlier asks, which leave no checkpoint. A positional replay of it would hand a later ask's
    value to an earlier ask, so it is refused."""


def positional_key(name: str, seen: dict[Key, int]) -> Key:
    """A step checkpoint's name without its occurrence, for a replay that indexes by position.

    `seen` counts the occurrences read so far, per name, in commit order; an occurrence that is
    not the next one raises `SparsePrefix`."""
    base, occurrence = split_occurrence(name)
    key = Key.parse(base)
    count = seen.get(key, 0) + 1
    if (occurrence or 1) != count:
        raise SparsePrefix(
            f"{name!r} is occurrence {occurrence or 1} of {base!r}, and the prefix holds "
            f"{count - 1} before it: an earlier ask was refused and left no checkpoint, so a "
            "positional replay would bind the wrong answer"
        )
    seen[key] = count
    return key


def is_step_checkpoint(name: str, markers: tuple[str, ...] = NON_STEP) -> bool:
    """Whether a checkpoint commits a `Step`'s value — one spelling for both bridges and the pin.

    `unframed` first: a checkpoint inside a `scoped(...)` reads `rec:0;ledger:…`, which does not
    START with its tag, so a bare `startswith` calls it a Step. That defect shipped twice."""
    return not unframed(name, tags=markers).startswith(markers)


def read_sqlite_task(
    path: str | Path, task_id: UUID, *, exclude: tuple[str, ...] = ENGINE_INTERNAL
) -> tuple[Checkpoint, ...]:
    """Every committed checkpoint of ONE SQLite-engine task, in commit order.

    The 0↔1 half of the reader pair; `effective.bridge_absurd.read_absurd_task` is the 0↔N half.
    The signatures differ where the engines do — a file path here
    (it opens its own read-only connection), a live `conn` there — and agree where the contract is:
    `(task_id) -> tuple[Checkpoint, ...]`, unfiltered except for `ENGINE_INTERNAL`, states
    deserialized, duplicate `name#k` suffixes PRESERVED.

    `task_id` is **required**. `Checkpoint` carries no task id, so a read across every task would
    interleave their checkpoints in rowid order, and `fork_seed` (which returns at the FIRST match
    for its `through` key) would silently describe whichever lineage came first. N children
    forked off one base in ONE store make lineage mixing the steady state. A required parameter
    makes a mixed-lineage read unrepresentable.

    Deliberately UNFILTERED, unlike `bridge_sqlite.export_measured_prefix`, whose `NON_STEP`
    filter is right for a *measured prefix* (the trip replays Steps) and wrong for a *key
    sequence* (where a ledger append is part of the shape). The filter belongs to the caller.

    `state` is JSON-deserialized to a Python value (the SQLite `state` column is a `json.dumps`
    string), so this reader has parity with Absurd's jsonb reader and with `from_trace` — every
    `Checkpoint.state` is a Python value regardless of source. Envelope decoding stays the driver's
    (see `Checkpoint`)."""
    conn = connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        return read_sqlite_conn(conn, task_id, exclude=exclude)
    finally:
        conn.close()


def read_sqlite_conn(
    conn: sqlite3.Connection, task_id: UUID, *, exclude: tuple[str, ...] = ENGINE_INTERNAL
) -> tuple[Checkpoint, ...]:
    """`read_sqlite_task` over an ALREADY-OPEN connection — the same read, minus the file.

    An `:memory:` engine has no path to reopen (its store dies with the connection), so the
    conformance harness and any in-process caller need this entry point; `read_sqlite_task` is it
    plus a read-only `file:` open.

    **`exclude` names WHICH PROJECTION you want, and the default is the SEED's.** Dropping the
    engine's suspension bookkeeping is right for a fork (nothing replays an await through
    `ctx.step`, so a seeded one would sit in `unconsumed()` forever) and wrong for a VIEW, where
    the await is the node a user is waiting on. Pass `exclude=()` for the raw sequence. It is a
    parameter because a projection assumed to be the other one mis-renders: a run parked on
    `review:r1` would project to two nodes and no await."""
    rows = conn.execute(
        *bind(t"SELECT name, state FROM checkpoints WHERE task_id = {task_id} ORDER BY rowid")
    )
    return tuple(
        # `Key.parse` — the SQLite read boundary. The name is text in the column and becomes
        # an identity here, at the one named door, rather than by being assumed to be one.
        Checkpoint(Key.parse(name), json.loads(state) if state is not None else None)
        for name, state in rows
        if not is_engine_internal(name, exclude)
    )


def from_trace(trace: Sequence[TraceEntry]) -> tuple[Checkpoint, ...]:
    """An in-process recording/replay trace as the same shape."""
    return tuple(Checkpoint(entry.key, entry.result) for entry in trace)


__all__ = [
    "ENGINE_INTERNAL",
    "NON_STEP",
    "Checkpoint",
    "from_trace",
    "is_engine_internal",
    "is_step_checkpoint",
    "keys",
    "read_sqlite_conn",
    "read_sqlite_task",
]
