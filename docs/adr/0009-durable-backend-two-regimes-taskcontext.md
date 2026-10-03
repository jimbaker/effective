# ADR-0009: Durable backend: two regimes over one `TaskContext`; Absurd/Postgres (0↔N) primary, embedded SQLite (0↔1) for laptop and edge

- **Date:** 2026-06-19
- **Status:** Accepted. Built: the `TaskContext` Protocol (`src/effective/handlers/base.py`), the
  generic `DurableHandler` that interprets ops onto it (`src/effective/handlers/absurd.py`), and
  the embedded engine `SqliteApp`, `SqliteTaskContext` and `SqliteLedger`
  (`src/effective/sqlite.py`). One conformance suite (`tests/test_conformance.py` over
  `tests/_conformance.py`) runs the same workflows and assertions on both engines. Unbuilt: a pull
  driver on a Cloudflare Durable Object, and a Turso or libSQL engine.
- **Relates to:** ADR-0002 (the handler stack `TaskContext` types the seam beneath), ADR-0008
  (structural branch keys make `gather` correct on a single writer), ADR-0026 (a cancelled step's
  result is checkpointed like any other).

## Context

The primary durable path is Absurd on Postgres: the handler interprets the op stream onto an
Absurd task context, checkpoints are JSON, and the ledger is a separate append-only table. That
engine earns its `SKIP LOCKED` and MVCC machinery under concurrent multi-worker drains. It also
makes every durable run require a Postgres, which the test suite meets with a digest-pinned
server under rootless Podman (`just pgt-up`).

A second regime is wanted: durable execution from one file with no server, on a laptop without
Postgres or in a Cloudflare Durable Object (single-writer, transactional SQLite storage, idle to
zero, woken by a request or an alarm). The laptop file and the Durable Object are one engine in
two homes.

Two facts make this cheap:

- **The handler touches three ctx methods** (`step`, `await_event`, `sleep_until`), and the
  domain interpreter and ledger writer beside it are already `Protocol`s. "Engine-independent"
  reduces to typing one more seam. Left implicit, it is an assertion no reader and no `ty` can
  check.
- **The SQL surface is modern SQLite; the engine is not.** SQLite 3.45+, which Python 3.14
  bundles, has upsert, `RETURNING`, JSON operators, `JSONB`, partial indexes and triggers. Absurd's
  `plpgsql` (queue, lease, checkpoint, await and emit, run inside Postgres) does not port, and is
  reimplemented in Python. Multi-worker `SKIP LOCKED` does not port either, and the single-writer
  regime has no use for it.

## Decision

### 1. Two regimes, one generator, one conformance suite

| regime | engine | scaling | home | concurrency |
|---|---|---|---|---|
| 0↔N (primary) | Absurd on Postgres | zero (a scale-to-zero managed Postgres) to N workers | a server | MVCC and `SKIP LOCKED` |
| 0↔1 (secondary) | the embedded engine over stdlib `sqlite3` | zero (an idle file) to one worker | a laptop file, a Durable Object | single writer |

The same workflow generator runs on both, and choosing a regime changes nothing in the workflow.
Single-writer is the definition of 0↔1, so it is no limitation there.

### 2. The seam is a `TaskContext` Protocol

`DurableHandler` takes `ctx: TaskContext`. The signatures are small; the durability semantics in
the docstrings are the contract the conformance suite pins:

| member | contract |
|---|---|
| `step(name: Key, thunk, /)` | run `thunk` at most once per `name` and checkpoint its JSON-able result; on replay return the recorded value without running `thunk`. A thunk that returns is checkpointed, a `Cancelled` included; one that raises is not. A crash after the thunk but before commit re-runs it |
| `await_event(name: Key, /)` | suspend durably until an external `emit_event(name, payload)`, and return `payload`. The payload is recorded, so replay re-binds it by name with no captured continuation. Survives worker death and resume on a fresh worker |
| `sleep_until(when, /, *, name: Key)` | a durable timer that pins no worker while parked and resumes at or after `when` across crashes. `name` is the timer's identity, minted above the seam |

Names are `Key`s, never `str`: a checkpoint name and a park name are what replay binds to, so an
f-string there is a `ty` error. Further capabilities (`peek_event`, `repark`, `await_until`,
`peek_step`, `settle`, `step_resolved`, `concurrent_safe`) are optional and discovered by
`getattr`, so a minimal test ctx stays small and a ctx lacking one meets a named refusal.

### 3. The SQLite side is an engine, and there is no SQLite handler

`DurableHandler` is a generic `TaskContext` interpreter, so the SQLite side supplies an engine
(driver, ctx, ledger) that the same handler runs on. It reimplements only the subset of Absurd
that Effective uses:

| piece | implementation |
|---|---|
| claim and lease | single writer, so no `SKIP LOCKED`: a claim selects and updates one claimable task under the engine's write lock, and an expired `claim_expires_at` makes a dead worker's task reclaimable |
| checkpoint store | rows keyed `(task_id, name)`, upserted with `ON CONFLICT ... DO UPDATE` |
| await and emit | a waits table; `await_event` parks, `emit_event` wakes by name, and the recorded payload re-binds on replay |
| ledger | `SqliteLedger`, append-only by a trigger that raises `RAISE(ABORT, ...)` on UPDATE or DELETE, idempotent by `ON CONFLICT(event_id) DO NOTHING` |
| task ids | `uuid7`, generated in Python |
| drive | pull-only like Absurd: `SqliteApp.work_batch` claims one ready task; `run_until_result` drains |

The engine is not Absurd, so it is free of the `absurd.sql` and SDK co-version pin and must pass
the conformance suite on every change. It adds no runtime dependency.

### 4. Turso is a reserved seam

`TaskContext` and the engine seam stay clean, so a libSQL or Turso engine would be a driver swap
whose optimistic concurrency lifts the single-writer ceiling to 0↔N without Postgres. Postgres
already covers 0↔N, so it is a documented seam and no work.

### 5. Testing: conformance across engines, unit tests within one

- **Conformance.** One parameterized suite (`tests/test_conformance.py`) runs on both engines
  through the same handler and pins the §2 semantics: replay returns recorded values without
  re-running thunks; a crash at every op resumes to an identical outcome; a suspending
  `await_event` survives suspend and resume; retry is durable; the ledger refuses UPDATE and
  DELETE. `tests/test_worker_death.py` covers worker death and resume on a fresh worker.
- **Unit.** The engine is plain Python over SQLite, so each piece is tested infra-free against
  `:memory:` or a temporary file (`tests/test_sqlite_engine.py`, `tests/test_sqlite_claim_fence.py`,
  `tests/test_sqlite_concurrency.py`): claim, lease expiry and reclaim, the fence on a stale
  claim, sleep, wake by name, spawn idempotency, and concurrent branches committing each
  checkpoint once.

## Consequences

- The durable suite runs without Podman on the SQLite engine, inside `just test-fast`.
- One typed seam and one small engine, with no new dependency.
- The determinism boundary is untouched. `gather` keys by structural position (ADR-0008), so
  single-writer SQLite serializes commits that never conflict.
- The Postgres ledger (`src/effective/ledger.py`, SQLModel over `JSONB`) and `SqliteLedger` are
  two implementations of one append-only, idempotent contract.

## Open questions

- **The Durable Object pull driver.** Absurd is pull-only; on a laptop the driver is a process
  loop, on a Durable Object an alarm or an inbound request driving one claim. It is the one new
  integration point the 0↔1 regime adds.
- **Python on Cloudflare** is early (Python Workers on Pyodide). The laptop file is the certain
  target; the Durable Object is the generalization still to prove.
